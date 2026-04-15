#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cpptrace/cpptrace.hpp>

#include "osqp/osqp.h"

constexpr int kN = 30;
constexpr double kT = 0.1;

/**
 * J = (xN - xN_ref)^T P (xN - xN_ref) + Sigma{ (xi-xi_ref)^T Q (xi-xi_ref) +
 * ui^T R ui } J = endian cost + state cost + energy cost minJ s.t. x_i+1 = Ax_i
 * + Bu_i u_min <= ui <= u_max x_min <= xi <= x_max
 *
 * => argmin{u}{0.5 * u^T H u + g^T u}
 */

/** 2D Tracking - Differential Wheel Model
 * x=[x y theta]    // theta: right-hand  and  relative to X-Axis
 * u=[v omega]
 */

/// @brief 将Eigen的稀疏矩阵转换成OSQP的矩阵格式
/// @param mat 必须makeCompressed
/// @return OSQP矩阵
OSQPCscMatrix* eigenSparseToOSQP(const Eigen::SparseMatrix<double>& mat) {
  // get martrix shape and non-zero count
  OSQPInt r = (OSQPInt)mat.rows();
  OSQPInt c = (OSQPInt)mat.cols();
  OSQPInt nnz = (OSQPInt)mat.nonZeros();

  // actuall value
  OSQPFloat* x = (OSQPFloat*)malloc(sizeof(OSQPFloat) * nnz);
  // row index
  OSQPInt* i = (OSQPInt*)malloc(sizeof(OSQPInt) * nnz);
  // col offset
  OSQPInt* p = (OSQPInt*)malloc(sizeof(OSQPInt) * (c + 1));

  // copy: Eigen 的内部指针可以通过 valuePtr, innerIndexPtr, outerIndexPtr
  // 直接访问
  for (int k = 0; k < nnz; k++) {
    x[k] = (OSQPFloat)mat.valuePtr()[k];
    i[k] = (OSQPInt)mat.innerIndexPtr()[k];
  }
  for (int k = 0; k < c + 1; k++) {
    p[k] = (OSQPInt)mat.outerIndexPtr()[k];
  }

  auto ret = OSQPCscMatrix_new(r, c, nnz, x, i, p);
  // 最后一个参数设置为 1 表示由 OSQP
  // 结构体在释放时自动释放上述 malloc 的内存
  ret->owned = 1;
  return ret;
}

class MPCController {
 public:
  static constexpr int n_ = 3;  // state dim
  static constexpr int m_ = 2;  // control dim

  using State = Eigen::Vector3d;    // [x, y, theta]
  using Control = Eigen::Vector2d;  // [v, omega]
  using StateTransformMatrix = Eigen::Matrix<double, n_, n_>;
  using ControlTransformMatrix = Eigen::Matrix<double, n_, m_>;

  MPCController(int N, double T) {
    N_ = N;
    T_ = T;
    Q_ = Eigen::Vector3d(10.0, 10.0, 1.0).asDiagonal();  // state weight
    R_ = Eigen::Vector2d(0.1, 0.1).asDiagonal();         // control weight
    P_ = Q_;                                             // endian weight
  }

  ~MPCController() {}

  Eigen::SparseMatrix<double> computeHessianMatrix() {
    // hessian only associalated with P Q R;
    // this can only compute once.
    int numStates = (N_ + 1) * n_;
    int numControls = N_ * m_;
    int numVars = numStates + numControls;

    Eigen::SparseMatrix<double> H(numVars, numVars);
    std::vector<Eigen::Triplet<double>> H_triplets;

    // fill state
    for (size_t i = 0; i < N_ + 1; i++) {
      // common state
      if (i < N_) {
        for (size_t j = 0; j < n_; j++) {
          int idx = i * n_ + j;
          double v = 2.0 * Q_(j, j);
          H_triplets.push_back({idx, idx, v});
        }
      }
      // endian state
      else {
        for (size_t j = 0; j < n_; j++) {
          int idx = i * n_ + j;
          double v = 2.0 * P_(j, j);
          H_triplets.push_back({idx, idx, v});
        }
      }
    }

    // fill control state
    int u_offset = (N_ + 1) * n_;
    for (size_t i = 0; i < N_; i++) {
      for (size_t j = 0; j < m_; j++) {
        int idx = u_offset + i * m_ + j;
        double v = 2.0 * R_(j, j);
        H_triplets.push_back({idx, idx, v});
      }
    }

    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    H.makeCompressed();
    return H;
  }

  Eigen::VectorXd computeGradientVector(const std::vector<State>& X,
                                        const std::vector<State>& x_ref,
                                        const std::vector<Control>& U) {
    // gradient associalates with current state, ref state, current control
    // this needs to be computed every times.

    assert(X.size() == N_ + 1);
    assert(x_ref.size() == N_ + 1);
    assert(U.size() == N_);

    int numStates = (N_ + 1) * n_;
    int numControls = N_ * m_;
    int numVars = numStates + numControls;

    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(numVars);

    for (int i = 0; i < N_ + 1; i++) {
      if (i < N_) {
        gradient.segment<n_>(i * n_) = -2.0 * Q_ * (X[i] - x_ref[i]);
      } else {
        gradient.segment<n_>(i * n_) = -2.0 * P_ * (X[i] - x_ref[i]);
      }
    }

    int u_offset = (N_ + 1) * n_;
    for (int i = 0; i < N_; i++) {
      gradient.segment<m_>(u_offset + i * m_) = -2.0 * R_ * U[i];
    }
    return gradient;
  }

  int computeConstraintMatrix(Eigen::SparseMatrix<double>& A,
                              Eigen::VectorXd& lower, Eigen::VectorXd& upper) {
    // compute size
    int numStates = (N_ + 1) * n_;
    int numControls = N_ * m_;
    int numVars = numStates + numControls;
    int numCons = numStates +  // 动力学约束
                  numVars;     // 变量范围约束

    // resize
    A = Eigen::SparseMatrix<double>(numCons, numVars);
    lower = Eigen::VectorXd::Zero(numCons);
    upper = Eigen::VectorXd::Zero(numCons);

    // compute A matrix
    std::vector<Eigen::Triplet<double>> triplets;

    // 填充动力学约束: x{k+1} = A*x{k} + B*u{k}  =>  x{k+1} - A*x{k} - B*u{k} =
    // 0
    {
      // 初始状态约束
      {
        for (int j = 0; j < n_; ++j) {
          triplets.push_back({j, j, 1.0});
          lower[j] = cur_state_(j);
          upper[j] = cur_state_(j);
        }
      }
      // 后续动力学状态约束
      {
        for (size_t i = 0; i < N_; i++) {
          int row_offset = (i + 1) * n_;  // 跳过前n_行
          int x_i_col_start = i * n_;
          int x_i_next_col_start = (i + 1) * n_;
          int u_i_col_start = numStates + i * m_;

          // 填充I
          for (int r = 0; r < n_; r++)
            triplets.push_back({row_offset + r, x_i_next_col_start + r, 1.0});

          // 填充-A
          for (int r = 0; r < n_; r++)
            for (int c = 0; c < n_; ++c)
              triplets.push_back(
                  {row_offset + r, x_i_col_start + c, -A_(r, c)});

          // 填充-B
          for (int r = 0; r < n_; r++)
            for (int c = 0; c < m_; ++c)
              triplets.push_back(
                  {row_offset + r, u_i_col_start + c, -B_(r, c)});
        }
      }
    }

    // 填充变量约束
    {
      // 状态约束
      int x_offset = numStates;
      for (size_t i = 0; i < N_ + 1; i++) {
        for (size_t j = 0; j < n_; j++) {
          triplets.push_back({x_offset + i * n_ + j, i * n_ + j, 1.0});
          lower[x_offset + i * n_ + j] = min_state_(j);
          upper[x_offset + i * n_ + j] = max_state_(j);
        }
      }

      // 控制量约束
      int u_offset = x_offset + numStates;
      for (size_t i = 0; i < N_; i++) {
        for (size_t j = 0; j < m_; j++) {
          triplets.push_back(
              {u_offset + i * m_ + j, numStates + i * m_ + j, 1.0});
          lower[u_offset + i * m_ + j] = min_control_(j);
          upper[u_offset + i * m_ + j] = max_control_(j);
        }
      }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    return numCons;
  }

  Control computeControl(const State& x_curr, const std::vector<State>& x_ref) {
    // 更新当前状态
    cur_state_ = x_curr;

    // 构造 QP 问题矩阵 (假设这里X和U已经有了初步猜测或者设为0)
    std::vector<State> X_guess(N_ + 1, x_curr);
    std::vector<Control> U_guess(N_, Control::Zero());

    std::cout << "0" << std::endl;
    Eigen::SparseMatrix<double> H = computeHessianMatrix();
    std::cout << "1" << std::endl;
    Eigen::VectorXd g = computeGradientVector(X_guess, x_ref, U_guess);
    std::cout << "2" << std::endl;
    Eigen::SparseMatrix<double> A_cons;
    Eigen::VectorXd l, u;
    auto numCons = computeConstraintMatrix(A_cons, l, u);
    std::cout << "3" << std::endl;

    std::cout << "H:" << std::endl;
    std::cout << "rows: " << H.rows() << std::endl;
    std::cout << "cols: " << H.cols() << std::endl;
    std::cout << "g:" << std::endl;
    std::cout << "rows: " << g.rows() << std::endl;
    std::cout << "cols: " << g.cols() << std::endl;
    std::cout << "A:" << std::endl;
    std::cout << "rows: " << A_cons.rows() << std::endl;
    std::cout << "cols: " << A_cons.cols() << std::endl;

    // 转换为 OSQP 格式
    OSQPCscMatrix* P_osqp = eigenSparseToOSQP(H);
    OSQPCscMatrix* A_osqp = eigenSparseToOSQP(A_cons);

    // 配置 OSQP
    OSQPSettings* settings = OSQPSettings_new();
    settings->alpha = 1.0;      // 这里的alpha设为1.0提高收敛性
    settings->verbose = false;  // 关闭输出，提高速度
    osqp_set_default_settings(settings);

    OSQPInt osqp_num_cons = A_cons.rows();
    OSQPInt osqp_num_dims = H.rows();
    // OSQPInt osqp_num_dims = (N_ + 1) * n_ + N_ * m_;

    // 初始化并求解
    std::cout << "4" << std::endl;
    OSQPSolver* solver = nullptr;
    osqp_setup(&solver, P_osqp, g.data(), A_osqp, l.data(), u.data(),
               osqp_num_cons, osqp_num_dims, settings);
    osqp_solve(solver);

    std::cout << "5" << std::endl;

    // 提取结果 (z = [x0...xN, u0...uN-1])
    // u0 的起始索引是 (N+1)*n_
    int u0_idx = (N_ + 1) * n_;
    Control u_optimal;
    u_optimal << solver->solution->x[u0_idx], solver->solution->x[u0_idx + 1];

    // 释放内存 (防止 ROS 节点内存溢出)
    // 注意：P_osqp 和 A_osqp 在 osqp_cleanup中设置了 owned 也会被释放
    osqp_cleanup(solver);
    if (settings) free(settings);

    return u_optimal;
  }

 private:
  int N_ = 10;       // predict region
  double T_ = 0.05;  // control cycle

  State cur_state_;
  Control cur_control_;

  State min_state_;
  State max_state_;
  Control min_control_;
  Control max_control_;

  Eigen::MatrixXd P_;  // endian cost weight matrix
  Eigen::MatrixXd Q_;  // state cost weight matrix
  Eigen::MatrixXd R_;  // energy cost weight matrix

  StateTransformMatrix A_;
  ControlTransformMatrix B_;
};

std::unique_ptr<MPCController> mpc;

std::shared_ptr<nav_msgs::Odometry> g_robot_odom;
std::mutex g_robot_odom_lock;

std::shared_ptr<nav_msgs::Path> g_tracking_path;
std::mutex g_tracking_path_lock;

void compute();

void handler_robot_odometry(nav_msgs::OdometryConstPtr msg) {
  {
    std::lock_guard<std::mutex> glock(g_robot_odom_lock);
    g_robot_odom = std::make_shared<nav_msgs::Odometry>();
    *g_robot_odom = *msg;
  }
  compute();
}

void handler_tracking_path(nav_msgs::PathConstPtr msg) {
  {
    std::lock_guard<std::mutex> glock(g_tracking_path_lock);
    g_tracking_path = std::make_shared<nav_msgs::Path>();
    *g_tracking_path = *msg;
  }
}

/// @brief
/// @param odom
/// @param path must be aligned with robot odom; must be uniform resolution
/// @param N MPC predict count
bool compute_states(const nav_msgs::Odometry& odom, const nav_msgs::Path& path,
                    int N, MPCController::State& x_curr,
                    std::vector<MPCController::State>& x_ref) {
  Eigen::Vector3d robot_pos(odom.pose.pose.position.x,
                            odom.pose.pose.position.y,
                            odom.pose.pose.position.z);

  Eigen::Quaterniond robot_rot(
      odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
      odom.pose.pose.orientation.y, odom.pose.pose.orientation.z);

  Eigen::Vector3d xaxis = Eigen::Vector3d::UnitX();
  auto front_axis = robot_rot * xaxis;
  double yaw = std::atan2(front_axis.y(), front_axis.x());
  x_curr = MPCController::State(robot_pos.x(), robot_pos.y(), yaw);

  size_t beg_idx = 0;
  {
    double min_dis = 1e17;
    for (size_t i = 0; i < path.poses.size(); i++) {
      double dx = robot_pos.x() - path.poses[i].pose.position.x;
      double dy = robot_pos.y() - path.poses[i].pose.position.y;
      double dis = std::sqrt(dx * dx + dy * dy);
      if (dis < min_dis) {
        min_dis = dis;
        beg_idx = i;
      }
    }
  }

  if ((path.poses.size() - (beg_idx + 1)) < N + 1) {
    return false;
  }

  x_ref.clear();
  x_ref.reserve(N + 1);
  for (size_t i = beg_idx + 1; i < path.poses.size(); i++) {
    if (i - 1 >= N + 1) {
      break;
    }

    auto pt0 = Eigen::Vector3d(path.poses[i - 1].pose.position.x,
                               path.poses[i - 1].pose.position.y,
                               path.poses[i - 1].pose.position.z);

    auto pt1 = Eigen::Vector3d(path.poses[i].pose.position.x,
                               path.poses[i].pose.position.y,
                               path.poses[i].pose.position.z);

    auto vec = pt1 - pt0;
    double yaw = std::atan2(vec.y(), vec.x());
    x_ref.emplace_back(pt0.x(), pt0.y(), yaw);
  }
  return true;
}

void compute() {
  std::shared_ptr<nav_msgs::Odometry> odom;
  std::shared_ptr<nav_msgs::Path> path;
  {
    std::lock_guard<std::mutex> glock(g_robot_odom_lock);
    odom = g_robot_odom;
    if (odom == nullptr) {
      std::cerr << "Odom is nullptr" << std::endl;
      return;
    }
  }

  {
    std::lock_guard<std::mutex> glock(g_tracking_path_lock);
    path = g_tracking_path;
    if (path == nullptr) {
      std::cerr << "Path is nullptr" << std::endl;
      return;
    }
  }

  // assume: the timestamp of odom and path is the same.
  // only be true while using local_planner

  MPCController::State x_curr;
  std::vector<MPCController::State> x_ref;
  if (!compute_states(*odom, *path, kN, x_curr, x_ref)) {
    std::cerr << "failed to ComputeStates();" << std::endl;
  }

  std::cout << "a1" << std::endl;
  auto control = mpc->computeControl(x_curr, x_ref);
  std::cout << "-------------------------------------" << std::endl;
  std::cout << control << std::endl;

  // std::cout << "-------------------------------------" << std::endl;
  // std::cout << "x_curr:" << std::endl;
  // std::cout << x_curr << std::endl;
  // std::cout << "x_ref:" << x_ref.size() << std::endl;
  // for (size_t i = 0; i < x_ref.size(); i++) {
  //   std::cout << "ref: " << i << ": " << x_ref[i] << std::endl;
  // }
}

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "mpc_controller_node");
  ros::NodeHandle nh;

  mpc = std::make_unique<MPCController>(kN, kT);

  auto suber_robot_odom =
      nh.subscribe("/high_frequency_odometry", 1, handler_robot_odometry);

  auto suber_tracking_path =
      nh.subscribe("/local_planner/path", 1, handler_tracking_path);

  ros::spin();
  return 0;
}