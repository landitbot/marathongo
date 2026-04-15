#ifndef GTSAM_SOLVER_H_
#define GTSAM_SOLVER_H_
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <Eigen/Geometry>
#include <vector>
#include <map>
// 高度约束因子 
// class HeightFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
// private:
//     double measuredHeight_; // 测量值
 
// public:
//     // 构造函数 
//     // key: 变量ID, measuredHeight: 期望的高度, noise: 噪声模型
//     HeightFactor(gtsam::Key key, double measuredHeight, 
//                  const gtsam::noiseModel::Base::shared_ptr& noiseModel) :
//         NoiseModelFactor1<gtsam::Pose3>(noiseModel, key), 
//         measuredHeight_(measuredHeight) {}
 
//     // 计算误差向量 
//     // error = pose.z - measurement
//     gtsam::Vector evaluateError(const gtsam::Pose3& pose, 
//                                 boost::optional<gtsam::Matrix&> H = boost::none) const override {
        
//         if (H) {
//             // 计算雅可比矩阵
//             *H = gtsam::Matrix::Zero(1, 6);
//             (*H)(0, 2) = 1.0;  // dz/dz = 1
//         }
        
//         // 返回误差值 
//         return gtsam::Vector1(pose.z() - measuredHeight_);
//     }
// };
class HeightFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  double measuredHeight_;
 
 public:
  typedef boost::shared_ptr<HeightFactor> shared_ptr;
 
  HeightFactor(gtsam::Key j, double measuredHeight, 
              const gtsam::SharedNoiseModel& model) 
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, j), 
        measuredHeight_(measuredHeight) {}
 
  virtual ~HeightFactor() {}
 
  gtsam::Vector evaluateError(const gtsam::Pose3& q,
                             boost::optional<gtsam::Matrix&> H = boost::none) const {
    if (H) {
      *H = gtsam::Matrix::Zero(1, 6);
      (*H)(0, 2) = 1.0;  // 只有z坐标对误差有贡献
    }
    return (gtsam::Vector(1) << q.z() - measuredHeight_).finished();
  }
 
  virtual gtsam::NonlinearFactor::shared_ptr clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new HeightFactor(*this)));
  }
};

class GTSAMSolver {
public:
    using Ptr = std::shared_ptr<GTSAMSolver>;
    // 新增求解器类型枚举 
    enum SolverType {
        LEVENBERG_MARQUARDT,
        ISAM2
    };

    GTSAMSolver(SolverType type = ISAM2);
    ~GTSAMSolver() = default;
 
    void Clear();
    void Compute(bool has_loop_flag=false);
    
    // 修改为3D节点接口
    void AddNode(int id, const Eigen::Affine3d& pose);
    void SetPriorPose(int id, const Eigen::Affine3d& pose, 
                              const Eigen::Matrix<double,6,6>& noise_sigmas );
    void SetPriorPose(int id, const Eigen::Affine3d& pose, 
                              const gtsam::noiseModel::Base::shared_ptr& noise );
    // 修改为3D约束接口
    void AddConstraint(int sourceId, int targetId, 
                             const Eigen::Affine3d& relative_pose,
                             const Eigen::Matrix<double, 6, 6>& covariance );
    void AddConstraint(int sourceId, int targetId, 
                             const Eigen::Affine3d& relative_pose,
                             const gtsam::noiseModel::Base::shared_ptr& noise );
    void AddConstraint(int sourceId, int targetId, 
                             const gtsam::Pose3& relative_pose,
                             const gtsam::noiseModel::Base::shared_ptr& noise );
    // GPS因子接口
    void AddGpsPositionFactor(
        int id, 
        const Eigen::Vector3d& gps_position,
        const Eigen::Matrix3d& covariance
        );
    void AddGpsPositionFactor(
        int id,
        const Eigen::Vector3d& gps_position,
        const gtsam::noiseModel::Base::shared_ptr& noise
        );
    // [新增] 高度约束接口 
    // id: 节点ID, height: 期望高度, noise_sigma: 高度测量的噪声标准差
    void AddHeightFactor(int id, double height, double noise_sigma);
    // 获取优化后的3D位姿图 
    void getGraph(std::vector<Eigen::Affine3d>& nodes, 
                 std::vector<std::pair<int, int>>& edges);
 
    // 获取优化后的位姿 
    const std::map<int, Eigen::Affine3d>& GetCorrections() const;
    const gtsam::Values& GetCurrentEstimate() const;
    size_t GetNodeNum() const;
    Eigen::Matrix<double, 6, 6> GetPoseCovariance(int id) const;
    // void AddHeightFactorSimple(int id, double height, double noise_sigma) {
    //     auto noise = gtsam::noiseModel::Isotropic::Sigma(1, noise_sigma);
        
    //     // 创建一个选择器，指定只约束 Pose3 的第 5 个维度 (索引从0开始: 0,1,2=旋转, 3,4,5=平移)
    //     // 即只约束 Z 轴
    //     gtsam::Key key = id;
        
    //     // 构造函数: (Key, 维度索引, 测量值, 噪声)
    //     // 这里的维度索引 5 对应 Pose3 的 Z 轴平移
    //     graph_.add(gtsam::PartialPriorFactor<gtsam::Pose3>(key, 5, height, noise));
    // }
 
private:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initialGuess_;
    std::map<int, Eigen::Affine3d> corrections_;
    std::vector<std::pair<int, int>> graphEdges_;
    
    SolverType solverType_;
    std::unique_ptr<gtsam::ISAM2> isam2_;  // ISAM2实例 
    gtsam::Values isamCurrentEstimate_;     // ISAM2的当前估计值 
};
 
#endif // GTSAM_SOLVER_H_