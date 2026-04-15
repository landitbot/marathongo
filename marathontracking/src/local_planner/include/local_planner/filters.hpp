#ifndef FILTERS_HPP
#define FILTERS_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <numeric>

class RobotShakeFilter {
 public:
  RobotShakeFilter(int size = 10) : buffer_size_(size) {}

  void push(const Eigen::Quaterniond& q) {
    if (quats_.size() >= buffer_size_) {
      quats_.pop_front();
    }
    quats_.push_back(q.normalized());
  }

  Eigen::Quaterniond getFilteredQuat() {
    if (quats_.empty()) return Eigen::Quaterniond::Identity();
    if (quats_.size() == 1) return quats_.front();

    // 构造矩阵 M = \sum (q_i * q_i^T)
    // 四元数在 Eigen 中以 (x, y, z, w) 存储，但这不影响特征值分解
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();

    for (const auto& q : quats_) {
      Eigen::Vector4d v(q.x(), q.y(), q.z(), q.w());
      // 处理双倍覆盖：确保所有四元数都在同一个半球
      // 虽然特征值分解法对符号不敏感，但累加 M 时方向一致更稳健
      M += v * v.transpose();
    }

    // 这里的 M 是对称矩阵，通过特征值分解找到最大特征值对应的特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(M);
    Eigen::Vector4d average_v =
        solver.eigenvectors().col(3);  // 特征值默认升序排列

    return Eigen::Quaterniond(average_v[3], average_v[0], average_v[1],
                              average_v[2])
        .normalized();
  }

 private:
  std::size_t buffer_size_;
  std::deque<Eigen::Quaterniond> quats_;
};

class Smoother {
 private:
  double kUp_ = 0.3;
  double kDown_ = 0.5;
  double dt_ = 0.1;
  double v_ = 0;
  double last_ts_ = -1;

 public:
  Smoother(double kup, double kdown, double dt)
      : kUp_(kup), kDown_(kdown), dt_(dt) {
    last_ts_ = -1;
  }

  Smoother() {
    kUp_ = 0.6;
    kDown_ = 0.3;
    dt_ = 0.1;
    v_ = 0;
    last_ts_ = -1;
  }

  ~Smoother() {}

  double get_ts() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  double compute(double v) {
    if (last_ts_ < 0) {
      last_ts_ = get_ts();
      v_ = v;
      return v_;
    }
    auto ts = get_ts();
    if (ts - last_ts_ < dt_) {
      return v_;
    }
    if (v >= v_) {
      v_ += kUp_ * (v - v_);
    } else {
      v_ += kDown_ * (v - v_);
    }
    last_ts_ = ts;
    return v_;
  }
};

#endif