#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <initializer_list>
#include <vector>

#include "angles/angles.h"

/// @brief Path sampler
/// 1. stores pre-sampled path using relative coordinates
/// 2. generate pathes with given robot pose
/// 3. coordinate: FLU

struct PathControlInfo {
  double linear_velocity;
  double yaw_error;
  double curvature;
};

namespace pathlib {
class Path {
 public:
  Path() = default;

  Path(const Path& p) { path_ = p.path_; }

  Path(Path&& p) { path_ = std::move(p.path_); }

  Path(const std::vector<double>& points) {
    size_t size = points.size() / 3;
    path_.reserve(size);
    for (size_t i = 0; i < size; i++) {
      double x = points[i * 3 + 0];
      double y = points[i * 3 + 1];
      double z = points[i * 3 + 2];
      path_.emplace_back(x, y, z);
    }
  }

  Path(const std::vector<Eigen::Vector3d>& points) { path_ = points; }

  void operator=(const Path& p) { path_ = p.path_; }

  void operator=(Path&& p) { path_ = std::move(p.path_); }

  ~Path() {}

  void transform(const Eigen::Vector3d& position,
                 const Eigen::Quaterniond& rotation) {
    for (auto& p : path_) {
      p = rotation * p + position;
    }
  }

  const std::vector<Eigen::Vector3d>& getPath() const { return path_; }

  std::vector<Eigen::Vector3d>& getPath() { return path_; }

  void setControlParam(const Eigen::Vector3d& robot_pos,
                       const Eigen::Quaterniond& robot_rot, std::size_t ahead) {
    robot_pos_ = robot_pos;
    robot_rot_ = robot_rot;
    if (ahead < path_.size()) {
      ahead_idx_ = ahead;
    } else {
      ahead_idx_ = path_.size() - 1;
    }
  }

  Eigen::Vector3d getAheadPoint() { return path_[ahead_idx_]; }

  PathControlInfo getControlInfo() {
    PathControlInfo info;
    info.linear_velocity = 4.5;
    info.yaw_error = compute_yaw_error();
    info.curvature = compute_curvature();
    return info;
  }

  double compute_yaw_error() {
    Eigen::Vector3d ahead = path_[ahead_idx_];
    double dx = ahead.x() - robot_pos_.x();
    double dy = ahead.y() - robot_pos_.y();
    const double ahead_yaw = std::atan2(dy, dx);

    Eigen::Vector3d front = robot_rot_ * Eigen::Vector3d::UnitX();
    const double current_yaw = std::atan2(front.y(), front.x());
    double yaw_error =
        angles::shortest_angular_distance(current_yaw, ahead_yaw);
    return yaw_error;
  }

  double compute_curvature(int offset = 0) {
    double curvature = 0;
    const Eigen::Vector3d& p_beg = path_.at(offset);
    const Eigen::Vector3d& p_end = path_.back();
    const Eigen::Vector2d v0(p_end.x() - p_beg.x(), p_end.y() - p_beg.y());
    const double theta_beg_end = std::atan2(v0.y(), v0.x());

    for (std::size_t i = 1 + offset; i < path_.size(); i += 1) {
      const Eigen::Vector3d& p_cur = path_[i];
      Eigen::Vector2d v1(p_cur.x() - p_beg.x(), p_cur.y() - p_beg.y());
      const double theta_beg_cur = std::atan2(v1.y(), v1.x());
      double d_ang =
          angles::shortest_angular_distance(theta_beg_cur, theta_beg_end);
      curvature += std::abs(d_ang) * 57.3;
    }
    return curvature / 1000.0;
  }

 private:
  std::size_t ahead_idx_ = 0;
  Eigen::Vector3d robot_pos_;
  Eigen::Quaterniond robot_rot_;
  std::vector<Eigen::Vector3d> path_;
};

class PathSampler {
 public:
  PathSampler() {}
  ~PathSampler() {}

  std::size_t size() { return this->samples_.size(); }

  void addSample(const Path& path) { samples_.emplace_back(path); }

  void addSample(Path&& path) { samples_.emplace_back(std::move(path)); }

  std::vector<Path> sample(const Eigen::Vector3d& position,
                           const Eigen::Quaterniond& rotation) const {
    std::vector<Path> pathes;
    for (auto&& path : samples_) {
      Path new_path = path;
      new_path.transform(position, rotation);
      pathes.emplace_back(std::move(new_path));
    }
    return pathes;
  }

 private:
  std::vector<Path> samples_;
};

}  // namespace pathlib
