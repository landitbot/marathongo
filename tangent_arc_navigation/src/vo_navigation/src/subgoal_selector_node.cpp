#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <limits>
#include <vector>
#include <cmath>
#include <string>

class SubgoalSelector {
 public:
  SubgoalSelector()
      : nh_("~"),
        has_path_(false),
        has_odom_(false),
        last_nearest_index_(0),
        current_target_index_(0),
        has_target_index_(false),
        lookahead_dist_(1.0),
        enable_adaptive_lookahead_(false),
        speed_lookahead_gain_(0.5),
        max_lookahead_dist_(6.0),
        min_lookahead_dist_(1.5),
        nearest_reacquire_dist_(8.0),
        refresh_dist_(1.0),
        target_advance_step_(3),
        max_search_window_(200),
        publish_period_sec_(0.05),
        curvature_window_(2.0),
        end_distance_threshold_(0.5),
        end_confirmation_count_(5),
        reached_end_count_(0),
        is_at_end_(false),
        path_length_when_end_(0.0),
        path_topic_("/central/smoothed_path"),
        odom_topic_("/odometry_body"),
        subgoal_topic_("/vo/subgoal") {
    nh_.param("lookahead_dist", lookahead_dist_, 1.0);
    nh_.param("enable_adaptive_lookahead", enable_adaptive_lookahead_, false);
    nh_.param("speed_lookahead_gain", speed_lookahead_gain_, 0.5);
    nh_.param("max_lookahead_dist", max_lookahead_dist_, 6.0);
    nh_.param("min_lookahead_dist", min_lookahead_dist_, 1.5);
    nh_.param("nearest_reacquire_dist", nearest_reacquire_dist_, 8.0);
    nh_.param("refresh_dist", refresh_dist_, 1.0);
    nh_.param("target_advance_step", target_advance_step_, 3);
    nh_.param("max_search_window", max_search_window_, 200);
    nh_.param("publish_period_sec", publish_period_sec_, 0.05);
    nh_.param("curvature_window", curvature_window_, 2.0);
    nh_.param("end_distance_threshold", end_distance_threshold_, 0.5);
    nh_.param("end_confirmation_count", end_confirmation_count_, 5);
    nh_.param<std::string>("path_topic", path_topic_, "/central/smoothed_path");
    nh_.param<std::string>("odom_topic", odom_topic_, "/odometry_body");
    nh_.param<std::string>("subgoal_topic", subgoal_topic_, "/vo/subgoal");

    path_sub_ = nh_.subscribe(path_topic_, 1,
                              &SubgoalSelector::pathCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 1,
                              &SubgoalSelector::odomCallback, this);
    subgoal_pub_ = nh_.advertise<geometry_msgs::PointStamped>(subgoal_topic_, 1);

    timer_ = nh_.createTimer(ros::Duration(publish_period_sec_),
                             &SubgoalSelector::timerCallback, this);

    ROS_INFO("subgoal_selector started, lookahead_dist=%.2f", lookahead_dist_);
  }

 private:
  ros::NodeHandle nh_;
  ros::Subscriber path_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher subgoal_pub_;
  ros::Timer timer_;

  nav_msgs::Path current_path_;
  nav_msgs::Odometry current_odom_;
  bool has_path_;
  bool has_odom_;
  std::size_t last_nearest_index_;
  std::size_t current_target_index_;
  bool has_target_index_;
  double lookahead_dist_;
  bool enable_adaptive_lookahead_;
  double speed_lookahead_gain_;
  double max_lookahead_dist_;
  double min_lookahead_dist_;
  double nearest_reacquire_dist_;
  double refresh_dist_;
  int target_advance_step_;
  int max_search_window_;
  double publish_period_sec_;
  double curvature_window_;
  double end_distance_threshold_;
  int end_confirmation_count_;
  int reached_end_count_;
  bool is_at_end_;
  double path_length_when_end_;
  std::string path_topic_;
  std::string odom_topic_;
  std::string subgoal_topic_;

  static double squaredDistance(double x1, double y1, double x2, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return dx * dx + dy * dy;
  }

  // 计算路径从start_idx到end_idx的平均曲率（基于方向角变化）
  double computePathCurvature(std::size_t start_idx, std::size_t end_idx) const {
    if (end_idx - start_idx < 2 || end_idx >= current_path_.poses.size()) {
      return 0.0;
    }

    double total_angle_change = 0.0;
    double total_distance = 0.0;

    for (std::size_t i = start_idx; i < end_idx; ++i) {
      const auto& p1 = current_path_.poses[i].pose.position;
      const auto& p2 = current_path_.poses[i + 1].pose.position;
      
      double angle1 = std::atan2(p1.y - current_path_.poses[start_idx].pose.position.y,
                                  p1.x - current_path_.poses[start_idx].pose.position.x);
      double angle2 = std::atan2(p2.y - current_path_.poses[start_idx].pose.position.y,
                                  p2.x - current_path_.poses[start_idx].pose.position.x);
      
      double dx = p2.x - p1.x;
      double dy = p2.y - p1.y;
      double dist = std::hypot(dx, dy);
      
      if (dist > 1e-6) {
        total_angle_change += std::abs(angle2 - angle1);
        total_distance += dist;
      }
    }

    if (total_distance < 1e-6) {
      return 0.0;
    }

    // 曲率 = 角度变化 / 路径长度，单位为 1/meter
    return total_angle_change / total_distance;
  }

  // 计算路径总长度
  double computePathLength(const nav_msgs::Path& path) const {
    double length = 0.0;
    for (std::size_t i = 0; i + 1 < path.poses.size(); ++i) {
      const auto& p1 = path.poses[i].pose.position;
      const auto& p2 = path.poses[i + 1].pose.position;
      length += std::hypot(p2.x - p1.x, p2.y - p1.y);
    }
    return length;
  }

  void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
    current_path_ = *msg;
    has_path_ = !current_path_.poses.empty();
    
    if (!has_path_) {
      last_nearest_index_ = 0;
      current_target_index_ = 0;
      has_target_index_ = false;
      is_at_end_ = false;
      reached_end_count_ = 0;
    } else if (last_nearest_index_ >= current_path_.poses.size()) {
      // 最近点索引越界，重置最近点但保留目标点
      last_nearest_index_ = 0;
    }
    
    // 关键修复：路径更新时，当目标索引越界或路径长度变化过大时，重置目标索引
    // 这防止了旧索引在新路径上指向错误位置（超级远或超级近）
    if (current_target_index_ >= current_path_.poses.size()) {
      current_target_index_ = 0;
      has_target_index_ = false;
      ROS_INFO("Path size changed, target_index reset (old_size implied > %zu)", current_path_.poses.size());
    }
    
    // 仅在“已经到达终点并锁定”后，若路径明显变长，才重置终点状态。
    // 否则在正常跟踪阶段 path_length_when_end_=0 会导致每次回调都触发重置。
    const double new_path_length = computePathLength(current_path_);
    if (is_at_end_ && path_length_when_end_ > 1e-3 &&
        new_path_length > path_length_when_end_ * 1.5) {
      is_at_end_ = false;
      reached_end_count_ = 0;
      ROS_INFO("Path refreshed (old: %.2f, new: %.2f), endpoint detection reset",
               path_length_when_end_, new_path_length);
    }
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    current_odom_ = *msg;
    has_odom_ = true;
  }

  void timerCallback(const ros::TimerEvent&) {
    if (!has_path_ || !has_odom_) {
      return;
    }

    const double rx = current_odom_.pose.pose.position.x;
    const double ry = current_odom_.pose.pose.position.y;
    const double speed = std::hypot(current_odom_.twist.twist.linear.x,
                    current_odom_.twist.twist.linear.y);

    const std::size_t n = current_path_.poses.size();
    const std::size_t start = last_nearest_index_;
    const std::size_t end = std::min<std::size_t>(n, start + static_cast<std::size_t>(std::max(1, max_search_window_)));

    std::size_t nearest = start;
    double best_d2 = std::numeric_limits<double>::infinity();

    for (std::size_t i = start; i < end; ++i) {
      const auto& p = current_path_.poses[i].pose.position;
      const double d2 = squaredDistance(rx, ry, p.x, p.y);
      if (d2 < best_d2) {
        best_d2 = d2;
        nearest = i;
      }
    }

    // 防止路径更新/回环/重定位后，最近点落在局部搜索窗之外导致子目标跳远。
    const double reacquire_d2 = nearest_reacquire_dist_ * nearest_reacquire_dist_;
    if (!std::isfinite(best_d2) || best_d2 > reacquire_d2) {
      const std::size_t prev_nearest = nearest;
      for (std::size_t i = 0; i < n; ++i) {
        const auto& p = current_path_.poses[i].pose.position;
        const double d2 = squaredDistance(rx, ry, p.x, p.y);
        if (d2 < best_d2) {
          best_d2 = d2;
          nearest = i;
        }
      }
      if (nearest != prev_nearest) {
        current_target_index_ = nearest;
        has_target_index_ = true;
      }
    }

    last_nearest_index_ = nearest;

    if (!has_target_index_) {
      current_target_index_ = nearest;
      has_target_index_ = true;
    }

    if (current_target_index_ < nearest) {
      current_target_index_ = nearest;
    }

    // 计算前方路径曲率用于自适应前视距离
    std::size_t curvature_end = nearest;
    double cumulative_dist = 0.0;
    while (curvature_end + 1 < n && cumulative_dist < curvature_window_) {
      const auto& p1 = current_path_.poses[curvature_end].pose.position;
      const auto& p2 = current_path_.poses[curvature_end + 1].pose.position;
      cumulative_dist += std::hypot(p2.x - p1.x, p2.y - p1.y);
      curvature_end++;
    }

    double path_curvature = computePathCurvature(nearest, curvature_end);

    double final_lookahead = lookahead_dist_;
    if (enable_adaptive_lookahead_) {
      // 基于曲率调整前视距离：曲率越大（弯道）, 前视距离越小；曲率越小（直线）, 前视距离越大
      double curvature_adjusted_lookahead = lookahead_dist_;
      if (path_curvature < 0.1) {
        curvature_adjusted_lookahead = lookahead_dist_ * 1.5;
      } else if (path_curvature > 0.3) {
        curvature_adjusted_lookahead = lookahead_dist_ * 0.4;
      }

      // 结合速度的自适应前视距离
      const double adaptive_lookahead = std::min(
        std::max(curvature_adjusted_lookahead,
                 curvature_adjusted_lookahead + speed_lookahead_gain_ * speed),
        max_lookahead_dist_);
      final_lookahead = adaptive_lookahead;
    }

    // 在固定前视模式下也遵守上下界。
    final_lookahead = std::max(min_lookahead_dist_, std::min(max_lookahead_dist_, final_lookahead));

    // 只要接近当前子目标点，就提前向前刷新，而不是等到完全到达。
    while (current_target_index_ + 1 < n) {
      const auto& p = current_path_.poses[current_target_index_].pose.position;
      const double d2 = squaredDistance(rx, ry, p.x, p.y);
      if (d2 > refresh_dist_ * refresh_dist_) {
        break;
      }
      const std::size_t step = static_cast<std::size_t>(std::max(1, target_advance_step_));
      current_target_index_ = std::min(n - 1, current_target_index_ + step);
    }

    // 终点检测：稳定确认机制防止SLAM卡顿导致误判
    const auto& last_point = current_path_.poses[n - 1].pose.position;
    double dist_to_end = std::sqrt(squaredDistance(rx, ry, last_point.x, last_point.y));
    
    if (dist_to_end < end_distance_threshold_ && !is_at_end_) {
      // 接近终点，计数器+1
      reached_end_count_++;
      if (reached_end_count_ >= end_confirmation_count_) {
        is_at_end_ = true;
        path_length_when_end_ = computePathLength(current_path_);
        ROS_INFO("Endpoint detected and locked! Distance: %.3f m", dist_to_end);
      }
    } else if (dist_to_end >= end_distance_threshold_ * 1.5) {
      // 远离终点，重置计数器（滞后处理，只有距离足够远才重置）
      reached_end_count_ = 0;
    }

    // 按要求：以“最近路径点”为锚点，沿前进方向累计 final_lookahead 选取子目标。
    std::size_t target = nearest;
    
    if (is_at_end_) {
      // 已到达终点，始终发布最后一个点
      target = n - 1;
    } else {
      double accumulated = 0.0;
      target = nearest;

      for (std::size_t i = nearest + 1; i < n; ++i) {
        const auto& p_prev = current_path_.poses[i - 1].pose.position;
        const auto& p_curr = current_path_.poses[i].pose.position;
        accumulated += std::hypot(p_curr.x - p_prev.x, p_curr.y - p_prev.y);
        target = i;

        if (accumulated >= final_lookahead) {
          break;
        }
      }
    }

    current_target_index_ = target;

    geometry_msgs::PointStamped subgoal;
    subgoal.header.stamp = ros::Time::now();
    subgoal.header.frame_id = current_path_.header.frame_id;
    subgoal.point = current_path_.poses[target].pose.position;
    subgoal_pub_.publish(subgoal);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "subgoal_selector_node");
  SubgoalSelector node;
  ros::spin();
  return 0;
}
