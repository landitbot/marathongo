#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

#include "angles/angles.h"
#include "atomic"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"

#define SQURE(x) ((x) * (x))

ros::Publisher g_puber_tracking_info;
ros::Publisher g_puber_robot_on_path;
ros::Publisher g_puber_front_peek_point;
ros::Publisher g_puber_back_peek_point;
ros::Publisher g_puber_short_peek_point;

std::mutex g_path_mtx;
std::shared_ptr<nav_msgs::Path> g_path;
std::atomic<double> g_path_timestamp;
std::atomic<std::size_t> g_start_index;

/// @brief publish the final control info
/// @param target_x The furest point in the path
/// @param near_angle The biggest angle within the 0.5m radius of the path
/// @param path_curvature The curvature of the path within 8m.
void publish_tracking_info(double target_x, double near_angle = 0,
                           double path_curvature = 0,
                           double path_curvature_back = 0) {
  std_msgs::Float64MultiArray msg;
  msg.data.push_back(target_x);
  msg.data.push_back(near_angle);
  msg.data.push_back(path_curvature);
  msg.data.push_back(path_curvature_back);
  g_puber_tracking_info.publish(msg);
}

void publish_robot_on_path(double x, double y){
  nav_msgs::Odometry msg;
  msg.header.frame_id = "map";
  msg.child_frame_id = "robot";
  msg.pose.pose.position.x = x;
  msg.pose.pose.position.y = y;
  msg.pose.pose.position.z = 0;
  msg.pose.pose.orientation.w = 1;
  msg.pose.pose.orientation.x = 0;
  msg.pose.pose.orientation.y = 0;
  msg.pose.pose.orientation.z = 0;
  g_puber_robot_on_path.publish(msg);
}

void publish_front_peek(double x, double y){
  nav_msgs::Odometry msg;
  msg.header.frame_id = "map";
  msg.child_frame_id = "front_peek";
  msg.pose.pose.position.x = x;
  msg.pose.pose.position.y = y;
  msg.pose.pose.position.z = 0;
  msg.pose.pose.orientation.w = 1;
  msg.pose.pose.orientation.x = 0;
  msg.pose.pose.orientation.y = 0;
  msg.pose.pose.orientation.z = 0;
  g_puber_front_peek_point.publish(msg);
}

void publish_back_peek(double x, double y){
  nav_msgs::Odometry msg;
  msg.header.frame_id = "map";
  msg.child_frame_id = "back_peek";
  msg.pose.pose.position.x = x;
  msg.pose.pose.position.y = y;
  msg.pose.pose.position.z = 0;
  msg.pose.pose.orientation.w = 1;
  msg.pose.pose.orientation.x = 0;
  msg.pose.pose.orientation.y = 0;
  msg.pose.pose.orientation.z = 0;
  g_puber_back_peek_point.publish(msg);
}

void publish_short_peek(double x, double y){
  nav_msgs::Odometry msg;
  msg.header.frame_id = "map";
  msg.child_frame_id = "back_peek";
  msg.pose.pose.position.x = x;
  msg.pose.pose.position.y = y;
  msg.pose.pose.position.z = 0;
  msg.pose.pose.orientation.w = 1;
  msg.pose.pose.orientation.x = 0;
  msg.pose.pose.orientation.y = 0;
  msg.pose.pose.orientation.z = 0;
  g_puber_short_peek_point.publish(msg);
}



template <class T>
T clamp(T v, T min, T max) {
  if (v < min) {
    return min;
  }
  if (v > max) {
    return max;
  }
  return v;
}

// WARNING: path must be fixed
// std::size_t find_origin(nav_msgs::Odometry::ConstPtr& msg,
//                         std::shared_ptr<nav_msgs::Path>& path, std::size_t beg,
//                         std::size_t end) {
//   double min_dis2 = -1e9;
//   bool flag_decent = false;
//   std::size_t origin_index = 0;
//   for (std::size_t i = beg; i < path->poses.size(); ++i) {
//     double dx = (msg->pose.pose.position.x - path->poses[i].pose.position.x);
//     double dy = (msg->pose.pose.position.y - path->poses[i].pose.position.y);
//     double dis2 = dx * dx + dy * dy;

//     if (min_dis2 < 0) {
//       min_dis2 = dis2;
//       origin_index = i;
//       continue;
//     }

//     if (dis2 < min_dis2) {
//       min_dis2 = dis2;
//       origin_index = i;
//       flag_decent = true;
//     } else {
//       if (flag_decent && dis2 - min_dis2 > 1.0) {
//         break;
//       }
//     }
//   }
//   return origin_index;
// }

std::size_t find_origin(nav_msgs::Odometry::ConstPtr& msg,
                        std::shared_ptr<nav_msgs::Path>& path, std::size_t beg,
                        std::size_t end) {
  double min_dis2 = -1e9;
  std::size_t origin_index = 0;
  for (std::size_t i = beg; i < path->poses.size(); ++i) {
    double dx = (msg->pose.pose.position.x - path->poses[i].pose.position.x);
    double dy = (msg->pose.pose.position.y - path->poses[i].pose.position.y);
    double dis2 = dx * dx + dy * dy;

    if (min_dis2 < -1.0) {
      min_dis2 = dis2;
      origin_index = i;
      continue;
    }

    if (dis2 < min_dis2) {
      min_dis2 = dis2;
      origin_index = i;
    }
  }
  return origin_index;
}

double speed_derivate(double ts, double x, double y) {
  static bool first = true;
  static double last_x = 0;
  static double last_y = 0;
  static double last_ts = 0;

  if (first) {
    first = false;
    last_x = x;
    last_y = y;
    last_ts = ts;
    return 0;
  }

  double diff_x = (x - last_x);
  double diff_y = (y - last_y);
  double dx = std::sqrt(diff_x * diff_x + diff_y * diff_y);
  double dt = ts - last_ts;
  if (dt < 1e-5) {
    return 0;
  }
  return dx / dt;
}

void handler_robot_odom(nav_msgs::Odometry::ConstPtr msg) {
  std::cout << "OdomTS:" << msg->header.stamp.toSec() << std::endl;

  ros::Time ts0 = ros::Time::now();
  std::cout << "Now Begin:" << ts0.toSec() << std::endl;

  double robot_velocity =
      speed_derivate(msg->header.stamp.toSec(), msg->pose.pose.position.x,
                     msg->pose.pose.position.y);

  std::shared_ptr<nav_msgs::Path> path;
  {
    std::lock_guard<std::mutex> glock(g_path_mtx);
    path = g_path;
  }

  if (path == nullptr) {
    std::cout << "Path is nullptr!" << std::endl;
    publish_tracking_info(0, 0, 0, 0);
    return;
  }

  // check path timeout
  /*
  {
    double path_ts = g_path_timestamp.load(std::memory_order_acquire);
    if (msg->header.stamp.toSec() - path_ts > 0.5) {
      std::lock_guard<std::mutex> glock(g_path_mtx);
      g_path = nullptr;
      publish_tracking_info(0, 0, 0, 0);
      std::cout << "Large time diff" << std::endl;
      return;
    }
  }*/

  // simple stop judgement
  {
    Eigen::Vector2d final_point = Eigen::Vector2d(
        path->poses.back().pose.position.x, path->poses.back().pose.position.y);

    Eigen::Vector2d cur_pos =
        Eigen::Vector2d(msg->pose.pose.position.x, msg->pose.pose.position.y);

    double dist = (final_point - cur_pos).norm();
    if (dist < 0.5) {
      std::lock_guard<std::mutex> glock(g_path_mtx);
      g_path = nullptr;
      std::cout << "Got the final point!" << std::endl;
      publish_tracking_info(0, 0, 0, 0);
      return;
    }
  }

  // 1. crop the path to the robot origin.
  std::size_t start_index = g_start_index.load(std::memory_order_acquire);
  start_index = find_origin(msg, path, 0, 0);
  g_start_index.store(start_index, std::memory_order_release);

  double lookahead_time = 5.0;
  double min_ld = 20.0;
  double max_ld = 20.1;
  double dynamic_ld = clamp(robot_velocity * lookahead_time, min_ld, max_ld);
  double dynamic_ld_sq = SQURE(dynamic_ld);
  double path_resolution = 0.3;
  
  double short_front_peek_ld = 6.0; // min: 1.5
  double short_front_peek_ld_sq = SQURE(short_front_peek_ld);

  {
    auto& p = path->poses[start_index].pose.position;
    publish_robot_on_path(p.x, p.y);
  }

  std::cout << "DynamicLD: " << dynamic_ld << "   "
            << "Velocity: " << robot_velocity << std::endl;
            
  std::cout << "StartIndex: " << start_index << "   "
            << "Size: " << path->poses.size() << std::endl;

  // compute for the error variables
  double target_x = 4.9 * 4.9;
  double error_angle = 0;
  double curvature_front = 0;
  double curvature_back = 0;

  // Compute thr front curvature
  {
    int _start_index = start_index;
    int _end_index = start_index + (dynamic_ld / path_resolution);
    if (_start_index > 1 && _end_index < path->poses.size() - 1) {
      const auto& p_beg = path->poses[_start_index].pose.position;
      const auto& p_end = path->poses[_end_index].pose.position;
      publish_front_peek(p_end.x, p_end.y);
      Eigen::Vector2d v0(p_end.x - p_beg.x, p_end.y - p_beg.y);
      const double theta_beg_end = std::atan2(v0.y(), v0.x());
      for (std::size_t i = _start_index + 20; i < _end_index; i+=5) {
        const auto& p_cur = path->poses[i].pose.position;
        Eigen::Vector2d v1(p_cur.x - p_beg.x, p_cur.y - p_beg.y);
        const double theta_beg_cur = std::atan2(v1.y(), v1.x());
        double d_ang = angles::shortest_angular_distance(theta_beg_cur, theta_beg_end);
        curvature_front += std::abs(d_ang) * 57.3;
      }
    }
  }

  // Compute the back curvature
  {
    int _start_index = start_index - (3.0 / path_resolution);
    int _end_index = start_index;
    if (_start_index > 1 && _end_index < path->poses.size() - 1) {
      const auto& p_beg = path->poses[_start_index].pose.position;
      const auto& p_end = path->poses[_end_index].pose.position;
      publish_back_peek(p_beg.x, p_beg.y);
      Eigen::Vector2d v0(p_end.x - p_beg.x, p_end.y - p_beg.y);
      const double theta_beg_end = std::atan2(v0.y(), v0.x());
      for (std::size_t i = _start_index + 20; i < _end_index; i+=5) {
        const auto& p_cur = path->poses[i].pose.position;
        Eigen::Vector2d v1(p_cur.x - p_beg.x, p_cur.y - p_beg.y);
        const double theta_beg_cur = std::atan2(v1.y(), v1.x());
        double d_ang = angles::shortest_angular_distance(theta_beg_cur, theta_beg_end);
        curvature_back += std::abs(d_ang) * 57.3;
      }
    }
  }

  // Compute Angle Error
  {
    const int short_peek_index = start_index + short_front_peek_ld / path_resolution;
    if(short_peek_index > 0 && short_peek_index < path->poses.size()) {
      const double tar_x = path->poses[short_peek_index].pose.position.x;
      const double tar_y = path->poses[short_peek_index].pose.position.y;

      double dx = (tar_x - msg->pose.pose.position.x);
      double dy = (tar_y - msg->pose.pose.position.y);

      publish_short_peek(tar_x, tar_y);
      const double global_plan_angle = std::atan2(dy, dx);

      Eigen::Quaterniond robot_head_quat(msg->pose.pose.orientation.w,
                                        msg->pose.pose.orientation.x,
                                        msg->pose.pose.orientation.y,
                                        msg->pose.pose.orientation.z);
      robot_head_quat.normalize();
      Eigen::Vector3d front = robot_head_quat * Eigen::Vector3d::UnitX();
      double current_yaw = std::atan2(front.y(), front.x());
      double new_error_angle = angles::shortest_angular_distance(current_yaw, global_plan_angle);
      error_angle = new_error_angle;
    }
  }

  publish_tracking_info(std::sqrt(target_x), error_angle, curvature_front / 1000.0, curvature_back / 1000.0);
  ros::Time ts1 = ros::Time::now();
  std::cout << "TS used in handler_robot_odom:" << (ts1.toSec() - ts0.toSec()) * 1000 << std::endl;

  // Compute the target point and the error angle
  // {
  //   bool first = true;
  //   double start_dist2 = 0;
  //   for (std::size_t i = start_index + 1; i < path->poses.size() - 1; i++) {
  //     double dx = (path->poses[i].pose.position.x - msg->pose.pose.position.x);
  //     double dy = (path->poses[i].pose.position.y - msg->pose.pose.position.y);
  //     double dist2 = dx * dx + dy * dy;
      
  //     if(first) {
  //       first = false;
  //       start_dist2 = dist2;
  //     }
      
  //     double dist2_offset = dist2 - start_dist2;

  //     // get the furest point
  //     if (dist2 > target_x) target_x = dist2;

  //     // get the target angle
  //     if (dist2_offset > short_front_peek_ld_sq && dist2_offset <= short_front_peek_ld_sq + 1.5) {
  //       double global_plan_angle = std::atan2(dy, dx);

  //       publish_short_peek(path->poses[i].pose.position.x, path->poses[i].pose.position.y);

  //       Eigen::Quaterniond robot_head_quat(
  //           msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
  //           msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  //       robot_head_quat.normalize();

  //       Eigen::Vector3d front = robot_head_quat * Eigen::Vector3d::UnitX();

  //       double current_yaw = std::atan2(front.y(), front.x());
  //       double new_error_angle = angles::shortest_angular_distance(current_yaw, global_plan_angle);
  //       if (std::abs(new_error_angle) > std::abs(error_angle)) {
  //         error_angle = new_error_angle;
  //       }
  //     }

  //     if (dist2_offset > dynamic_ld_sq) {
  //       break;
  //     }
  //   }
  // }

  

}

void handler_tracking_path(nav_msgs::PathConstPtr msg) {
  if (msg->poses.empty()) {
    std::lock_guard<std::mutex> glock(g_path_mtx);
    g_path = nullptr;
    g_start_index.store(0, std::memory_order_release);
  } else {
    auto p = std::make_shared<nav_msgs::Path>();
    *p = *msg;
    {
      std::lock_guard<std::mutex> glock(g_path_mtx);
      g_path = p;
      g_start_index.store(0, std::memory_order_release);
    }
  }

  g_path_timestamp.store(msg->header.stamp.toSec(), std::memory_order_release);
}

void test() {
  double dx = 1.0;
  double dy = -1.0;

  double global_err_angle = -0.992718;  // std::atan2(dy, dx);
  // Eigen::Quaterniond robot_head_quat(
  //     Eigen::AngleAxisd(angles::from_degrees(30), Eigen::Vector3d::UnitZ()));

  Eigen::Quaterniond robot_head_quat(-0.6329629239392734, -0.10793546057468237,
                                     -0.08343474352590974, 0.7620672892482542);
  Eigen::Vector3d front = robot_head_quat * Eigen::Vector3d::UnitX();
  double current_yaw = std::atan2(front.y(), front.x());
  double error_angle =
      angles::shortest_angular_distance(current_yaw, global_err_angle);
  std::cout << "current_yaw:" << current_yaw << std::endl;
  std::cout << "err:" << angles::to_degrees(error_angle) << std::endl;
}

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "path_process_node");
  ros::NodeHandle nh;

  // test();

  g_path = nullptr;

  g_puber_tracking_info =
      nh.advertise<std_msgs::Float64MultiArray>("/tracking_info", 1, false);

  g_puber_robot_on_path = nh.advertise<nav_msgs::Odometry>("/robot_on_path", 1, false);

  g_puber_front_peek_point = nh.advertise<nav_msgs::Odometry>("/front_peek_point", 1, false);

  g_puber_back_peek_point = nh.advertise<nav_msgs::Odometry>("/back_peek_point", 1, false);

  g_puber_short_peek_point = nh.advertise<nav_msgs::Odometry>("/short_peek_point", 1, false);

  // ros::Subscriber suber_path =
  //     nh.subscribe("/move_base_flex/GoodLocalPlannerROS/global_plan", 1,
  //                  handler_tracking_path);

  ros::Subscriber suber_path =
      nh.subscribe("/original_path", 1,
                   handler_tracking_path);


  ros::Subscriber suber_robot_odom =
      nh.subscribe("/high_frequency_odometry", 1, handler_robot_odom);

  ros::spin();
  return 0;
}
