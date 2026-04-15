#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "angles/angles.h"
#include "gazebo_msgs/ModelStates.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Float64MultiArray.h"
#include "tbb/tbb.h"
#include "visualization_msgs/MarkerArray.h"

class sim_interface_node {
 private:
  ros::NodeHandle* nh_;
  ros::Subscriber suber_model_status_;
  ros::Subscriber suber_lidar_;
  ros::Subscriber suber_tracking_cmd_vel_;

  ros::Publisher puber_robot_odom_;
  ros::Publisher puber_lidar_;
  ros::Publisher puber_free_path_;
  ros::Publisher puber_cmd_vel_;

  void handler_model_status(gazebo_msgs::ModelStates::ConstPtr msg) {
    int idx_robot_pose = -1;
    for (size_t i = 0; i < msg->name.size(); i++) {
      if (msg->name[i] == "e1") {
        idx_robot_pose = i;
      }
    }
    if (idx_robot_pose >= 0) {
      nav_msgs::Odometry robot_odom_msg;
      robot_odom_msg.header.frame_id = "map";
      robot_odom_msg.header.stamp = ros::Time::now();
      robot_odom_msg.child_frame_id = "base_link";

      robot_odom_msg.pose.pose.position.x =
          msg->pose[idx_robot_pose].position.x;
      robot_odom_msg.pose.pose.position.y =
          msg->pose[idx_robot_pose].position.y;
      robot_odom_msg.pose.pose.position.z =
          msg->pose[idx_robot_pose].position.z;

      robot_odom_msg.pose.pose.orientation.w =
          msg->pose[idx_robot_pose].orientation.w;
      robot_odom_msg.pose.pose.orientation.x =
          msg->pose[idx_robot_pose].orientation.x;
      robot_odom_msg.pose.pose.orientation.y =
          msg->pose[idx_robot_pose].orientation.y;
      robot_odom_msg.pose.pose.orientation.z =
          msg->pose[idx_robot_pose].orientation.z;

      robot_odom_msg.twist.twist.linear.x = msg->twist[idx_robot_pose].linear.x;
      robot_odom_msg.twist.twist.linear.y = msg->twist[idx_robot_pose].linear.y;
      robot_odom_msg.twist.twist.linear.z = msg->twist[idx_robot_pose].linear.z;

      robot_odom_msg.twist.twist.angular.x =
          msg->twist[idx_robot_pose].angular.x;
      robot_odom_msg.twist.twist.angular.y =
          msg->twist[idx_robot_pose].angular.y;
      robot_odom_msg.twist.twist.angular.z =
          msg->twist[idx_robot_pose].angular.z;
      puber_robot_odom_.publish(robot_odom_msg);
    }
  }

  void handler_lidar(sensor_msgs::PointCloud2::ConstPtr msg) {
    puber_lidar_.publish(*msg);
  }

  void handler_track_cmd_vel(geometry_msgs::Twist::ConstPtr msg) {
    puber_cmd_vel_.publish(*msg);
  }

 public:
  sim_interface_node(ros::NodeHandle* nh) : nh_(nh) {}
  ~sim_interface_node() {}

  void start() {
    suber_model_status_ =
        nh_->subscribe("/gazebo/model_states", 1,
                       &sim_interface_node::handler_model_status, this);

    suber_lidar_ = nh_->subscribe("/lidar_points", 1,
                                  &sim_interface_node::handler_lidar, this);

    suber_tracking_cmd_vel_ = nh_->subscribe(
        "/fuzzy_cmd_vel", 1, &sim_interface_node::handler_track_cmd_vel, this);

    puber_robot_odom_ =
        nh_->advertise<nav_msgs::Odometry>("/high_frequency_odometry", 1);

    puber_lidar_ =
        nh_->advertise<sensor_msgs::PointCloud2>("/rslidar_points", 1);

    puber_free_path_ = nh_->advertise<nav_msgs::Path>("/free_path", 1);

    puber_cmd_vel_ = nh_->advertise<geometry_msgs::Twist>("/cmd_vel", 1);
  }

  bool loadFreePath(const std::string& pathfile) {
    std::ifstream file(pathfile);
    if (!file.is_open()) {
      ROS_ERROR("Cannot open path: %s", pathfile.c_str());
      return false;
    }
    std::string line;
    nav_msgs::Path path_msg_;

    double last_x, last_y, last_z;
    bool first = true;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      double timestamp, x, y, z, qx, qy, qz, qw;

      // 解析每一行的8个数值
      if (!(iss >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {
        ROS_WARN("Skip line: %s", line.c_str());
        continue;
      }

      if (first) {
        first = false;
      } else {
        double dx = last_x - x;
        double dy = last_y - y;
        double dz = last_z - z;
        double dis = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dis < 0.3) {
          continue;
        }
      }

      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.stamp = ros::Time(timestamp);
      pose_stamped.header.frame_id = "map";

      // 设置位置
      pose_stamped.pose.position.x = x;
      pose_stamped.pose.position.y = y;
      pose_stamped.pose.position.z = z;

      // 设置姿态（四元数）
      pose_stamped.pose.orientation.x = qx;
      pose_stamped.pose.orientation.y = qy;
      pose_stamped.pose.orientation.z = qz;
      pose_stamped.pose.orientation.w = qw;

      path_msg_.poses.push_back(pose_stamped);

      last_x = x;
      last_y = y;
      last_z = z;
    }

    file.close();

    if (path_msg_.poses.empty()) {
      ROS_ERROR("No poses");
      return false;
    }

    // 设置Path消息的header
    path_msg_.header.frame_id = "map";
    path_msg_.header.stamp = ros::Time::now();

    ROS_INFO("Loaded %zu Poses", path_msg_.poses.size());
    puber_free_path_.publish(path_msg_);
    return true;
  }
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "sim_interface_node");
  ros::NodeHandle nh;

  sim_interface_node node(&nh);
  node.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  node.loadFreePath("./free_path.txt");

  ros::spin();
  return 0;
}
