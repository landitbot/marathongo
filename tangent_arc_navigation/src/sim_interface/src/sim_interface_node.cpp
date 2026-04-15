#include <geometry_msgs/Twist.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "angles/angles.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include "tbb/tbb.h"

class sim_interface_node {
 private:
  ros::NodeHandle* nh_;
  ros::Subscriber suber_odometry_;

  ros::Publisher puber_odometry_body_;

  ros::Time last_stamp_;
  std::string odom_input_topic_;
  std::string odom_body_topic_;
  std::string odom_body_frame_id_;
  std::string odom_body_child_frame_id_;

  tf::TransformBroadcaster tf_broadcaster_;

  void publishOdomBodyTF(const nav_msgs::Odometry& odom_msg) {
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(odom_msg.pose.pose.position.x,
                                    odom_msg.pose.pose.position.y,
                                    odom_msg.pose.pose.position.z));

    const auto& q = odom_msg.pose.pose.orientation;
    transform.setRotation(tf::Quaternion(q.x, q.y, q.z, q.w));

    tf_broadcaster_.sendTransform(
        tf::StampedTransform(transform,
                             odom_msg.header.stamp,
                             odom_msg.header.frame_id,
                             odom_msg.child_frame_id));
  }

  void handler_odometry(nav_msgs::Odometry::ConstPtr msg) {
    // 将线速度和角速度转换到 child_frame（robot body）坐标系，然后发布到 odometry_body
    nav_msgs::Odometry odom_body_msg;
    odom_body_msg.header.stamp = msg->header.stamp;
    odom_body_msg.header.frame_id = odom_body_frame_id_;      // camera_init (世界)
    odom_body_msg.child_frame_id = odom_body_child_frame_id_; // body (机器人)

    // 直接拷贝位姿（假定原位姿是在世界系或用户期望的系中）
    odom_body_msg.pose = msg->pose;

    // 把线速度和角速度从消息坐标系旋转到 body（通过四元数逆旋转）
    tf::Quaternion q(msg->pose.pose.orientation.x,
                     msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z,
                     msg->pose.pose.orientation.w);
    tf::Quaternion q_inv = q.inverse();

    tf::Vector3 lin(msg->twist.twist.linear.x,
                    msg->twist.twist.linear.y,
                    msg->twist.twist.linear.z);
    tf::Vector3 ang(msg->twist.twist.angular.x,
                    msg->twist.twist.angular.y,
                    msg->twist.twist.angular.z);

    tf::Vector3 lin_body = tf::quatRotate(q_inv, lin);
    tf::Vector3 ang_body = tf::quatRotate(q_inv, ang);

    odom_body_msg.twist.twist.linear.x = lin_body.x();
    odom_body_msg.twist.twist.linear.y = lin_body.y();
    odom_body_msg.twist.twist.linear.z = lin_body.z();

    odom_body_msg.twist.twist.angular.x = ang_body.x();
    odom_body_msg.twist.twist.angular.y = ang_body.y();
    odom_body_msg.twist.twist.angular.z = ang_body.z();

    puber_odometry_body_.publish(odom_body_msg);
    publishOdomBodyTF(odom_body_msg);
  }

 public:
  sim_interface_node(ros::NodeHandle* nh) : nh_(nh) {
    // 参数通过 launch 文件传入
    nh_->param<std::string>("odom_input_topic", odom_input_topic_, "/odometry");
    nh_->param<std::string>("odom_body_topic", odom_body_topic_, "/odometry_body");
    nh_->param<std::string>("odom_body_frame_id", odom_body_frame_id_, "camera_init");
    nh_->param<std::string>("odom_body_child_frame_id", odom_body_child_frame_id_, "body");
  }
  
  ~sim_interface_node() {}

  void start() {
        // 订阅 odometry 输入
        suber_odometry_ = nh_->subscribe(odom_input_topic_, 10,
                    &sim_interface_node::handler_odometry, this);

        // 发布处理后的 odometry_body
        puber_odometry_body_ = nh_->advertise<nav_msgs::Odometry>(odom_body_topic_, 10);

        ROS_INFO("sim_interface_node started");
        ROS_INFO("Subscribing to odometry: %s", odom_input_topic_.c_str());
        ROS_INFO("Publishing odometry_body: %s (frame=%s child=%s)",
           odom_body_topic_.c_str(), odom_body_frame_id_.c_str(), odom_body_child_frame_id_.c_str());
  }
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "sim_interface_node");
  ros::NodeHandle nh("~");  // 使用私有命名空间以便获取参数

  sim_interface_node node(&nh);
  node.start();

  ros::spin();
  return 0;
}