#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

#include "geometry_msgs/Twist.h"
#include "geometry_msgs/TwistStamped.h"
#include "ros/ros.h"
#include "signal.h"
#include "zmq.hpp"
#include "zmq_addon.hpp"

constexpr const char* remote_ip = "xxx.xxx.xxx.xxx";

std::string makeAddress(short port) {
  return std::string("tcp://") + remote_ip + ":" + std::to_string(port);
}

class RobotInterfaceNode {
 private:
  std::atomic<bool> running{false};
  zmq::context_t ctx;
  zmq::socket_t sock_suber_robot_status;
  std::thread th_process_;

 public:
  RobotInterfaceNode() = default;

  ~RobotInterfaceNode() = default;

  void start() {
    sock_suber_robot_status = zmq::socket_t(ctx, zmq::socket_type::sub);
    sock_suber_robot_status.connect(makeAddress(7700));
    sock_suber_robot_status.set(zmq::sockopt::subscribe, "");
    running.store(true);

    th_process_ = std::thread(&RobotInterfaceNode::process, this);
  }

  void stop() {
    running.store(false);
    th_process_.join();
  }

  void process() {
    while (running.load(std::memory_order_acquire)) {
      if (!sock_suber_robot_status) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }

      zmq::message_t msg;
      auto result = sock_suber_robot_status.recv(msg, zmq::recv_flags::none);
      if (result) {
        std::string data = msg.to_string();
        std::cout << data << std::endl;
      }
    }
  }
};

void handler_exit(int) { std::exit(0); }

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "robot_interface_node");
  ros::NodeHandle nh;

  signal(SIGINT, handler_exit);
  signal(SIGKILL, handler_exit);

  RobotInterfaceNode node;
  node.start();

  std::cout << "Started" << std::endl;

  while (ros::ok()) {
    ros::spinOnce();
  }

  node.stop();
  return 0;
}