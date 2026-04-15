#!/bin/bash

# 工作空间路径（修改为你的实际路径）
WORKSPACE="$HOME/catkin_ws"
ROS_DISTRO="noetic"

# Source 环境
source /opt/ros/$ROS_DISTRO/setup.bash
source $WORKSPACE/devel/setup.bash

echo "Starting all ROS launch files in parallel..."

# 启动所有 launch 文件（后台运行）
roslaunch sim_interface sim_navigation.launch &
roslaunch obstacle_detector obstacle_detector.launch &
roslaunch costmap_converter laserscan_obstacle_converter.launch &

# 等待所有后台进程完成
wait

echo "All launch files have been started"
