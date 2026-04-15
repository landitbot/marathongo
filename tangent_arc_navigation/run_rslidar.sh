#!/bin/bash

# RSLidar 启动脚本
# 用法: ./run_rslidar.sh

# 设置工作空间路径（使用你的实际路径）
RSLIDAR_WS="$HOME/workspace/driver/rslidar_ws"

# 设置ROS环境（根据你的ROS版本修改，如 noetic/melodic/kinetic）
ROS_DISTRO="noetic"  # 改为你的ROS版本

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RSLidar driver...${NC}"
echo -e "${GREEN}Workspace: $RSLIDAR_WS${NC}"

# 检查工作空间是否存在
if [ ! -d "$RSLIDAR_WS" ]; then
    echo -e "${RED}Error: Workspace $RSLIDAR_WS not found!${NC}"
    exit 1
fi

# 进入工作空间
cd $RSLIDAR_WS

# Source ROS 环境
if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    source /opt/ros/$ROS_DISTRO/setup.bash
else
    echo -e "${RED}Error: ROS $ROS_DISTRO not found!${NC}"
    exit 1
fi

# Source 工作空间
if [ -f "devel/setup.bash" ]; then
    source devel/setup.bash
else
    echo -e "${RED}Error: devel/setup.bash not found! Please run catkin_make first.${NC}"
    exit 1
fi

# 启动 RSLidar
roslaunch rslidar_sdk start.launch

# 如果退出，打印信息
echo -e "${RED}RSLidar driver stopped${NC}"
