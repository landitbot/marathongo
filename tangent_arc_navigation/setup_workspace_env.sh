#!/usr/bin/env bash
set -euo pipefail

# Configure this ROS catkin workspace on Ubuntu + ROS Noetic desktop
# Usage:
#   cd ~/catkin_ws
#   bash setup_workspace_env.sh
#
# Optional env vars:
#   ROS_DISTRO=noetic

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-noetic}"

# Ensure user-level pip scripts are reachable when pip installs into ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

echo "[1/5] Checking ROS installation..."
if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ERROR: /opt/ros/${ROS_DISTRO}/setup.bash not found."
  echo "Please install ROS ${ROS_DISTRO} desktop first."
  exit 1
fi
source "/opt/ros/${ROS_DISTRO}/setup.bash"

echo "[2/5] Installing base tooling..."
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  git \
  python3-pip \
  python3-empy \
  python3-testresources \
  python3-catkin-tools \
  python3-vcstool

echo "[3/5] Installing ROS dependencies via apt (no rosdep)..."
sudo apt-get install -y \
  libeigen3-dev \
  libpcl-dev \
  libtbb-dev \
  "ros-${ROS_DISTRO}-angles" \
  "ros-${ROS_DISTRO}-costmap-converter" \
  "ros-${ROS_DISTRO}-eigen-conversions" \
  "ros-${ROS_DISTRO}-geometry-msgs" \
  "ros-${ROS_DISTRO}-message-generation" \
  "ros-${ROS_DISTRO}-message-runtime" \
  "ros-${ROS_DISTRO}-nav-msgs" \
  "ros-${ROS_DISTRO}-pcl-conversions" \
  "ros-${ROS_DISTRO}-pcl-ros" \
  "ros-${ROS_DISTRO}-pointcloud-to-laserscan" \
  "ros-${ROS_DISTRO}-roscpp" \
  "ros-${ROS_DISTRO}-roslib" \
  "ros-${ROS_DISTRO}-rospy" \
  "ros-${ROS_DISTRO}-sensor-msgs" \
  "ros-${ROS_DISTRO}-std-msgs" \
  "ros-${ROS_DISTRO}-tf2" \
  "ros-${ROS_DISTRO}-tf2-geometry-msgs" \
  "ros-${ROS_DISTRO}-tf2-ros" \
  "ros-${ROS_DISTRO}-visualization-msgs"

echo "[4/5] Installing Python requirements..."
python3 -m pip install --upgrade pip
if [[ -f "${WS_DIR}/requirements.txt" ]]; then
  python3 -m pip install -r "${WS_DIR}/requirements.txt"
else
  echo "requirements.txt not found, skip pip install."
fi

echo "[5/5] Done."
echo "Add this to your shell rc if needed:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  source ${WS_DIR}/devel/setup.bash"
