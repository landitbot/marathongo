cmake_minimum_required(VERSION 3.5)
project(vo_navigation)

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 catkin 及其依赖包
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  roslib
  sensor_msgs
  speed_controller
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)

find_package(PCL REQUIRED COMPONENTS common filters segmentation search)

# 系统依赖（如果需要 Boost，可以取消注释）
# find_package(Boost REQUIRED COMPONENTS system)

################################################
# 声明消息、服务和动作
################################################

# 添加消息文件
add_message_files(
  FILES
  Obstacle.msg
  ObstacleArray.msg
)

# 生成消息
generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

################################################
# catkin 配置
################################################
catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs pcl_conversions pcl_ros roscpp roslib sensor_msgs speed_controller std_msgs tf2 tf2_ros tf2_geometry_msgs visualization_msgs
  DEPENDS
)

###########
# 构建
###########

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

# Include this package's headers
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_definitions(${PCL_DEFINITIONS})

# 创建库（可选，也可以直接编译到可执行文件中）
# add_library(${PROJECT_NAME} src/vo_navigation.cpp)

# 添加可执行文件
add_executable(vo_node
  src/vo_node.cpp
)

add_executable(subgoal_selector_node
  src/subgoal_selector_node.cpp
)

add_executable(lidar_obstacle_node
  src/lidar_obstacle_node.cpp
)


# 添加依赖（消息生成等）
add_dependencies(vo_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

add_dependencies(subgoal_selector_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

add_dependencies(lidar_obstacle_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)


# 链接库
target_link_libraries(vo_node
  ${catkin_LIBRARIES}
)

target_link_libraries(subgoal_selector_node
  ${catkin_LIBRARIES}
)

target_link_libraries(lidar_obstacle_node
  ${catkin_LIBRARIES}
)


#############
# 安装
#############

# 安装可执行文件
install(TARGETS vo_node subgoal_selector_node lidar_obstacle_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装头文件（如果有）
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

# 安装消息文件（由 catkin 自动处理，无需手动）