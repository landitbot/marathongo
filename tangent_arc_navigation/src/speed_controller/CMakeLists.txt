cmake_minimum_required(VERSION 3.5)
project(speed_controller)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  angles
  tf2
  message_generation
)

add_message_files(
  FILES
  SpeedCommand.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp nav_msgs geometry_msgs angles tf2 message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})

# Discrete control version (default)
add_executable(${PROJECT_NAME}_discrete src/speed_controller_discrete.cpp)
target_link_libraries(${PROJECT_NAME}_discrete ${catkin_LIBRARIES})
add_dependencies(${PROJECT_NAME}_discrete ${PROJECT_NAME}_generate_messages_cpp)

# Continuous PID control version
add_executable(${PROJECT_NAME}_continuous src/speed_controller_continuous.cpp)
target_link_libraries(${PROJECT_NAME}_continuous ${catkin_LIBRARIES})
add_dependencies(${PROJECT_NAME}_continuous ${PROJECT_NAME}_generate_messages_cpp)

# For backward compatibility, also provide the main node name (defaults to discrete)
add_executable(${PROJECT_NAME}_node src/speed_controller_discrete.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})
add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_generate_messages_cpp)