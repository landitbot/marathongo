#pragma once

#include <sensor_msgs/point_cloud2_iterator.h>

#include "local_planner/grids_types.hpp"
#include "local_planner/hashvoxel_ring.hpp"
#include "local_planner/path_sampler.hpp"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

inline visualization_msgs::MarkerArray make_robot_maker(
    const Eigen::Vector3d& pos, const Eigen::Quaterniond& rot) {
  visualization_msgs::MarkerArray msg;

  visualization_msgs::Marker marker;
  marker.ns = "robot_marker";
  marker.id = 0;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.scale.x = 0.5;
  marker.scale.y = 0.5;
  marker.scale.z = 1.0;
  marker.pose.position.x = pos.x();
  marker.pose.position.y = pos.y();
  marker.pose.position.z = pos.z();
  marker.pose.orientation.w = rot.w();
  marker.pose.orientation.x = rot.x();
  marker.pose.orientation.y = rot.y();
  marker.pose.orientation.z = rot.z();
  marker.color.r = 1.0;
  marker.color.g = 0.3;
  marker.color.b = 0.3;
  marker.color.a = 1.0;
  msg.markers.push_back(std::move(marker));
  return msg;
}

inline visualization_msgs::MarkerArray make_target_point_marker(double x,
                                                                double y,
                                                                double z) {
  visualization_msgs::MarkerArray msg;

  visualization_msgs::Marker marker;
  marker.ns = "target_point_marker";
  marker.id = 0;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.scale.x = 0.5;
  marker.scale.y = 0.5;
  marker.scale.z = 0.5;
  marker.pose.position.x = x;
  marker.pose.position.y = y;
  marker.pose.position.z = z;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.color.r = 0.3;
  marker.color.g = 0.3;
  marker.color.b = 1.0;
  marker.color.a = 1.0;
  msg.markers.push_back(std::move(marker));
  return msg;
}

template <typename T>
inline sensor_msgs::PointCloud2 make_ringvoxel_map_cloud(
    rvoxel::RingVoxelMap<T>& map) {
  sensor_msgs::PointCloud2 msg;

  size_t point_count = 0;
  for (rvoxel::RingVoxelMapIterator<T> iter(map); !iter.EOI(); iter++) {
    if (map.isInside(*iter)) {
      point_count++;
    }
  }

  // 设置头部信息
  msg.header.frame_id = "map";
  msg.header.stamp = ros::Time::now();

  // 使用 Modifier 设置字段结构
  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2Fields(3,                                         //
                                "x", 1, sensor_msgs::PointField::FLOAT32,  //
                                "y", 1, sensor_msgs::PointField::FLOAT32,  //
                                "z", 1, sensor_msgs::PointField::FLOAT32);
  modifier.resize(point_count);

  // 使用 Iterator 填充数据
  sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");

  for (rvoxel::RingVoxelMapIterator<T> iter(map); !iter.EOI(); iter++) {
    auto idx = *iter;
    if (map.isInside(idx)) {
      auto pos = map.toPosition(idx);

      *iter_x = pos.x * 1.0f;
      *iter_y = pos.y * 1.0f;
      *iter_z = pos.z * 1.0f;

      ++iter_x;
      ++iter_y;
      ++iter_z;
    }
  }
  return msg;
}

inline sensor_msgs::PointCloud2 make_point_cloud(
    const std::vector<Eigen::Vector3d>& pts) {
  sensor_msgs::PointCloud2 msg;

  // 设置头部信息
  msg.header.frame_id = "map";
  msg.header.stamp = ros::Time::now();

  // 使用 Modifier 设置字段结构
  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2Fields(3,                                         //
                                "x", 1, sensor_msgs::PointField::FLOAT32,  //
                                "y", 1, sensor_msgs::PointField::FLOAT32,  //
                                "z", 1, sensor_msgs::PointField::FLOAT32);
  modifier.resize(pts.size());

  // 使用 Iterator 填充数据
  sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");

  for (size_t i = 0; i < pts.size(); i++) {
    *iter_x = pts[i].x() * 1.0f;
    *iter_y = pts[i].y() * 1.0f;
    *iter_z = pts[i].z() * 1.0f;

    ++iter_x;
    ++iter_y;
    ++iter_z;
  }
  return msg;
}

inline visualization_msgs::MarkerArray make_path_group(
    const std::vector<pathlib::Path>& pathes) {
  visualization_msgs::MarkerArray msg;
  for (size_t i = 0; i < pathes.size(); i++) {
    visualization_msgs::Marker marker;
    marker.ns = "path_group";
    marker.id = i;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.r = 0.3;
    marker.color.g = 1.0;
    marker.color.b = 0.3;
    marker.color.a = 1.0;
    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    for (auto&& pose : pathes[i].getPath()) {
      geometry_msgs::Point p;
      p.x = pose.x();
      p.y = pose.y();
      p.z = pose.z();
      marker.points.push_back(p);
    }
    msg.markers.push_back(std::move(marker));
  }

  return msg;
}

inline nav_msgs::Path make_path(const std::vector<Eigen::Vector3d>& pts) {
  nav_msgs::Path msg;
  msg.header.frame_id = "map";
  msg.header.stamp = ros::Time::now();
  msg.poses.reserve(pts.size());
  for (auto&& pt : pts) {
    geometry_msgs::PoseStamped p;
    p.header.frame_id = "map";
    p.pose.position.x = pt.x();
    p.pose.position.y = pt.y();
    p.pose.position.z = pt.z();
    p.pose.orientation.w = 1.0;
    p.pose.orientation.x = 0.0;
    p.pose.orientation.y = 0.0;
    p.pose.orientation.z = 0.0;
    msg.poses.push_back(p);
  }
  return msg;
}

inline visualization_msgs::MarkerArray make_collision_range_marker(
    double x, double y, double z, double radius) {
  visualization_msgs::MarkerArray msg;

  visualization_msgs::Marker marker;
  marker.ns = "collision_range_marker";
  marker.id = 0;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.scale.x = radius * 2;
  marker.scale.y = radius * 2;
  marker.scale.z = radius * 2;
  marker.pose.position.x = x;
  marker.pose.position.y = y;
  marker.pose.position.z = z;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.color.r = 0.3;
  marker.color.g = 0.3;
  marker.color.b = 1.0;
  marker.color.a = 0.2;
  msg.markers.push_back(std::move(marker));
  return msg;
}