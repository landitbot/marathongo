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
#include <random>
#include <thread>
#include <unordered_set>

#include "angles/angles.h"
#include "local_planner/djikstra.hpp"
#include "local_planner/filters.hpp"
#include "local_planner/grids_types.hpp"
#include "local_planner/hashvoxel_ring.hpp"
#include "local_planner/kinematic_envelope.hpp"
#include "local_planner/msg_tool.hpp"
#include "local_planner/path_sampler.hpp"
#include "local_planner/scopetimer.hpp"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include "tbb/tbb.h"
#include "visualization_msgs/MarkerArray.h"

enum class PlannerMode : int {
  TRACKING = 0,
  AVOIDANCE,
  RECOVERY,
};

struct LineParam {
  Eigen::Vector2d p0;
  Eigen::Vector2d n_vec;
};

inline double getTimestamp() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class local_planner_node {
 public:
  local_planner_node(ros::NodeHandle* nh) : nh_(nh) {
    elevation_map.setGeometry(20.0, 20.0, 3.0, 0.2);
    auto local_map_size = elevation_map.getSize();
    obstacle_map.setGeometry(local_map_size.x, local_map_size.y, 1, 0.2);
    planner_ = std::make_unique<slamchain::Dijkstra>(local_map_size.x,
                                                     local_map_size.y, 1);
    smoother_vel_ = std::make_unique<Smoother>(0.5, 0.3, 0.1);
    smoother_rad_ = std::make_unique<Smoother>(0.3, 0.3, 0.1);
    robot_shake_filter_ = std::make_unique<RobotShakeFilter>(5);
  }
  ~local_planner_node() {}

  void start() {
    suber_free_path_ =
        nh_->subscribe("/central/smoothed_path", 1,
                       &local_planner_node::handler_free_path, this);

    suber_obstacle_path1_ =
        nh_->subscribe("/left/smoothed_path", 1,
                       &local_planner_node::handler_left_obstacle_path, this);

    suber_obstacle_path2_ =
        nh_->subscribe("/right/smoothed_path", 1,
                       &local_planner_node::handler_right_obstacle_path, this);

    // suber_lidar_ = nh_->subscribe("/rslidar_points", 1,
    //                               &local_planner_node::handler_lidar, this);

    suber_lidar_ = nh_->subscribe("/current_scan_body", 1,
                                  &local_planner_node::handler_lidar, this);

    suber_lidar_odom_ = nh_->subscribe(
        "/odometry", 1, &local_planner_node::handler_lidar_odom, this);

    // suber_target_point_ =
    //     nh_->subscribe("/local_planner/target", 1,
    //                    &local_planner_node::handler_lidar_odom, this);

    puber_debug_local_map_ =
        nh_->advertise<sensor_msgs::PointCloud2>("/debug/local_map", 1);

    puber_debug_obstacle_map_ = nh_->advertise<sensor_msgs::PointCloud2>(
        "/debug/local_obstacle_map", 1);

    puber_debug_point_cloud_ =
        nh_->advertise<sensor_msgs::PointCloud2>("/debug/pc2", 1, true);

    puber_debug_cropped_path_ = nh_->advertise<nav_msgs::Path>(
        "/debug/local_planner/cropped_path", 1, true);

    puber_debug_cropped_left_path_ = nh_->advertise<nav_msgs::Path>(
        "/debug/local_planner/cropped_left_path", 1, true);

    puber_debug_cropped_right_path_ = nh_->advertise<nav_msgs::Path>(
        "/debug/local_planner/cropped_right_path", 1, true);

    puber_debug_collision_range_ =
        nh_->advertise<visualization_msgs::MarkerArray>(
            "/debug/collision_range", 1, true);

    puber_debug_cropped_path_curvature_ =
        nh_->advertise<std_msgs::Float64MultiArray>(
            "/debug/cropped_path_curvature", 1, true);

    puber_debug_voxel_collision_range_ =
        nh_->advertise<sensor_msgs::PointCloud2>("/debug/voxel_collision_range",
                                                 1, true);

    puber_robot_marker_ = nh_->advertise<visualization_msgs::MarkerArray>(
        "/debug/robot_marker", 1, true);

    puber_path_group_ = nh_->advertise<visualization_msgs::MarkerArray>(
        "/debug/path_group", 1, true);

    puber_path_ =
        nh_->advertise<nav_msgs::Path>("/local_planner/path", 1, true);

    puber_ahead_pt_ = nh_->advertise<visualization_msgs::MarkerArray>(
        "/debug/ahead_marker", 1, true);

    puber_cmd_vel_ =
        nh_->advertise<geometry_msgs::Twist>("/fuzzy_cmd_vel", 1, false);

    puber_recover_path_ =
        nh_->advertise<nav_msgs::Path>("/local_planner/recover_path", 1, true);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    init_sampler();
    init_kinematic_envelope();
    collision_range_ = generate_collision_body(kAvoidanceCollisionRange);
    planner_goal_collision_range_ =
        generate_collision_body(kPlannerCollisionRange);
  }

  void init_sampler() {
    std::string folder_path = "./pathes";

    auto readPath = [](const std::string& pathfile) {
      std::vector<double> result;

      std::ifstream file(pathfile, std::ios::binary);
      file.seekg(0, std::ios::end);
      size_t size = file.tellg();
      file.seekg(0, std::ios::beg);

      std::string content(size, '\0');
      file.read(content.data(), size);

      std::stringstream ss(content);
      std::string token;
      while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
          try {
            double value = std::stod(token);
            result.push_back(value);
          } catch (const std::exception& e) {
            std::cerr << "failed to parse double: " << token << std::endl;
          }
        }
      }
      return result;
    };

    std::vector<std::filesystem::path> path_files;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
      auto suffix = entry.path().extension();
      if (entry.is_regular_file() && suffix == ".path") {
        path_files.push_back(entry.path());
      }
    }

    std::sort(path_files.begin(), path_files.end(), std::less<>());

    int count = 0;
    for (const auto& path : path_files) {
      std::cout << "Loading Path:" << path << std::endl;
      auto path_data = readPath(path.string());
      sampler_.addSample(pathlib::Path(std::move(path_data)));
      count++;
    }
  }

  void init_kinematic_envelope() {
    {
      ke.addDim("velx", 0.0, 3.5, 300, 10);
      ke.addDim("velz", -2.3, 2.3, 300, 20);
    }

    {
      // Fast Rotation:
      // velx = [1.6, 2.3]
      // velz = [-1.5, 1.5]

      // min velx
      {
        auto e = KE::LinearEnvelope<'>'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->config(1.0, 0.0, 1.6);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::AVOIDANCE;
        });
        ke.addEnvelope(std::move(e));
      }

      // max velx
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->config(1.0, 0.0, 2.3);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::AVOIDANCE;
        });
        ke.addEnvelope(std::move(e));
      }

      // velz
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->config(0.0, 1.0, 1.5);
        e->setYAbs(true);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::AVOIDANCE;
        });
        ke.addEnvelope(std::move(e));
      }

      // speed constraint
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->setYAbs(true);
        e->config(1.0, 0.4, 2.3);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::AVOIDANCE;
        });
        ke.addEnvelope(std::move(e));
      }
    }

    {
      // Fast Dash:
      // velx = [2.5, 3.5]
      // velz = [-2.3, 2.3]

      // min velx
      {
        auto e = KE::LinearEnvelope<'>'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->config(1.0, 0.0, 2.5);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::TRACKING;
        });
        ke.addEnvelope(std::move(e));
      }

      // max velx
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->config(1.0, 0.0, 3.5);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::TRACKING;
        });
        ke.addEnvelope(std::move(e));
      }

      // velz
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->setYAbs(true);
        e->config(0.0, 1.0, 2.3);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::TRACKING;
        });
        ke.addEnvelope(std::move(e));
      }

      // speed constraint
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->setYAbs(true);
        e->config(1.0, 1.8, 7.0);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::TRACKING;
        });
        ke.addEnvelope(std::move(e));
      }
    }

    {
      // Recovery:
      // velx = [0, 0.6]
      // velz = [-0.8, 0.8]

      // min velx
      {
        auto e = KE::LinearEnvelope<'>'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->config(1.0, 0.0, 0.0);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::RECOVERY;
        });
        ke.addEnvelope(std::move(e));
      }

      // max velx
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->config(1.0, 0.0, 0.6);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::RECOVERY;
        });
        ke.addEnvelope(std::move(e));
      }

      // velz
      {
        auto e = KE::LinearEnvelope<'<'>::create();
        e->setXName("velx");
        e->setYName("velz");
        e->setYAbs(true);
        e->config(0.0, 1.0, 0.8);
        e->setCondition([&](double x, double y) {
          return ctx_planner_mode == PlannerMode::RECOVERY;
        });
        ke.addEnvelope(std::move(e));
      }
    }

    ke.compile();
    ke.setMaxIteration(10);
  }

  void publish_cmd_vel(double velx, double velz, bool smooth = true) {
    if (smooth) {
      velx = smoother_vel_->compute(velx);
      // velz = smoother_rad_->compute(velz);
    }

    ctx_last_velx = velx;

    geometry_msgs::Twist msg;
    msg.linear.x = velx / 3.5;
    msg.linear.y = 0;
    msg.linear.z = 0;
    msg.angular.x = 0;
    msg.angular.y = 0;
    msg.angular.z = velz;
    puber_cmd_vel_.publish(msg);
  }

 private:
  ros::NodeHandle* nh_;

  pathlib::PathSampler sampler_;
  std::vector<PathControlInfo> path_control_info_;

  ros::Subscriber suber_lidar_;
  ros::Subscriber suber_lidar_odom_;
  ros::Subscriber suber_target_point_;

  ros::Subscriber suber_free_path_;
  ros::Subscriber suber_obstacle_path1_;
  ros::Subscriber suber_obstacle_path2_;

  ros::Publisher puber_debug_local_map_;
  ros::Publisher puber_debug_obstacle_map_;
  ros::Publisher puber_debug_point_cloud_;
  ros::Publisher puber_debug_cropped_path_;
  ros::Publisher puber_debug_cropped_left_path_;
  ros::Publisher puber_debug_cropped_right_path_;
  ros::Publisher puber_debug_collision_range_;
  ros::Publisher puber_debug_cropped_path_curvature_;
  ros::Publisher puber_debug_voxel_collision_range_;
  ros::Publisher puber_robot_marker_;
  ros::Publisher puber_path_;
  ros::Publisher puber_path_group_;
  ros::Publisher puber_ahead_pt_;
  ros::Publisher puber_cmd_vel_;
  ros::Publisher puber_recover_path_;

  rvoxel::RingVoxelMap<GridElevation> elevation_map;
  rvoxel::RingVoxelMap<GridObstacle> obstacle_map;
  std::unique_ptr<slamchain::Dijkstra> planner_;

  KE::KinematicEnvelope ke;

  std::unique_ptr<Smoother> smoother_vel_;
  std::unique_ptr<Smoother> smoother_rad_;
  std::unique_ptr<RobotShakeFilter> robot_shake_filter_;

  std::vector<rvoxel::Index> collision_range_;
  std::vector<rvoxel::Index> planner_goal_collision_range_;
  // tag:config
  double kRobotMaxSpeed = 3.5;

  double kAvoidanceModeObstacleDistance = 7.0;
  double kMinScoreSafety = 3.0;
  double kFatalMinScoreSafety = 1.5;
  int kAvoidanceCollisionRange = 3;

  double kSamplePathLength = 8.0;
  double kSamplePathInnerLength = 1.0;
  int kPathMaxAheadPointIdx = 90;

  double kFreePathAhead = 20.0;
  double kFreePathBhead = 5.0;
  double kInterpolationPathResolution = 0.3;

  double kFreePathPadding = 1.0;
  double kFreePathPaddingEdgeProbability = 0.8;

  double kWeightEnergyConsistency = 1.0;

  // recovery
  int kPlannerCollisionRange = 5;
  int kPlannerInflationRange = 3;
  int kPlannerReplanDetectRange = 1;
  double kPlannerGoalAhead = 3.0;
  double kWatchDogExpSec = 3.0;
  double kRecoveryAhead = 0.5;
  double kMinRecoveryPathLength = 2.0;
  double kMaxRecoveryDistanceToFreePath = 1.0;
  double kMaxRecoveryYawRangeDeg = 30;

  // store free path (free: unconstrainted)
  std::mutex free_path_lock_;
  nav_msgs::Path::ConstPtr free_path_;

  std::mutex left_obstacle_path_lock_;
  nav_msgs::Path::ConstPtr left_obstacle_path_;

  std::mutex right_obstacle_path_lock_;
  nav_msgs::Path::ConstPtr right_obstacle_path_;

  // Context Variables
  //{
  pathlib::Path cropped_path_;
  pathlib::Path cropped_left_path_;
  pathlib::Path cropped_right_path_;
  int free_path_origin_index = 0;
  int left_path_origin_index = 0;
  int right_path_origin_index = 0;

  PlannerMode ctx_planner_mode = PlannerMode::RECOVERY;
  double p_one_tracking = 0;
  double ctx_distance_to_free_path_;
  double ctx_last_energy = 0;
  double ctx_last_velx = 0;
  double ctx_yaw_error_gain = 0;

  // recovery
  pathlib::Path ctx_recovery_path;
  double ctx_watch_dog_exp_ts = -1;
  //}

  // store odometry
  std::mutex lidar_odom_lock_;
  std::map<double, nav_msgs::Odometry> lidar_odom_;

  std::atomic<double> robot_vel_x{0.0};
  std::atomic<double> robot_vel_z{0.0};

  std::vector<rvoxel::Index> generate_collision_body(int r) {
    std::vector<rvoxel::Index> range;
    for (int i = -r; i <= r; i++) {
      for (int j = -r; j <= r; j++) {
        range.push_back({i, j, 0});
      }
    }
    return range;
  }

  void precompute_path_control_info() {
    path_control_info_.resize(sampler_.size());
  }

  void handler_free_path(nav_msgs::Path::ConstPtr msg) {
    std::lock_guard<std::mutex> glock(free_path_lock_);
    free_path_ = msg;
  }

  void handler_left_obstacle_path(nav_msgs::Path::ConstPtr msg) {
    std::lock_guard<std::mutex> glock(left_obstacle_path_lock_);
    left_obstacle_path_ = msg;
  }

  void handler_right_obstacle_path(nav_msgs::Path::ConstPtr msg) {
    std::lock_guard<std::mutex> glock(right_obstacle_path_lock_);
    right_obstacle_path_ = msg;
  }

  void handler_lidar_odom_pose_stamped(geometry_msgs::PoseStampedConstPtr msg) {
    nav_msgs::OdometryPtr odom_msg = boost::make_shared<nav_msgs::Odometry>();
    odom_msg->child_frame_id = "lidar_link";
    odom_msg->header.frame_id = "map";
    odom_msg->header.stamp = msg->header.stamp;
    odom_msg->pose.pose = msg->pose;
    handler_lidar_odom(odom_msg);
  }

  void handler_lidar_odom(nav_msgs::OdometryConstPtr msg) {
    std::lock_guard<std::mutex> glock(lidar_odom_lock_);

    if (lidar_odom_.size() > 2000) {
      lidar_odom_.erase(lidar_odom_.begin());
    }

    auto ts = msg->header.stamp.toSec();
    lidar_odom_[ts] = *msg;

    Eigen::Vector3d robot_pos(msg->pose.pose.position.x,
                              msg->pose.pose.position.y,
                              msg->pose.pose.position.z);

    Eigen::Quaterniond robot_rot(
        msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);

    compute_robot_status(robot_pos, robot_rot);
  }

  nav_msgs::OdometryConstPtr getOdom(double ts, double err) {
    std::lock_guard<std::mutex> g(lidar_odom_lock_);
    if (lidar_odom_.empty()) return nullptr;
    // first >= ts
    auto it = lidar_odom_.lower_bound(ts);
    std::array<decltype(it), 2> cands{it, it};
    if (it != lidar_odom_.begin()) cands[1] = std::prev(it);
    // find closest
    auto best = lidar_odom_.end();
    double bestErr = err;
    for (auto c : cands) {
      double e = std::fabs(c->first - ts);
      if (e < bestErr) {
        bestErr = e;
        best = c;
      }
    }
    if (best == lidar_odom_.end()) return nullptr;
    auto ret = boost::make_shared<const nav_msgs::Odometry>(best->second);
    lidar_odom_.erase(lidar_odom_.begin(), best);
    return ret;
  }

  /// @brief Main
  /// @param msg
  void handler_lidar(sensor_msgs::PointCloud2ConstPtr msg) {
    // get odometry
    auto odom = getOdom(msg->header.stamp.toSec(), 1.0);
    if (odom == nullptr) {
      std::cout << "CANNOT find lidar odom" << std::endl;
      return;
    }
    Eigen::Vector3d robot_pos(odom->pose.pose.position.x,
                              odom->pose.pose.position.y, 0);

    Eigen::Quaterniond robot_rot(
        odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
        odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);

    robot_shake_filter_->push(robot_rot);
    Eigen::Quaterniond robot_rot_filtered =
        robot_shake_filter_->getFilteredQuat();

    // visualize robot
    {
      puber_robot_marker_.publish(
          make_robot_maker(robot_pos, robot_rot_filtered));
    }

    // get pathes
    nav_msgs::Path::ConstPtr free_path;
    nav_msgs::Path::ConstPtr left_path;
    nav_msgs::Path::ConstPtr right_path;
    {
      std::unique_lock<std::mutex> ulock1(free_path_lock_, std::defer_lock);
      std::unique_lock<std::mutex> ulock2(left_obstacle_path_lock_,
                                          std::defer_lock);
      std::unique_lock<std::mutex> ulock3(right_obstacle_path_lock_,
                                          std::defer_lock);
      std::lock(ulock1, ulock2, ulock3);

      free_path = free_path_;
      left_path = left_obstacle_path_;
      right_path = right_obstacle_path_;

      free_path_origin_index = -1;
      left_path_origin_index = -1;
      right_path_origin_index = -1;
    }

    // pre-process pathes
    {
      if (free_path != nullptr) {
        bool ret_free = crop_path_boost(robot_pos, free_path, cropped_path_,
                                        free_path_origin_index);
        if (ret_free) {
          auto msg_free_path = make_path(cropped_path_.getPath());
          puber_debug_cropped_path_.publish(msg_free_path);
        } else {
          std::cout << "[WARN] No Middle path to crop!" << std::endl;
        }
      }

      if (left_path != nullptr) {
        bool ret_left = crop_path_boost(
            robot_pos, left_path, cropped_left_path_, left_path_origin_index);
        if (ret_left) {
          auto msg_left_path = make_path(cropped_left_path_.getPath());
          puber_debug_cropped_left_path_.publish(msg_left_path);
        } else {
          std::cout << "[WARN] No Left/Right path to crop!" << std::endl;
        }
      }

      if (right_path != nullptr) {
        bool ret_right =
            crop_path_boost(robot_pos, right_path, cropped_right_path_,
                            right_path_origin_index);

        if (ret_right) {
          auto msg_right_path = make_path(cropped_right_path_.getPath());
          puber_debug_cropped_right_path_.publish(msg_right_path);
        } else {
          std::cout << "[WARN] No Left/Right path to crop!" << std::endl;
        }
      }
    }

    // sample pathes
    auto sampled_pathes = sampler_.sample(robot_pos, robot_rot_filtered);

    // visualize path group
    {
      auto path_group_msg = make_path_group(sampled_pathes);
      puber_path_group_.publish(path_group_msg);
    }

    // convert to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pcl_cloud);

    // crop it!
    pcl::PointCloud<pcl::PointXYZ>::Ptr croped_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    {
      auto maplength = elevation_map.getLength();
      double valid_radius_sq = std::pow(maplength.x, 2);
      double valid_inner_radius_sq = std::pow(0.3, 2);
      std::copy_if(pcl_cloud->begin(), pcl_cloud->end(),
                   std::back_inserter(croped_cloud->points),
                   [&](const pcl::PointXYZ& p) {
                     return std::isfinite(p.x) && std::isfinite(p.y) &&
                            p.x > 1.0 &&
                            (p.x * p.x + p.y * p.y) <= valid_radius_sq;
                   });
      // (p.x * p.x + p.y * p.y) > valid_inner_radius_sq
    }

    // transfrom
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    {
      auto _t = Eigen::Vector3f(odom->pose.pose.position.x,
                                odom->pose.pose.position.y,
                                odom->pose.pose.position.z);
      auto _q = Eigen::Quaternionf(
          odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
          odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
      pcl::transformPointCloud(*croped_cloud, *aligned_cloud, _t, _q);
    }

    elevation_map.setOriginCenter({odom->pose.pose.position.x,
                                   odom->pose.pose.position.y,
                                   odom->pose.pose.position.z});
    obstacle_map.setOriginCenter(
        {odom->pose.pose.position.x, odom->pose.pose.position.y, 0});

    {
      if (puber_debug_point_cloud_.getNumSubscribers() > 0) {
        sensor_msgs::PointCloud2 _msg;
        pcl::toROSMsg(*aligned_cloud, _msg);
        _msg.header.frame_id = "map";
        puber_debug_point_cloud_.publish(_msg);
      }
    }

    // fill current
    {
      // tbb::parallel_for(
      //     tbb::blocked_range<size_t>(0, aligned_cloud->size(), 500),
      //     [&](const tbb::blocked_range<size_t>& r) {
      //       for (size_t i = r.begin(); i != r.end(); ++i) {
      //         const auto& p = aligned_cloud->points[i];
      //         elevation_map.setGrid(rvoxel::Position{p.x, p.y, p.z},
      //                               GridElevation());
      //       }
      //     });
      ScopedTimerMS timer("BuildElevationMap");
      elevation_map.clear();
      for (auto&& p : aligned_cloud->points) {
        auto successs = elevation_map.setGrid(rvoxel::Position{p.x, p.y, p.z},
                                              GridElevation());
      }
    }

    // remove ground
    // {
    //   rvoxel::RingVoxelMapGroundIterator ground_iterator(elevation_map);
    //   ground_iterator.setGroundThickness(2);
    //   auto fn = [](rvoxel::Index idx, GridElevation& grid) {
    //     grid.setValid(false);
    //   };
    //   ground_iterator.traverse(fn);
    // }

    // generate obstacle map
    {
      ScopedTimerMS timer("GenerateObstacleMap");
      obstacle_map.clear();
      rvoxel::RingVoxelMapObstacleIterator obstacle_iterator(elevation_map);
      obstacle_iterator.setGroundThickness(2);
      obstacle_iterator.setObstacleThickness(1);
      auto fn = [&](rvoxel::Index idx, GridElevation& grid) {
        rvoxel::Index obstacle_idx{idx.x, idx.y, 0};
        obstacle_map.setGrid(obstacle_idx, GridObstacle());
      };

      auto gen_left_obs = [&]() {
        nav_msgs::Path::ConstPtr obs_path;
        {
          std::lock_guard<std::mutex> glock(left_obstacle_path_lock_);
          obs_path = left_obstacle_path_;
        }
        if (obs_path == nullptr) {
          return;
        }

        for (auto&& p : obs_path->poses) {
          obstacle_map.setGrid(
              rvoxel::Position{p.pose.position.x, p.pose.position.y, 0},
              GridObstacle());
        }
      };

      auto gen_right_obs = [&]() {
        nav_msgs::Path::ConstPtr obs_path;
        {
          std::lock_guard<std::mutex> glock(right_obstacle_path_lock_);
          obs_path = right_obstacle_path_;
        }
        if (obs_path == nullptr) {
          return;
        }

        for (auto&& p : obs_path->poses) {
          obstacle_map.setGrid(
              rvoxel::Position{p.pose.position.x, p.pose.position.y, 0},
              GridObstacle());
        }
      };

      gen_left_obs();
      gen_right_obs();

      obstacle_iterator.traverse(fn);
    }

    // visualizaiton
    {
      ScopedTimerMS timer("VisualizeEvevationMap");
      auto msg0 = make_ringvoxel_map_cloud(elevation_map);
      auto msg1 = make_ringvoxel_map_cloud(obstacle_map);
      puber_debug_local_map_.publish(msg0);
      puber_debug_obstacle_map_.publish(msg1);
    }

    // select path
    {
      ScopedTimerMS timer("SelectPath");
      select_path(sampled_pathes, robot_pos, robot_rot_filtered);
    }

    std::cout << "-------------------------------" << std::endl;
  }

  void compute_distance_to_path(const Eigen::Vector3d& robot_pos,
                                const pathlib::Path& path) {
    double min_dis2 = std::numeric_limits<double>::max();
    auto& pts = path.getPath();
    for (size_t i = 0; i < pts.size(); i++) {
      double dx = robot_pos.x() - pts[i].x();
      double dy = robot_pos.y() - pts[i].y();
      double dis2 = dx * dx + dy * dy;
      if (dis2 < min_dis2) {
        min_dis2 = dis2;
      }
    }

    ctx_distance_to_free_path_ = std::sqrt(min_dis2);

    {
      // y=f(x)=ax^2
      const double a = kFreePathPaddingEdgeProbability /
                       (kFreePathPadding * kFreePathPadding);
      p_one_tracking =
          a * (ctx_distance_to_free_path_ * ctx_distance_to_free_path_);
      p_one_tracking = std::clamp<double>(p_one_tracking, 0.0, 1.0);
    }
  }

  bool crop_path_boost(const Eigen::Vector3d& robot_pos,
                       nav_msgs::Path::ConstPtr path,
                       pathlib::Path& cropped_path, int& ahead_index) {
    if (path == nullptr || path->poses.empty()) {
      return false;
    }

    // 寻找离机器人最近的起点索引
    int origin_index = -1;
    double min_dis2 = std::numeric_limits<double>::max();
    for (size_t i = 0; i < path->poses.size(); i++) {
      double dx = robot_pos.x() - path->poses[i].pose.position.x;
      double dy = robot_pos.y() - path->poses[i].pose.position.y;
      double dis2 = dx * dx + dy * dy;
      if (dis2 < min_dis2) {
        min_dis2 = dis2;
        origin_index = i;
      }
    }

    // 基础校验：至少需要两个点才能形成线段
    if (origin_index < 0 ||
        origin_index >= static_cast<int>(path->poses.size()) - 1) {
      return false;
    }

    // Find the back path
    int begin_index = 0;
    {
      double acc_dis = 0.0;
      for (int i = origin_index; i > 0; i--) {
        const auto& p1 = path->poses[i - 1].pose.position;
        const auto& p2 = path->poses[i].pose.position;
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double len = std::sqrt(dx * dx + dy * dy);
        acc_dis += len;
        begin_index = i;
        if (acc_dis >= kFreePathBhead) {
          break;
        }
      }
    }

    ahead_index = origin_index - begin_index;

    // 线性插值与路径截取
    auto& point_buffer = cropped_path.getPath();
    point_buffer.clear();

    double acc_dis = 0.0;
    for (size_t i = begin_index; i < path->poses.size() - 1; i++) {
      const auto& p1 = path->poses[i].pose.position;
      const auto& p2 = path->poses[i + 1].pose.position;

      double dx = p2.x - p1.x;
      double dy = p2.y - p1.y;
      double segment_len = std::sqrt(dx * dx + dy * dy);

      if (segment_len < 0.01) continue;  // 跳过重合点

      // 计算当前段需要插多少个点
      int num_steps = std::ceil(segment_len / kInterpolationPathResolution);
      double actual_step = segment_len / num_steps;  // 均匀分配步长

      for (int j = 0; j < num_steps; ++j) {
        // 计算插值比例 t [0, 1)
        double t = static_cast<double>(j) / num_steps;
        double interp_x = p1.x + t * dx;
        double interp_y = p1.y + t * dy;

        point_buffer.emplace_back(interp_x, interp_y, 0.0);

        // 检查累计距离是否达到上限
        acc_dis += actual_step;
        if (acc_dis >= kFreePathAhead) {
          return true;
        }
      }
    }

    // 如果运行到这里，说明整条路径跑完也没达到 kFreePathAhead，
    // 把最后一个原始点补上
    const auto& last_p = path->poses.back().pose.position;
    point_buffer.emplace_back(last_p.x, last_p.y, 0.0);

    return !point_buffer.empty();
  }

  std::vector<LineParam> compute_parallel_lines(
      const Eigen::Vector3d& robot_pos, const Eigen::Quaterniond& robot_rot,
      double d_min, double d_max, int count) {
    std::vector<LineParam> lines;
    lines.reserve(count);

    auto forward_axis = robot_rot * Eigen::Vector3d(1, 0, 0);
    double yaw = std::atan2(forward_axis.y(), forward_axis.x());
    Eigen::Rotation2Dd rot(yaw);

    Eigen::Vector2d robot_p2d(robot_pos.x(), robot_pos.y());
    Eigen::Vector2d unit_forward =
        rot * Eigen::Vector2d(1.0, 0.0);  // 机器人正前方单位向量
    Eigen::Vector2d line_vec =
        rot * Eigen::Vector2d(0.0, 1.0);  // 直线方向单位向量

    double dd = (count > 1) ? (d_max - d_min) / (count - 1) : 0;

    for (int i = 0; i < count; i++) {
      double d = d_min + i * dd;
      // p0 = 机器人位置 + 朝前走 d 距离
      Eigen::Vector2d p0 = robot_p2d + unit_forward * d;
      lines.push_back({p0, line_vec});
    }
    return lines;
  }

  double compute_dis_to_line(const LineParam& line, const Eigen::Vector2d& pt) {
    Eigen::Vector2d p0_p = pt - line.p0;
    double cross_product =
        p0_p.x() * line.n_vec.y() - p0_p.y() * line.n_vec.x();
    return std::abs(cross_product) / line.n_vec.norm();
  }

  std::vector<Eigen::Vector2d> sync_path_points(
      const pathlib::Path& path, const std::vector<LineParam>& lines) {
    auto& points = path.getPath();
    std::vector<Eigen::Vector2d> result;
    for (auto&& l : lines) {
      double min_dis = 1e9;
      int min_dis_idx = 0;
      for (size_t i = 0; i < points.size(); i++) {
        double dis = compute_dis_to_line(l, points[i].head<2>());
        if (dis < min_dis) {
          min_dis = dis;
          min_dis_idx = i;
        }
      }
      result.push_back(points[min_dis_idx].head(2));
    }
    return result;
  }

  double compute_score_unparallelism(
      const pathlib::Path& path, const std::vector<LineParam>& lines,
      const std::vector<Eigen::Vector2d>& ref_path) {
    double score = 0;
    auto synced_path = sync_path_points(path, lines);

    if (ref_path.size() != synced_path.size()) {
      std::cerr << "Fatal! Synced path has different length" << std::endl;
      return 0;
    }

    double initial_dis = -1;
    for (size_t i = 0; i < ref_path.size(); i++) {
      double dis = (ref_path.at(i) - synced_path.at(i)).norm();
      if (initial_dis < 0) {
        initial_dis = dis;
      } else {
        score += std::abs(dis - initial_dis);
      }
    }
    return score;
  }

  double compute_valid_path_range(const Eigen::Vector3d robot_pos,
                                  const Eigen::Quaterniond& robot_rot,
                                  const pathlib::Path& ref_path) {
    LineParam line;

    auto forward_axis = robot_rot * Eigen::Vector3d(1, 0, 0);
    double yaw = std::atan2(forward_axis.y(), forward_axis.x());
    Eigen::Rotation2Dd rot(yaw);

    Eigen::Vector2d robot_p2d(robot_pos.x(), robot_pos.y());
    Eigen::Vector2d line_vec =
        rot * Eigen::Vector2d(0.0, 1.0);  // 直线方向单位向量
    line.p0 = robot_p2d;
    line.n_vec = line_vec;

    auto& pts = ref_path.getPath();
    double max_dis = -1e9;
    for (size_t i = 0; i < pts.size(); i++) {
      auto dis = compute_dis_to_line(line, pts[i].head<2>());
      if (dis > max_dis) {
        max_dis = dis;
      }
    }

    if (max_dis > kSamplePathLength) {
      max_dis = kSamplePathLength;
    }

    return max_dis;
  }

  size_t compute_valid_path_idx(const pathlib::Path& path, double valid_range) {
    double acc_dis = 0;
    auto& pts = path.getPath();
    for (size_t i = 1; i < pts.size(); i++) {
      double dx = pts[i].x() - pts[i - 1].x();
      double dy = pts[i].y() - pts[i - 1].y();
      double dis = std::sqrt(dx * dx + dy * dy);
      acc_dis += dis;
      if (acc_dis >= valid_range) {
        return i;
      }
    }
    return pts.size();
  }

  double compute_score_safety(
      const pathlib::Path& path, int beg_idx, int end_idx,
      std::vector<Eigen::Vector3d>& draw_collision_range) {
    auto& points = path.getPath();
    double closest_obstacle_distance = 1e7;
    // for (auto& pt : points) {
    for (int i = beg_idx; i < end_idx; i++) {
      auto& pt = points[i];

      auto idx = obstacle_map.toIndex({pt.x(), pt.y(), pt.z()});
      for (auto&& i : collision_range_) {
        rvoxel::Index test_idx;
        test_idx.x = idx.x + i.x;
        test_idx.y = idx.y + i.y;
        test_idx.z = 0;

        {
          auto pos = obstacle_map.toPosition(test_idx);
          draw_collision_range.emplace_back(pos.x, pos.y, pos.z);
        }

        if (obstacle_map.isInside(test_idx)) {
          auto obstacle_pos = obstacle_map.toPosition(test_idx);
          double dx = pt.x() - obstacle_pos.x;
          double dy = pt.y() - obstacle_pos.y;
          double dis = std::sqrt(dx * dx + dy * dy);
          if (dis < closest_obstacle_distance) {
            closest_obstacle_distance = dis;
          }
        }
      }
    }
    return closest_obstacle_distance;
  }

  double compute_score_tracking(const pathlib::Path& path,
                                const pathlib::Path& ref_path, int ahead_idx) {
    // const auto final_pt = path.getPath().back();
    assert(ahead_idx >= 0);
    assert(ahead_idx < path.getPath().size());

    const auto final_pt = path.getPath().at(ahead_idx);
    const auto& pts = ref_path.getPath();
    double min_dis = 1e9;
    for (size_t i = 0; i < pts.size(); i++) {
      double dx = pts[i].x() - final_pt.x();
      double dy = pts[i].y() - final_pt.y();
      double dis = dx * dx + dy * dy;
      if (dis < min_dis) {
        min_dis = dis;
      }
    }
    return std::sqrt(min_dis);
  }

  double compute_score_energy(const Eigen::Vector3d& robot_pos,
                              const Eigen::Quaterniond& robot_rot,
                              pathlib::Path& path, int ahead_idx) {
    path.setControlParam(robot_pos, robot_rot, ahead_idx);
    double yaw_error = path.compute_yaw_error();
    return yaw_error / M_PI;
  }

  void compute_robot_status(const Eigen::Vector3d& robot_pos,
                            const Eigen::Quaterniond& robot_rot) {
    static thread_local Eigen::Vector3d last_robot_pos;
    static thread_local Eigen::Quaterniond last_robot_rot;
    static thread_local double last_ts = 0;
    static thread_local bool first = true;

    double ts = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();

    if (first) {
      last_robot_pos = robot_pos;
      last_robot_rot = robot_rot;
      last_ts = ts;
      first = false;
      return;
    }

    double dt = ts - last_ts;

    double delta_movement = (last_robot_pos - robot_pos).norm();
    Eigen::Vector3d xAxis = Eigen::Vector3d::UnitX();
    Eigen::Vector3d last_head = last_robot_rot * xAxis;
    Eigen::Vector3d curr_head = robot_rot * xAxis;
    double last_yaw = std::atan2(last_head.y(), last_head.x());
    double curr_yaw = std::atan2(last_head.y(), last_head.x());
    double delta_rotation =
        angles::shortest_angular_distance(last_yaw, curr_yaw);

    last_robot_pos = robot_pos;
    last_robot_rot = robot_rot;
    last_ts = ts;

    robot_vel_x.store(delta_movement / dt, std::memory_order_relaxed);
    robot_vel_z.store(delta_rotation / dt, std::memory_order_relaxed);
  }

  // tag:curvature
  double compute_velocity_by_curvature(double curvature) {
    if (curvature < 0.5) {
      return kRobotMaxSpeed;
    } else if (curvature < 0.8) {
      return 2.7;
    } else if (curvature < 1.2) {
      return 2.5;
    } else if (curvature < 1.5) {
      return 2.0;
    } else {
      return 1.2;
    }
  }

  double compute_velocity_by_road_width(double road_width) {
    // TODO: road_width
    return 0;
  }

  int ahead_to_index(double ahead) {
    double k = ahead / kSamplePathLength;
    return static_cast<int>(kPathMaxAheadPointIdx * k);
  }

  double compute_ahead_by_curvature(double curvature) {
    if (curvature < 0.5) {
      return kSamplePathLength;
    } else if (curvature < 0.8) {
      return 4.0;
    } else if (curvature < 1.2) {
      return 3.0;
    } else if (curvature < 1.5) {
      return 2.0;
    } else {
      return 1.5;
    }
  }

  double compute_velocity_by_ahead(double ahead) {
    // ahead: meter
    if (ahead < 1.5) {
      return 1.5;
    } else if (ahead < 3.0) {
      return 2.0;
    } else if (ahead < 5.0) {
      return 2.5;
    } else {
      return kRobotMaxSpeed;
    }
  }

  double compute_omega_gain_by_curvature(double C) {
    if (C < 0.8) {
      return 1.0;
    } else if (C < 1.5) {
      return 1.2;
    } else if (C < 2.0) {
      return 1.5;
    } else {
      return 2.0;
    }
  }

  void select_path(std::vector<pathlib::Path>& sampled_pathes,
                   const Eigen::Vector3d& robot_pos,
                   const Eigen::Quaterniond& robot_rot) {
    if (free_path_origin_index < 0) {
      std::cout << "[FATAL] No free path!" << std::endl;
      this->crash();
      return;
    }

    // compute the curvature of cropped path
    compute_distance_to_path(robot_pos, cropped_path_);

    // compute auxiliary lines
    auto lines = compute_parallel_lines(
        robot_pos, robot_rot, kSamplePathInnerLength, kSamplePathLength, 30);

    // compute curvatures
    double cropped_path_full_curvature = cropped_path_.compute_curvature();
    double cropped_path_front_curvature =
        cropped_path_.compute_curvature(free_path_origin_index);

    {
      std_msgs::Float64MultiArray msg;
      msg.data.push_back(cropped_path_full_curvature);
      msg.data.push_back(cropped_path_front_curvature);
      puber_debug_cropped_path_curvature_.publish(msg);
    }

    // compute dynamic variables

    double dyn_ahead_length =
        compute_ahead_by_curvature(cropped_path_front_curvature);

    double dyn_velocity_ahead = compute_velocity_by_ahead(dyn_ahead_length);

    int dyn_ahead_idx = ahead_to_index(dyn_ahead_length);

    auto dyn_omega_gain =
        compute_omega_gain_by_curvature(cropped_path_front_curvature);

    double valid_path_range =
        compute_valid_path_range(robot_pos, robot_rot, cropped_path_);

    {
      auto msg = make_collision_range_marker(robot_pos.x(), robot_pos.y(), 0,
                                             valid_path_range);
      puber_debug_collision_range_.publish(msg);
    }

    auto synced_cropped_path = sync_path_points(cropped_path_, lines);

    std::vector<double> scores_unparallelism;
    double min_score_unparallelism = std::numeric_limits<double>::max();
    double max_score_unparallelism = std::numeric_limits<double>::lowest();
    int min_score_unparallelism_idx = 0;
    int max_score_unparallelism_idx = 0;
    scores_unparallelism.reserve(sampled_pathes.size());

    std::vector<double> scores_safety;
    double min_score_safety = std::numeric_limits<double>::max();
    double max_score_safety = std::numeric_limits<double>::lowest();
    int min_score_safety_idx = 0;
    int max_score_safety_idx = 0;
    scores_safety.reserve(sampled_pathes.size());

    std::vector<double> scores_tracking;
    double min_score_tracking = std::numeric_limits<double>::max();
    double max_score_tracking = std::numeric_limits<double>::lowest();
    int min_score_tracking_idx = 0;
    int max_score_tracking_idx = 0;
    scores_tracking.reserve(sampled_pathes.size());

    std::vector<double> scores_energy;
    double min_score_energy = std::numeric_limits<double>::max();
    double max_score_energy = std::numeric_limits<double>::lowest();
    int min_score_energy_idx = 0;
    int max_score_energy_idx = 0;
    scores_energy.reserve(sampled_pathes.size());

    std::vector<Eigen::Vector3d> draw_collision_range_points;
    draw_collision_range_points.reserve(100 * 9 * 20 * 3);

    for (int i = 0; i < (int)sampled_pathes.size(); i++) {
      auto& path = sampled_pathes[i];

      size_t beg_idx = compute_valid_path_idx(path, kSamplePathInnerLength);
      size_t end_idx = compute_valid_path_idx(path, valid_path_range);
      double score_unparallelism =
          compute_score_unparallelism(path, lines, synced_cropped_path);

      double score_safety = compute_score_safety(path, beg_idx, end_idx,
                                                 draw_collision_range_points);

      double score_tracking =
          compute_score_tracking(path, cropped_path_, dyn_ahead_idx);

      double score_energy =
          compute_score_energy(robot_pos, robot_rot, path, dyn_ahead_idx);

      scores_unparallelism.push_back(score_unparallelism);
      scores_safety.push_back(score_safety);
      scores_tracking.push_back(score_tracking);
      scores_energy.push_back(score_energy);

      if (score_unparallelism < min_score_unparallelism) {
        min_score_unparallelism = score_unparallelism;
        min_score_unparallelism_idx = i;
      }
      if (score_unparallelism > max_score_unparallelism) {
        max_score_unparallelism = score_unparallelism;
        max_score_unparallelism_idx = i;
      }

      if (score_safety < min_score_safety) {
        min_score_safety = score_safety;
        min_score_safety_idx = i;
      }
      if (score_safety > max_score_safety) {
        max_score_safety = score_safety;
        max_score_safety_idx = i;
      }

      if (score_tracking < min_score_tracking) {
        min_score_tracking = score_tracking;
        min_score_tracking_idx = i;
      }
      if (score_tracking > max_score_tracking) {
        max_score_tracking = score_tracking;
        max_score_tracking_idx = i;
      }

      if (score_energy < min_score_energy) {
        min_score_energy = score_energy;
        min_score_energy_idx = i;
      }
      if (score_energy > max_score_energy) {
        max_score_energy = score_energy;
        max_score_energy_idx = i;
      }
    }

    // {
    //   auto msg = make_point_cloud(draw_collision_range_points);
    //   puber_debug_voxel_collision_range_.publish(msg);
    // }

    if (ctx_planner_mode == PlannerMode::RECOVERY) {
      recover_process(robot_pos, robot_rot, max_score_safety);
      return;
    }

    // main decision logic
    int best_tracking_path_idx = -1;
    int final_path_idx = -1;
    int final_path_idx_parallelism = -1;
    bool need_to_avoid_obstacles = false;
    double obstacle_distance = 1e5;

    // avoidance detection
    std::vector<int> sorted_scores_tracking_idx;
    sorted_scores_tracking_idx.resize(scores_tracking.size());
    const int topK = 4;
    {
      // find the pathes of the best score of tracking
      std::iota(sorted_scores_tracking_idx.begin(),
                sorted_scores_tracking_idx.end(), 0);
      std::sort(sorted_scores_tracking_idx.begin(),
                sorted_scores_tracking_idx.end(), [&](int a, int b) {
                  return scores_tracking[a] < scores_tracking[b];
                });

      best_tracking_path_idx = sorted_scores_tracking_idx[0];

      {
        int safe_count = 0;
        for (size_t i = 0; i < topK; i++) {
          int idx = sorted_scores_tracking_idx[i];
          if (scores_safety[idx] >= kAvoidanceModeObstacleDistance) {
            safe_count++;
          }
          if (scores_safety[idx] < obstacle_distance) {
            obstacle_distance = scores_safety[idx];
          }
        }

        double safe_rate = (double)safe_count / (double)topK;
        if (safe_rate < 0.8) {
          need_to_avoid_obstacles = true;
        }
      }
    }

    if (need_to_avoid_obstacles) {
      ctx_planner_mode = PlannerMode::AVOIDANCE;
      std::cout << "WARNING: Ready to crash!" << std::endl;
      std::cout << "MaxScoreSafety: " << max_score_safety << "/"
                << kFatalMinScoreSafety << std::endl;

      if (max_score_safety < kFatalMinScoreSafety) {
        std::cout << "WARNING: CANNOT pass it, stop!" << std::endl;
        final_path_idx = -1;
        this->crash();
      } else {
        std::cout << "WARNING: Tring to get across the obstacles!" << std::endl;

        // compute the candidates
        std::vector<int> candidates;
        {
          candidates.reserve(sampled_pathes.size());
          for (int i = 0; i < (int)scores_safety.size(); i++) {
            bool drop = false;
            for (int j = 0; j < topK; j++) {
              auto idx = sorted_scores_tracking_idx[j];
              if (i == idx) {
                drop = true;
                break;
              }
            }

            if (!drop && scores_safety[i] >= kAvoidanceModeObstacleDistance) {
              candidates.push_back(i);
            }
          }
        }

        if (candidates.empty()) {
          std::cout << "WARNING: Tried, but failed to pass it, soft stop!"
                    << std::endl;
          final_path_idx = -1;
          this->crash();
          return;
        }

        // tag:avoidance
        {
          // solution0:
          // {
          //   if (candidates.empty()) {
          //     this->crash();
          //   }

          //   double min_tracking_score = 1e9;
          //   for (auto&& idx : candidates) {
          //     if (scores_tracking[idx] < min_tracking_score) {
          //       min_tracking_score = min_tracking_score;
          //       final_path_idx = idx;
          //     }
          //   }
          // }

          // solution1: choose the energy balanced path
          {
            double min_cost = 1e9;
            double final_energy = 0;
            for (auto&& idx : candidates) {
              double energy = scores_energy[idx];
              double cost =
                  std::abs(energy) +
                  kWeightEnergyConsistency * std::abs(energy - ctx_last_energy);
              if (cost < min_cost) {
                min_cost = cost;
                final_path_idx = idx;
                final_energy = energy;
              }
            }
            ctx_last_energy = final_energy;
          }

          // avoidance yaw_error gain
          {
            // normalize to [0, 1]
            double k = std::clamp(obstacle_distance, 0.0,
                                  kAvoidanceModeObstacleDistance) /
                       kAvoidanceModeObstacleDistance;

            ctx_yaw_error_gain = 1.5 * std::pow(k, 2) + 1;
            ctx_yaw_error_gain = std::clamp(ctx_yaw_error_gain, 1.0, 2.0);
          }
        }
      }
    }
    // -----------------------------------------------------
    // Definitely Safe
    // -----------------------------------------------------
    else {
      ctx_last_energy = 0;
      ctx_planner_mode = PlannerMode::TRACKING;
      ctx_yaw_error_gain = 1.0;
      std::vector<int> candidates;
      candidates.reserve(sampled_pathes.size());
      for (int i = 0; i < (int)scores_safety.size(); i++) {
        if (scores_safety[i] > kMinScoreSafety) {
          candidates.push_back(i);
        }
      }

      std::cout << "CandidateCount=" << candidates.size() << std::endl;

      {
        {
          double min_score_tracking = std::numeric_limits<double>::max();
          for (auto&& idx : candidates) {
            if (scores_tracking[idx] < min_score_tracking) {
              min_score_tracking = scores_tracking[idx];
              final_path_idx = idx;
            }
          }
        }

        {
          double min_score_unparallelism = std::numeric_limits<double>::max();
          for (auto&& idx : candidates) {
            if (scores_unparallelism[idx] < min_score_unparallelism) {
              min_score_unparallelism = scores_unparallelism[idx];
              final_path_idx_parallelism = idx;
            }
          }
        }
      }
    }

    // tag:output
    //  Compute control value
    if (final_path_idx >= 0) {
      auto& path = sampled_pathes[final_path_idx];

      {
        auto msg = make_path(path.getPath());
        puber_path_.publish(msg);
      }

      // int control_ahead_idx = compute_valid_path_idx(path, valid_path_range);
      path.setControlParam(robot_pos, robot_rot, dyn_ahead_idx);
      {
        auto ahead_pt = path.getAheadPoint();
        auto msg_ahead_pt =
            make_target_point_marker(ahead_pt.x(), ahead_pt.y(), ahead_pt.z());
        puber_ahead_pt_.publish(msg_ahead_pt);
      }
      auto control_info = path.getControlInfo();

      // compute the second info
      // if (final_path_idx_parallelism >= 0) {
      //   auto& path = sampled_pathes[final_path_idx_parallelism];
      //   path.setControlParam(robot_pos, robot_rot, kPathAheadPointIdx);
      //   {
      //     auto ahead_pt = path.getAheadPoint();
      //     auto msg_ahead_pt = make_target_point_marker(
      //         ahead_pt.x(), ahead_pt.y(), ahead_pt.z());
      //     puber_ahead_pt_.publish(msg_ahead_pt);
      //   }
      //   auto control_info_2nd = path.getControlInfo();

      //   control_info.yaw_error =
      //       p_one_tracking * control_info.yaw_error +
      //       (1 - p_one_tracking) * control_info_2nd.yaw_error;
      // }

      compute_vel({kRobotMaxSpeed, dyn_velocity_ahead},
                  control_info.yaw_error * ctx_yaw_error_gain);

      std::cout << "------------------------" << std::endl;
      std::cout << "Curvature: " << cropped_path_front_curvature << std::endl;
      std::cout << "Ahead: " << dyn_ahead_length << std::endl;
      std::cout << "Vel[Ahead]: " << dyn_velocity_ahead << std::endl;
      std::cout << "Vel[Max]: " << kRobotMaxSpeed << std::endl;
      std::cout << "YawError: " << control_info.yaw_error << std::endl;

      std::cout << "Selected Path" << std::endl;
    } else {
      std::cout << "Failed to plan" << std::endl;
    }
  }

  void compute_vel(std::vector<double> speeds, double yaw_error) {
    double velx = *std::min_element(speeds.begin(), speeds.end());
    double velz = 0;

    if (ctx_planner_mode == PlannerMode::AVOIDANCE) {
      if (std::abs(yaw_error) < angles::from_degrees(5)) {
        velz = 0.3 * std::copysign(1.5, yaw_error);
      } else if (std::abs(yaw_error) < angles::from_degrees(10)) {
        velz = 5.0 * yaw_error;
      } else {
        velz = std::copysign(1.5, yaw_error);
      }
      ke.setMaxIteration(20);
      ke.setAdjustRange(0, 1.51, 2.29);
      ke.setAdjustRange(1, -1.49, 1.49);
    } else if (ctx_planner_mode == PlannerMode::TRACKING) {
      if (std::abs(yaw_error) < angles::from_degrees(5)) {
        velz = 1.0 * yaw_error;
      } else if (std::abs(yaw_error) < angles::from_degrees(20)) {
        velz = 6.0 * yaw_error;
      } else {
        velz = std::copysign(2.2, yaw_error);
      }
      ke.setMaxIteration(20);
      ke.setAdjustRange(0, 2.51, 3.49);
      ke.setAdjustRange(1, -2.29, 2.29);
    } else if (ctx_planner_mode == PlannerMode::RECOVERY) {
      velz = std::copysign(0.6, yaw_error);
      ke.setMaxIteration(20);
      ke.setAdjustRange(0, 0.0, 0.5);
      ke.setAdjustRange(1, -0.8, 0.8);
    }

    auto result = ke.clip({velx, velz});
    if (!result.success) {
      ROS_WARN("[FATAL] Failed to Crop! VelX=%.4lf, VelZ=%.4lf,  Mode=%d", velx,
               velz, (int)ctx_planner_mode);
    }

    velx = result.values[0];
    velz = result.values[1];
    publish_cmd_vel(velx, velz);
  }

  void crash() {
    ctx_last_energy = 0;
    ctx_planner_mode = PlannerMode::RECOVERY;
    ctx_watch_dog_exp_ts = -1;

    std::cout << "WARNING: Crashed!!!  Start to recover." << std::endl;

    double initial_velx = ctx_last_velx;
    int delay_ms = 20;
    double stop_sec = 1.0;
    double di = stop_sec / (stop_sec / (delay_ms * 0.001));
    for (double i = 2.0; i > 0; i -= di) {
      double velx = std::tanh(i) * initial_velx;
      publish_cmd_vel(velx, 0, false);
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
  }

  void recover_process(const Eigen::Vector3d robot_pos,
                       const Eigen::Quaterniond robot_rot,
                       double score_safety) {
    // check lists:
    // - [X] check robot distance to free_path
    // - [X] check max_safty_score
    // - [X] check robot quaternion
    double now_ts = getTimestamp();
    auto reset_watch_dog = [&]() {
      ctx_watch_dog_exp_ts = now_ts + kWatchDogExpSec;
    };

    if (ctx_watch_dog_exp_ts < 0) {
      std::cout << "[Recover] Reset: ExpTs < 0" << std::endl;
      reset_watch_dog();
    }

    double left_ts = ctx_watch_dog_exp_ts - now_ts;

    if (ctx_distance_to_free_path_ > kMaxRecoveryDistanceToFreePath) {
      std::cout << "[Recover] Reset: DisToFree < "
                << kMaxRecoveryDistanceToFreePath << std::endl;
      reset_watch_dog();
    }

    std::cout << "[Recover] DisToFree: " << ctx_distance_to_free_path_
              << std::endl;
    std::cout << "[Recover] MaxScoreSafety: " << score_safety << std::endl;
    std::cout << "[Recover] TimeOutSec: " << left_ts << std::endl;

    double velx = 0.5;
    if (score_safety < kFatalMinScoreSafety) {
      velx = 0.0;
      std::cout << "[Recover] Reset: MaxSafetyScore < " << kFatalMinScoreSafety
                << std::endl;
      reset_watch_dog();
    }
    std::cout << "[Recover] VelX: " << velx << std::endl;

    if (now_ts > ctx_watch_dog_exp_ts) {
      ctx_planner_mode = PlannerMode::TRACKING;
      return;
    }

    auto& path_pts = ctx_recovery_path.getPath();

    double path_yaw = compute_path_yaw(ctx_recovery_path, robot_pos);

    for (int retry = 0; retry < 2; retry++) {
      bool plan_ok = false;
      if (is_path_valid(ctx_recovery_path)) {
        plan_ok = true;
      } else {
        plan_ok = plan_path(ctx_recovery_path, robot_pos);
      }

      if (!plan_ok) {
        std::cout << "[WARN] [Recovery] Failed to plan" << std::endl;
        publish_cmd_vel(0, 0);
        return;
      }

      // crop path
      {
        size_t start_index = 0;
        double min_dis = 1e9;
        for (size_t i = 0; i < path_pts.size(); i++) {
          auto diff = path_pts[i] - robot_pos;
          double dis = diff.head<2>().norm();
          if (dis < min_dis) {
            min_dis = dis;
            start_index = i;
          }
        }

        if (min_dis > 0.5) {
          plan_ok = plan_path(ctx_recovery_path, robot_pos);
          continue;
        }

        std::vector<Eigen::Vector3d> new_pts;
        new_pts.reserve(path_pts.size());

        for (size_t i = start_index; i < path_pts.size(); i++) {
          new_pts.push_back(path_pts[i]);
        }
        path_pts = new_pts;
      }

      // compute length
      {
        double acc_dis = 0;
        for (size_t i = 1; i < path_pts.size(); i++) {
          auto diff = path_pts[i] - path_pts[i - 1];
          acc_dis += diff.head<2>().norm();
        }

        if (acc_dis < kMinRecoveryPathLength) {
          plan_ok = plan_path(ctx_recovery_path, robot_pos);
          continue;
        }
      }
      break;
    }

    {
      auto msg = make_path(path_pts);
      puber_recover_path_.publish(msg);
    }

    size_t ahead_idx = 0;
    {
      double acc_dis = 0;
      for (size_t i = 1; i < path_pts.size(); i++) {
        auto diff = path_pts[i] - path_pts[i - 1];
        double dis = diff.norm();
        acc_dis += dis;
        ahead_idx = i;
        if (acc_dis > kRecoveryAhead) {
          break;
        }
      }
    }

    {
      auto vec = robot_rot * Eigen::Vector3d::UnitX();
      double robot_yaw = std::atan2(vec.y(), vec.x());
      double yaw_diff = angles::shortest_angular_distance(robot_yaw, path_yaw);
      std::cout << "[Recover] RobotYaw: " << robot_yaw << std::endl;
      std::cout << "[Recover] PathYaw: " << path_yaw << std::endl;
      std::cout << "[Recover] YawDiff: " << yaw_diff << std::endl;

      if (std::abs(yaw_diff) > angles::from_degrees(kMaxRecoveryYawRangeDeg)) {
        std::cout << "[Recover] Reset: YawDiff > "
                  << angles::from_degrees(kMaxRecoveryYawRangeDeg) << std::endl;
        reset_watch_dog();
      }
    }

    // compute vel and yaw
    ctx_recovery_path.setControlParam(robot_pos, robot_rot, ahead_idx);
    double yaw_error = ctx_recovery_path.compute_yaw_error();
    compute_vel({velx}, yaw_error);
  }

  bool is_path_valid(const pathlib::Path& path) {
    auto& pts = path.getPath();
    if (pts.empty()) {
      return false;
    }
    auto detect_range = generate_collision_body(kPlannerReplanDetectRange);
    for (auto&& pt : pts) {
      auto idx = obstacle_map.toIndex({pt.x(), pt.y(), 0});
      for (auto&& outter_idx : detect_range) {
        rvoxel::Index test_idx;
        test_idx.x = idx.x + outter_idx.x;
        test_idx.y = idx.y + outter_idx.y;
        test_idx.z = 0;
        if (obstacle_map.isInside(test_idx)) {
          return false;
        }
      }
    }
    return true;
  }

  double compute_path_yaw(pathlib::Path& path,
                          const Eigen::Vector3d& robot_pos) {
    Eigen::Vector3d start_point;
    Eigen::Vector3d goal_point;
    auto& pts = cropped_path_.getPath();
    size_t begin_index = 0;
    {
      double min_dis = 1e9;
      for (size_t i = 0; i < pts.size(); i++) {
        Eigen::Vector3d diff = pts[i] - robot_pos;
        double dis = diff.head<2>().norm();
        if (dis < min_dis) {
          min_dis = dis;
          begin_index = i;
        }
      }
    }
    start_point = pts[begin_index];

    double acc_dis = 0;
    for (size_t i = begin_index + 1; i < pts.size(); i++) {
      Eigen::Vector3d diff = pts[i] - pts[i - 1];
      double dis = diff.norm();
      acc_dis += dis;
      if (acc_dis >= kPlannerGoalAhead) {
        goal_point = pts[i];
        break;
      }
    }

    Eigen::Vector3d diff_vec = goal_point - start_point;
    double path_yaw = std::atan2(diff_vec.y(), diff_vec.x());
    return path_yaw;
  }

  bool plan_path(pathlib::Path& path, const Eigen::Vector3d& robot_pos) {
    planner_->reset();
    // fill in with inflation
    auto inflation_range = generate_collision_body(kPlannerInflationRange);
    for (rvoxel::RingVoxelMapIterator<GridObstacle> iter(obstacle_map);
         !iter.EOI(); iter++) {
      auto idx = *iter;

      int linear_idx = obstacle_map.getLinearIndex(idx.x, idx.y, idx.z);

      if (obstacle_map.isInside(idx)) {
        planner_->setObstacle(idx.x, idx.y, idx.z, true);
        // generate inflation cost
        for (auto&& inflation_idx : inflation_range) {
          // double d = std::sqrt(inflation_idx.x * inflation_idx.x +
          //                      inflation_idx.y * inflation_idx.y +
          //                      inflation_idx.z * inflation_idx.z);
          rvoxel::Index new_idx;
          new_idx.x = idx.x + inflation_idx.x;
          new_idx.y = idx.y + inflation_idx.y;
          new_idx.z = 0;
          planner_->setCost(new_idx.x, new_idx.y, new_idx.z, 100.0);
        }
      } else {
        planner_->setObstacle(idx.x, idx.y, idx.z, false);
      }
    }
    // query
    Eigen::Vector3d start_point;
    Eigen::Vector3d goal_point;
    slamchain::Dijkstra::Coord start;
    start.x = planner_->getWidth() / 2;
    start.y = planner_->getHeight() / 2;
    start.z = planner_->getDepth() / 2;

    slamchain::Dijkstra::Coord goal;
    {
      auto& pts = cropped_path_.getPath();

      size_t begin_index = 0;
      {
        double min_dis = 1e9;
        for (size_t i = 0; i < pts.size(); i++) {
          Eigen::Vector3d diff = pts[i] - robot_pos;
          double dis = diff.head<2>().norm();
          if (dis < min_dis) {
            min_dis = dis;
            begin_index = i;
          }
        }
      }
      start_point = pts[begin_index];

      double acc_dis = 0;
      for (size_t i = begin_index + 1; i < pts.size(); i++) {
        Eigen::Vector3d diff = pts[i] - pts[i - 1];
        double dis = diff.norm();
        acc_dis += dis;

        if (acc_dis >= kPlannerGoalAhead) {
          auto& pt = pts[i];
          goal_point = pt;
          auto idx = obstacle_map.toIndex({pt.x(), pt.y(), pt.z()});
          bool conflicted = false;
          for (auto&& i : planner_goal_collision_range_) {
            rvoxel::Index test_idx;
            test_idx.x = idx.x + i.x;
            test_idx.y = idx.y + i.y;
            test_idx.z = 0;

            if (obstacle_map.isInside(test_idx)) {
              conflicted = true;
              break;
            }
          }

          if (!conflicted) {
            goal.x = idx.x;
            goal.y = idx.y;
            goal.z = 0;
            break;
          }
        }

        if (acc_dis >= 6.0) {
          break;
        }
      }
    }
    auto result = planner_->queryAStar(start, goal);

    if (result.found) {
      std::vector<Eigen::Vector3d> path_pts;
      path_pts.reserve(result.path.size());
      for (auto&& idx : result.path) {
        auto pos = obstacle_map.toPosition({idx.x, idx.y, idx.z});
        path_pts.push_back(Eigen::Vector3d(pos.x, pos.y, pos.z));
      }
      path = pathlib::Path(path_pts);
      return true;
    } else {
      return false;
    }
  }
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "local_planner_node");
  ros::NodeHandle nh;

  local_planner_node node(&nh);
  node.start();

  ros::spin();
  return 0;
}
