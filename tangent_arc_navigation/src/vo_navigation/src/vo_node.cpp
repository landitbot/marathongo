#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <vo_navigation/ObstacleArray.h>
#include <speed_controller/SpeedCommand.h>
#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Int32.h>
#include <vo_navigation/vo_state.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <sstream>
#include <sys/stat.h>
#include <string>
#include <utility>
#include <vector>

class VONode {
    // --- 防止左右圆弧频繁切换 ---
int side_switch_counter_ = 0;
    int side_switch_counter_thresh_ = 3; // 连续3帧才允许切换
    int last_side_sign_candidate_ = 0;
public:
    VONode() : nh_("~"), stop_logger_(false), tf_listener_(tf_buffer_) {
            nh_.param("side_switch_counter_thresh", side_switch_counter_thresh_, 3);
        nh_.param("control_period_sec", control_period_sec_, 0.05);
        nh_.param("max_linear_vel", max_linear_vel_, 3.5);
        nh_.param("robot_radius", robot_radius_, 0.25);
        nh_.param("obstacle_radius_inflation", obstacle_radius_inflation_, 0.10);
        nh_.param("avoidance_range_time_sec", avoidance_range_time_sec_, 2.0);
        nh_.param("enable_speed_smoothing", enable_speed_smoothing_, true);
        nh_.param("speed_lpf_alpha", speed_lpf_alpha_, 0.25);
        nh_.param("speed_rise_limit_mps2", speed_rise_limit_mps2_, 1.0);
        nh_.param("speed_fall_limit_mps2", speed_fall_limit_mps2_, 2.5);
        nh_.param("speed_arc_sample_count", speed_arc_sample_count_, 14);
        nh_.param("speed_arc_sim_steps", speed_arc_sim_steps_, 36);
        nh_.param("speed_arc_safety_clearance", speed_arc_safety_clearance_, 0.10);
        nh_.param("lane_obstacle_width", lane_obstacle_width_, 0.20);
        nh_.param("lane_downsample_step", lane_downsample_step_, 3);
        nh_.param("arc_sim_steps", arc_sim_steps_, 80);
        nh_.param("tangent_omega_step_ratio", tangent_omega_step_ratio_, 0.05);
        nh_.param("enable_runtime_debug", enable_runtime_debug_, true);
        nh_.param("enable_debug_output", enable_debug_output_, true);
        nh_.param("enable_rviz_visualization", enable_rviz_visualization_, true);
        nh_.param("enable_lidar_obstacle_markers", enable_lidar_obstacle_markers_, true);
        nh_.param("enable_lidar_polygon_markers", enable_lidar_polygon_markers_, true);
        nh_.param("enable_lidar_velocity_markers", enable_lidar_velocity_markers_, false);
        nh_.param("lidar_obstacle_marker_min_diameter", lidar_obstacle_marker_min_diameter_, 0.08);
        nh_.param("lidar_polygon_marker_line_width", lidar_polygon_marker_line_width_, 0.05);
        nh_.param("collision_slope_deg", collision_slope_deg_, 18.0);
        nh_.param("collision_slope_ref_speed", collision_slope_ref_speed_, 3.5);
        nh_.param("goal_stop_distance", goal_stop_distance_, 0.45);
        nh_.param("enable_file_log", enable_file_log_, true);
        nh_.param<std::string>("log_dir", log_dir_, "");
        nh_.param("apply_yaw_limit_on_output", apply_yaw_limit_on_output_, false);
        nh_.param("fallback_max_yaw_rate", fallback_max_yaw_rate_, 0.85);
        nh_.param("viz_direction_vector_length", viz_direction_vector_length_, 1.5);
        nh_.param<std::string>("angular_limit_csv_path", angular_limit_csv_path_, "config/vo_angular_limits.csv");

        if (!enable_runtime_debug_) {
            enable_debug_output_ = false;
            enable_file_log_ = false;
            enable_rviz_visualization_ = false;
            enable_lidar_obstacle_markers_ = false;
            enable_lidar_polygon_markers_ = false;
            enable_lidar_velocity_markers_ = false;
        }

        nh_.param<std::string>("odom_topic", odom_topic_, "/odometry");
        nh_.param<std::string>("obstacles_topic", obstacles_topic_, "/obstacles");
        nh_.param<std::string>("subgoal_topic", subgoal_topic_, "/vo/subgoal");
        nh_.param<std::string>("left_lane_path_topic", left_lane_path_topic_, "/left/smoothed_path");
        nh_.param<std::string>("right_lane_path_topic", right_lane_path_topic_, "/right/smoothed_path");
        nh_.param<std::string>("speed_command_topic", speed_command_topic_, "/speed_command");
        nh_.param<std::string>("optimal_direction_marker_topic", optimal_direction_marker_topic_, "/vo/optimal_direction_marker");

        const std::string resolved_csv = resolveCsvPath(angular_limit_csv_path_);
        loadAngularLimitCsv(resolved_csv);
        initLogFile();
        if (enable_file_log_ && log_file_.is_open()) {
            logger_thread_ = std::thread(&VONode::logWriterThread, this);
            logger_thread_started_ = true;
        }

        odom_sub_ = nh_.subscribe(odom_topic_, 1, &VONode::odomCallback, this);
        obs_sub_ = nh_.subscribe(obstacles_topic_, 1, &VONode::obstacleCallback, this);
        subgoal_sub_ = nh_.subscribe(subgoal_topic_, 1, &VONode::subgoalCallback, this);
        left_lane_sub_ = nh_.subscribe(left_lane_path_topic_, 1, &VONode::leftLaneCallback, this);
        right_lane_sub_ = nh_.subscribe(right_lane_path_topic_, 1, &VONode::rightLaneCallback, this);
        nh_.param("vo_state_timeout_sec", vo_state_timeout_sec_, 0.5);
        nh_.param("recovery_timeout_sec", recovery_timeout_sec_, 30.0);
        vo_state_sub_ = nh_.subscribe("/vo/state", 1, &VONode::voStateCallback, this);
        vo_state_pub_ = nh_.advertise<std_msgs::Int32>("/vo/state", 4);

        cmd_pub_ = nh_.advertise<speed_controller::SpeedCommand>(speed_command_topic_, 1);
        marker_pub_ = nh_.advertise<visualization_msgs::Marker>(optimal_direction_marker_topic_, 1);
        timer_ = nh_.createTimer(ros::Duration(std::max(0.01, control_period_sec_)), &VONode::timerCallback, this);

        if (enable_debug_output_) {
            ROS_INFO("vo_node rewritten: max_v=%.2f, avoid_t=%.2f, robot_r=%.2f, obs_infl=%.2f, csv=%s",
                     max_linear_vel_, avoidance_range_time_sec_, robot_radius_,
                     obstacle_radius_inflation_, resolved_csv.c_str());
        }
    }

    ~VONode() {
        stop_logger_ = true;
        queue_cv_.notify_one();
        if (logger_thread_started_ && logger_thread_.joinable()) {
            logger_thread_.join();
        }
        if (log_file_.is_open()) {
            log_file_.flush();
            log_file_.close();
        }
    }

private:
    struct Vec2 {
        double x;
        double y;
    };

    struct ObstacleState {
        Vec2 pos;
        Vec2 vel;
        double radius;
        int id = 0;
        std::vector<Vec2> polygon;
    };

    struct LaneSegment {
        Vec2 p0;
        Vec2 p1;
    };

    struct ArcHitResult {
        bool hit = false;
        double hit_time = 0.0;
        double hit_heading = 0.0;
        Vec2 hit_point{0.0, 0.0};
    };

    struct SideTangentResult {
        bool has_hit = false;
        double heading = 0.0;
        Vec2 hit_point{0.0, 0.0};
        int side_sign = 0;
        double hit_omega = 0.0; // 命中时的角速度
    };

    struct LogData {
        ros::Time stamp;
        double current_speed = 0.0;
        double target_linear_x = 0.0;
        double subgoal_distance = -1.0;


        double desired_heading = 0.0;
        double optimal_heading = 0.0;
        double motion_heading = 0.0;
        double robot_yaw = 0.0;
        double nearest_clear = -1.0;
        int avoidance_active = 0;
        int has_hit_point = 0;
        double hit_x = 0.0;
        double hit_y = 0.0;
        size_t obstacle_count = 0;
    };

    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;
    ros::Subscriber obs_sub_;
    ros::Subscriber subgoal_sub_;
    ros::Subscriber left_lane_sub_;
    ros::Subscriber right_lane_sub_;
    ros::Subscriber vo_state_sub_;
    ros::Publisher vo_state_pub_;
    ros::Publisher cmd_pub_;
    ros::Publisher marker_pub_;
    ros::Timer timer_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Params
    double control_period_sec_ = 0.05;
    double max_linear_vel_ = 3.5;
    double robot_radius_ = 0.25;
    double obstacle_radius_inflation_ = 0.10;
    double avoidance_range_time_sec_ = 2.0;
    bool enable_speed_smoothing_ = true;
    double speed_lpf_alpha_ = 0.25;
    double speed_rise_limit_mps2_ = 1.0;
    double speed_fall_limit_mps2_ = 2.5;
    int speed_arc_sample_count_ = 14;
    int speed_arc_sim_steps_ = 36;
    double speed_arc_safety_clearance_ = 0.10;
    double lane_obstacle_width_ = 0.20;
    int lane_downsample_step_ = 3;
    int arc_sim_steps_ = 80;
    double tangent_omega_step_ratio_ = 0.05;
    bool enable_runtime_debug_ = true;
    bool enable_debug_output_ = true;
    bool enable_rviz_visualization_ = true;
    bool enable_lidar_obstacle_markers_ = true;
    bool enable_lidar_polygon_markers_ = true;
    bool enable_lidar_velocity_markers_ = false;
    double lidar_obstacle_marker_min_diameter_ = 0.08;
    double lidar_polygon_marker_line_width_ = 0.05;
    double collision_slope_deg_ = 18.0;
    double collision_slope_ref_speed_ = 3.5;
    double goal_stop_distance_ = 0.45;
    bool enable_file_log_ = true;
    std::string log_dir_;
    bool apply_yaw_limit_on_output_ = false;
    double fallback_max_yaw_rate_ = 0.85;
    double viz_direction_vector_length_ = 1.5;
    std::string angular_limit_csv_path_;

    std::string odom_topic_;
    std::string obstacles_topic_;
    std::string subgoal_topic_;
    std::string left_lane_path_topic_;
    std::string right_lane_path_topic_;
    std::string speed_command_topic_;
    std::string optimal_direction_marker_topic_;

    // State
    Vec2 robot_pos_{0.0, 0.0};
    Vec2 robot_vel_{0.0, 0.0};
    double robot_yaw_ = 0.0;
    std::string odom_frame_id_ = "odom";
    Vec2 subgoal_pos_{0.0, 0.0};
    Vec2 last_subgoal_pos_{0.0, 0.0};
    bool has_subgoal_ = false;
    bool has_last_subgoal_ = false;
    bool has_smoothed_speed_cmd_ = false;
    double smoothed_speed_cmd_ = 0.0;
    int last_selected_side_sign_ = 0;
    ros::Time last_side_change_time_;
    bool has_last_hit_heading_ = false;
    double last_hit_heading_ = 0.0;
    // 状态机：定义放在头文件 vo_navigation/vo_state.h 中以供其他模块复用
    int vo_state_ = vo_navigation::STATE_NORMAL; // 默认正常
    double fallback_path_heading_ = 0.0;
    bool has_fallback_path_heading_ = false;
    // 当使用 /vo/state 话题控制状态时，如果在该超时时间内未收到新的状态消息则自动恢复到 STATE_NORMAL
    double vo_state_timeout_sec_ = 0.5;
    ros::Time last_state_msg_time_ = ros::Time::now();
    // Recovery planning (body-frame occupancy grid + A*)
    std::vector<Vec2> recovery_path_;
    size_t recovery_index_ = 0;
    double recovery_grid_res_ = 0.3; // meters
    double recovery_grid_radius_ = 12.0; // meters (half-size)
    double recovery_replan_interval_ = 1.0; // seconds
    ros::Time last_replan_time_ = ros::Time::now();
    ros::Time recovery_start_time_ = ros::Time(0);
    double recovery_v_ = 0.5; // m/s # 新机器人修改
    int stuck_counter_ = 0;
    int stuck_threshold_ = 200; // frames
    double v_stuck_threshold_ = 0.35; // m/s # 新机器人修改
    double recovery_reach_radius_ = 0.5; // m
    double recovery_timeout_sec_ = 30.0; // seconds

    std::ofstream log_file_;
    std::string log_file_path_;
    std::thread logger_thread_;
    std::atomic<bool> stop_logger_;
    bool logger_thread_started_ = false;
    std::queue<LogData> log_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::vector<ObstacleState> lidar_obstacles_;
    std::vector<ObstacleState> last_lidar_obstacles_;
    int next_obstacle_id_ = 1;
    std::vector<LaneSegment> left_lane_segments_;
    std::vector<LaneSegment> right_lane_segments_;

    std::vector<std::pair<double, double> > speed_to_yaw_rate_limits_;

    void initLogFile() {
        if (!enable_file_log_) {
            return;
        }

        std::string base_dir = log_dir_;
        if (base_dir.empty()) {
            const char* home = std::getenv("HOME");
            if (home == nullptr) {
                ROS_WARN("vo_node logging disabled: HOME is not set");
                return;
            }
            base_dir = std::string(home) + "/catkin_ws/logs/vo_node";
        }

        struct stat st;
        if (stat(base_dir.c_str(), &st) != 0) {
            const std::string cmd = "mkdir -p '" + base_dir + "'";
            if (std::system(cmd.c_str()) != 0) {
                ROS_WARN("vo_node logging disabled: failed to create log dir %s", base_dir.c_str());
                return;
            }
        }

        const std::time_t now = std::time(nullptr);
        const std::tm* tm_info = std::localtime(&now);
        if (tm_info == nullptr) {
            ROS_WARN("vo_node logging disabled: localtime failed");
            return;
        }

        std::ostringstream oss;
        oss << base_dir << "/vo_node_" << std::put_time(tm_info, "%Y%m%d_%H%M%S") << ".csv";
        log_file_path_ = oss.str();
        log_file_.open(log_file_path_.c_str(), std::ios::out);
        if (!log_file_.is_open()) {
            ROS_WARN("vo_node logging disabled: cannot open %s", log_file_path_.c_str());
            return;
        }

        log_file_ << "stamp,current_speed,target_linear_x,subgoal_distance,desired_heading,optimal_heading,motion_heading,robot_yaw,nearest_clear,avoidance_active,has_hit_point,hit_x,hit_y,obstacle_count\n";
        log_file_.flush();
        ROS_INFO("vo_node logging to %s", log_file_path_.c_str());
    }

    void writeCycleLog(double current_speed,
                       double target_linear_x,
                       double subgoal_distance,
                       double desired_heading,
                       double optimal_heading,
                       double motion_heading,
                       double nearest_clear,
                       bool avoidance_active,
                       bool has_selected_hit_point,
                       const Vec2& selected_hit_point,
                       size_t obstacle_count) {
        if (!enable_file_log_ || !log_file_.is_open()) {
            return;
        }

        LogData data;
        data.stamp = ros::Time::now();
        data.current_speed = current_speed;
        data.target_linear_x = target_linear_x;
        data.subgoal_distance = std::isfinite(subgoal_distance) ? subgoal_distance : -1.0;
        data.desired_heading = desired_heading;
        data.optimal_heading = optimal_heading;
        data.motion_heading = motion_heading;
        data.robot_yaw = robot_yaw_;
        data.nearest_clear = std::isfinite(nearest_clear) ? nearest_clear : -1.0;
        data.avoidance_active = avoidance_active ? 1 : 0;
        data.has_hit_point = has_selected_hit_point ? 1 : 0;
        data.hit_x = selected_hit_point.x;
        data.hit_y = selected_hit_point.y;
        data.obstacle_count = obstacle_count;

        std::lock_guard<std::mutex> lock(queue_mutex_);
        log_queue_.push(data);
        queue_cv_.notify_one();
    }

    void logWriterThread() {
        if (enable_debug_output_) {
            ROS_INFO("vo_node log writer thread started");
        }
        while (!stop_logger_) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100),
                               [this] { return !log_queue_.empty() || stop_logger_; });

            while (!log_queue_.empty()) {
                const LogData data = log_queue_.front();
                log_queue_.pop();
                lock.unlock();

                if (log_file_.is_open()) {
                    log_file_ << std::fixed << std::setprecision(6)
                              << data.stamp.toSec() << ","
                              << data.current_speed << ","
                              << data.target_linear_x << ","
                              << data.subgoal_distance << ","
                              << data.desired_heading << ","
                              << data.optimal_heading << ","
                              << data.motion_heading << ","
                              << data.robot_yaw << ","
                              << data.nearest_clear << ","
                              << data.avoidance_active << ","
                              << data.has_hit_point << ","
                              << data.hit_x << ","
                              << data.hit_y << ","
                              << data.obstacle_count << "\n";
                    log_file_.flush();
                }

                lock.lock();
            }
        }
        if (enable_debug_output_) {
            ROS_INFO("vo_node log writer thread stopped");
        }
    }

    static double normalizeAngle(double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    static double shortestAngularDistance(double from, double to) {
        return normalizeAngle(to - from);
    }

    static double dot(const Vec2& a, const Vec2& b) {
        return a.x * b.x + a.y * b.y;
    }

    static double norm(const Vec2& v) {
        return std::sqrt(v.x * v.x + v.y * v.y);
    }

    static Vec2 add(const Vec2& a, const Vec2& b) {
        return Vec2{a.x + b.x, a.y + b.y};
    }

    static Vec2 sub(const Vec2& a, const Vec2& b) {
        return Vec2{a.x - b.x, a.y - b.y};
    }

    static Vec2 mul(const Vec2& a, double k) {
        return Vec2{a.x * k, a.y * k};
    }

    static Vec2 unitFromHeading(double h) {
        return Vec2{std::cos(h), std::sin(h)};
    }

    void computeSlopeTrapezoidCorners(double heading,
                                      double lookahead_dist,
                                      double reference_speed,
                                      Vec2& p0,
                                      Vec2& p1,
                                      Vec2& p2,
                                      Vec2& p3) const {
        const Vec2 fwd = unitFromHeading(heading);
        const Vec2 left{-fwd.y, fwd.x};
        const double ref_v = std::max(0.1, collision_slope_ref_speed_);
        const double v_ratio = std::max(0.0, std::min(1.0, std::max(0.0, reference_speed) / ref_v));
        const double slope_deg = std::max(0.0, collision_slope_deg_) * v_ratio;
        const double slope_rad = slope_deg * M_PI / 180.0;
        const double slope_k = std::tan(slope_rad);

        const double near_half = std::max(0.0, robot_radius_);
        const double far_half = near_half + slope_k * std::max(0.0, lookahead_dist);
        const Vec2 far_center = add(robot_pos_, mul(fwd, std::max(0.0, lookahead_dist)));

        // p0->p1->p2->p3 构成朝前的梯形（近端右、近端左、远端左、远端右）
        p0 = add(robot_pos_, mul(left, -near_half));
        p1 = add(robot_pos_, mul(left, near_half));
        p2 = add(far_center, mul(left, far_half));
        p3 = add(far_center, mul(left, -far_half));
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        robot_pos_.x = msg->pose.pose.position.x;
        robot_pos_.y = msg->pose.pose.position.y;
        robot_vel_.x = msg->twist.twist.linear.x;
        robot_vel_.y = msg->twist.twist.linear.y;
        robot_yaw_ = tf2::getYaw(msg->pose.pose.orientation);
        if (!msg->header.frame_id.empty()) {
            odom_frame_id_ = msg->header.frame_id;
        }
    }

    Vec2 computeArcStart(double heading, int arc_side_sign) const {
        const Vec2 fwd = unitFromHeading(heading);
        const Vec2 left{-fwd.y, fwd.x};
        return add(robot_pos_, mul(left, -static_cast<double>(arc_side_sign) * robot_radius_));
    }

    void clearOptimalDirectionMarkers() {
        if (!enable_rviz_visualization_) {
            return;
        }

        visualization_msgs::Marker point_marker;
        point_marker.header.stamp = ros::Time::now();
        point_marker.header.frame_id = odom_frame_id_;
        point_marker.ns = "vo_optimal_dir";
        point_marker.id = 0;
        point_marker.action = visualization_msgs::Marker::DELETE;
        marker_pub_.publish(point_marker);

        visualization_msgs::Marker arrow_marker = point_marker;
        arrow_marker.id = 1;
        marker_pub_.publish(arrow_marker);
    }

    void publishOptimalDirectionMarkers(double optimal_heading,
                                        bool has_selected_hit,
                                        const Vec2& selected_hit_point) {
        if (!enable_rviz_visualization_) {
            return;
        }
        if (!has_selected_hit) {
            clearOptimalDirectionMarkers();
            return;
        }

        const Vec2 dir = unitFromHeading(optimal_heading);
        const Vec2 end = add(selected_hit_point, mul(dir, std::max(0.3, viz_direction_vector_length_)));

        visualization_msgs::Marker point_marker;
        point_marker.header.stamp = ros::Time::now();
        point_marker.header.frame_id = odom_frame_id_;
        point_marker.ns = "vo_optimal_dir";
        point_marker.id = 0;
        point_marker.type = visualization_msgs::Marker::SPHERE;
        point_marker.action = visualization_msgs::Marker::ADD;
        point_marker.pose.position.x = selected_hit_point.x;
        point_marker.pose.position.y = selected_hit_point.y;
        point_marker.pose.position.z = 0.05;
        point_marker.pose.orientation.w = 1.0;
        point_marker.scale.x = 0.16;
        point_marker.scale.y = 0.16;
        point_marker.scale.z = 0.16;
        point_marker.color.a = 1.0;
        point_marker.color.r = 1.0;
        point_marker.color.g = 0.4;
        point_marker.color.b = 0.1;
        marker_pub_.publish(point_marker);

        visualization_msgs::Marker arrow_marker;
        arrow_marker.header = point_marker.header;
        arrow_marker.ns = "vo_optimal_dir";
        arrow_marker.id = 1;
        arrow_marker.type = visualization_msgs::Marker::ARROW;
        arrow_marker.action = visualization_msgs::Marker::ADD;
        geometry_msgs::Point p0;
        p0.x = selected_hit_point.x;
        p0.y = selected_hit_point.y;
        p0.z = 0.08;
        geometry_msgs::Point p1;
        p1.x = end.x;
        p1.y = end.y;
        p1.z = 0.08;
        arrow_marker.points.push_back(p0);
        arrow_marker.points.push_back(p1);
        arrow_marker.scale.x = 0.06;
        arrow_marker.scale.y = 0.12;
        arrow_marker.scale.z = 0.12;
        arrow_marker.color.a = 1.0;
        arrow_marker.color.r = 0.1;
        arrow_marker.color.g = 0.9;
        arrow_marker.color.b = 0.2;
        marker_pub_.publish(arrow_marker);
    }

    std::vector<geometry_msgs::Point> buildArcPoints(const Vec2& arc_start,
                                                     double start_heading,
                                                     double speed,
                                                     double omega,
                                                     double horizon,
                                                     int steps) const {
        std::vector<geometry_msgs::Point> points;
        points.reserve(static_cast<size_t>(steps + 1));

        for (int i = 0; i <= steps; ++i) {
            const double t = horizon * static_cast<double>(i) / static_cast<double>(std::max(1, steps));
            Vec2 p;
            if (std::abs(omega) < 1e-6) {
                p.x = arc_start.x + speed * t * std::cos(start_heading);
                p.y = arc_start.y + speed * t * std::sin(start_heading);
            } else {
                const double h = start_heading + omega * t;
                p.x = arc_start.x + (speed / omega) * (std::sin(h) - std::sin(start_heading));
                p.y = arc_start.y - (speed / omega) * (std::cos(h) - std::cos(start_heading));
            }

            geometry_msgs::Point gp;
            gp.x = p.x;
            gp.y = p.y;
            gp.z = 0.03;
            points.push_back(gp);
        }
        return points;
    }

    std::vector<geometry_msgs::Point> buildCirclePoints(const Vec2& center,
                                                        double radius,
                                                        int segments) const {
        std::vector<geometry_msgs::Point> points;
        const int n = std::max(16, segments);
        points.reserve(static_cast<size_t>(n + 1));

        for (int i = 0; i <= n; ++i) {
            const double a = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n);
            geometry_msgs::Point p;
            p.x = center.x + radius * std::cos(a);
            p.y = center.y + radius * std::sin(a);
            p.z = 0.02;
            points.push_back(p);
        }
        return points;
    }

    void publishPlanningGeometryMarkers(double odom_heading,
                                        double plan_speed,
                                        double horizon,
                                        double omega_max) {
        if (!enable_rviz_visualization_) {
            return;
        }

        const int steps = std::max(20, arc_sim_steps_);
        const Vec2 left_start = computeArcStart(odom_heading, +1);
        const Vec2 right_start = computeArcStart(odom_heading, -1);

        visualization_msgs::Marker left_arc;
        left_arc.header.stamp = ros::Time::now();
        left_arc.header.frame_id = odom_frame_id_;
        left_arc.ns = "vo_planning_geom";
        left_arc.id = 10;
        left_arc.type = visualization_msgs::Marker::LINE_STRIP;
        left_arc.action = visualization_msgs::Marker::ADD;
        left_arc.pose.orientation.w = 1.0;
        left_arc.scale.x = 0.04;
        left_arc.color.a = 1.0;
        left_arc.color.r = 0.15;
        left_arc.color.g = 0.55;
        left_arc.color.b = 1.0;
        left_arc.points = buildArcPoints(left_start, odom_heading, plan_speed, +omega_max, horizon, steps);
        marker_pub_.publish(left_arc);

        visualization_msgs::Marker right_arc = left_arc;
        right_arc.id = 11;
        right_arc.color.r = 1.0;
        right_arc.color.g = 0.45;
        right_arc.color.b = 0.1;
        right_arc.points = buildArcPoints(right_start, odom_heading, plan_speed, -omega_max, horizon, steps);
        marker_pub_.publish(right_arc);

        visualization_msgs::Marker robot_circle = left_arc;
        robot_circle.id = 12;
        robot_circle.scale.x = 0.05;
        robot_circle.color.r = 1.0;
        robot_circle.color.g = 1.0;
        robot_circle.color.b = 0.0;
        robot_circle.points = buildCirclePoints(robot_pos_, robot_radius_, 64);
        marker_pub_.publish(robot_circle);

        // 额外绘制左右圆弧圆心，便于验证圆心是否位于 odom 航向法线方向上。
        visualization_msgs::Marker left_center;
        left_center.header = left_arc.header;
        left_center.ns = "vo_planning_geom";
        left_center.id = 13;
        left_center.type = visualization_msgs::Marker::SPHERE;
        left_center.action = visualization_msgs::Marker::ADD;
        left_center.pose.orientation.w = 1.0;
        left_center.scale.x = 0.10;
        left_center.scale.y = 0.10;
        left_center.scale.z = 0.10;
        left_center.color.a = 1.0;
        left_center.color.r = 0.15;
        left_center.color.g = 0.55;
        left_center.color.b = 1.0;

        visualization_msgs::Marker right_center = left_center;
        right_center.id = 14;
        right_center.color.r = 1.0;
        right_center.color.g = 0.45;
        right_center.color.b = 0.1;

        if (std::abs(omega_max) > 1e-6) {
            const double radius = plan_speed / std::abs(omega_max);
            const Vec2 fwd = unitFromHeading(odom_heading);
            const Vec2 left{-fwd.y, fwd.x};
            const Vec2 left_center_pos = add(left_start, mul(left, radius));
            const Vec2 right_center_pos = add(right_start, mul(left, -radius));

            left_center.pose.position.x = left_center_pos.x;
            left_center.pose.position.y = left_center_pos.y;
            left_center.pose.position.z = 0.04;
            marker_pub_.publish(left_center);

            right_center.pose.position.x = right_center_pos.x;
            right_center.pose.position.y = right_center_pos.y;
            right_center.pose.position.z = 0.04;
            marker_pub_.publish(right_center);
        }
    }

    void publishHeadingTrapezoidMarker(double heading,
                                       double lookahead_dist,
                                       double reference_speed,
                                       int marker_id,
                                       float r,
                                       float g,
                                       float b) {
        if (!enable_rviz_visualization_) {
            return;
        }

        Vec2 p0, p1, p2, p3;
        computeSlopeTrapezoidCorners(heading, lookahead_dist, reference_speed, p0, p1, p2, p3);

        visualization_msgs::Marker trap;
        trap.header.stamp = ros::Time::now();
        trap.header.frame_id = odom_frame_id_;
        trap.ns = "vo_planning_geom";
        trap.id = marker_id;
        trap.type = visualization_msgs::Marker::LINE_STRIP;
        trap.action = visualization_msgs::Marker::ADD;
        trap.pose.orientation.w = 1.0;
        trap.scale.x = 0.05;
        trap.color.a = 1.0;
        trap.color.r = r;
        trap.color.g = g;
        trap.color.b = b;

        geometry_msgs::Point g0; g0.x = p0.x; g0.y = p0.y; g0.z = 0.05;
        geometry_msgs::Point g1; g1.x = p1.x; g1.y = p1.y; g1.z = 0.05;
        geometry_msgs::Point g2; g2.x = p2.x; g2.y = p2.y; g2.z = 0.05;
        geometry_msgs::Point g3; g3.x = p3.x; g3.y = p3.y; g3.z = 0.05;
        trap.points.push_back(g0);
        trap.points.push_back(g1);
        trap.points.push_back(g2);
        trap.points.push_back(g3);
        trap.points.push_back(g0);
        marker_pub_.publish(trap);
    }

    void publishClusteredLidarObstacleMarkers(const std::vector<ObstacleState>& obstacles) {
        if (!enable_rviz_visualization_ || !enable_lidar_obstacle_markers_) {
            return;
        }

        const ros::Time now = ros::Time::now();
        const ros::Duration ttl(std::max(0.1, control_period_sec_ * 3.0));
        const double min_d = std::max(0.01, lidar_obstacle_marker_min_diameter_);

        for (size_t i = 0; i < obstacles.size(); ++i) {
            const ObstacleState& o = obstacles[i];
            const bool draw_polygon = enable_lidar_polygon_markers_ && o.polygon.size() >= 2;
            visualization_msgs::Marker base;
            base.header.stamp = now;
            base.header.frame_id = odom_frame_id_;
            base.id = static_cast<int>(i);
            base.action = visualization_msgs::Marker::ADD;
            base.pose.orientation.w = 1.0;
            base.lifetime = ttl;

            if (draw_polygon) {
                visualization_msgs::Marker poly = base;
                poly.ns = "vo_lidar_polygons";
                poly.type = visualization_msgs::Marker::LINE_STRIP;
                poly.scale.x = std::max(0.01, lidar_polygon_marker_line_width_);
                poly.color.a = 1.0;
                poly.color.r = 0.95;
                poly.color.g = 0.30;
                poly.color.b = 0.10;

                for (size_t k = 0; k < o.polygon.size(); ++k) {
                    geometry_msgs::Point p;
                    p.x = o.polygon[k].x;
                    p.y = o.polygon[k].y;
                    p.z = 0.10;
                    poly.points.push_back(p);
                }
                if (o.polygon.size() > 2) {
                    geometry_msgs::Point p0;
                    p0.x = o.polygon.front().x;
                    p0.y = o.polygon.front().y;
                    p0.z = 0.10;
                    poly.points.push_back(p0);
                }
                marker_pub_.publish(poly);

                // Filled polygon visualization (semi-transparent) using TRIANGLE_LIST
                if (o.polygon.size() > 2) {
                    visualization_msgs::Marker fill = base;
                    fill.ns = "vo_lidar_polygons_filled";
                    fill.type = visualization_msgs::Marker::TRIANGLE_LIST;
                    fill.action = visualization_msgs::Marker::ADD;
                    fill.scale.x = 1.0;
                    fill.scale.y = 1.0;
                    fill.scale.z = 1.0;
                    fill.color.a = 0.35;
                    fill.color.r = 0.95;
                    fill.color.g = 0.30;
                    fill.color.b = 0.10;

                    // center for triangle fan
                    geometry_msgs::Point center;
                    center.x = o.pos.x;
                    center.y = o.pos.y;
                    center.z = 0.10;

                    for (size_t k = 0; k < o.polygon.size(); ++k) {
                        const Vec2& pk = o.polygon[k];
                        const Vec2& pn = o.polygon[(k + 1) % o.polygon.size()];

                        geometry_msgs::Point pa = center;
                        geometry_msgs::Point pb;
                        pb.x = pk.x;
                        pb.y = pk.y;
                        pb.z = 0.10;
                        geometry_msgs::Point pc;
                        pc.x = pn.x;
                        pc.y = pn.y;
                        pc.z = 0.10;

                        fill.points.push_back(pa);
                        fill.points.push_back(pb);
                        fill.points.push_back(pc);
                    }
                    marker_pub_.publish(fill);
                }

                // Small centroid marker to keep obstacle center visible.
                visualization_msgs::Marker center = base;
                center.ns = "vo_lidar_clusters";
                center.type = visualization_msgs::Marker::SPHERE;
                center.pose.position.x = o.pos.x;
                center.pose.position.y = o.pos.y;
                center.pose.position.z = 0.10;
                center.scale.x = min_d;
                center.scale.y = min_d;
                center.scale.z = min_d;
                center.color.a = 0.85;
                center.color.r = 1.0;
                center.color.g = 0.95;
                center.color.b = 0.10;
                marker_pub_.publish(center);
            } else {
                visualization_msgs::Marker sphere = base;
                sphere.ns = "vo_lidar_clusters";
                sphere.type = visualization_msgs::Marker::SPHERE;
                sphere.pose.position.x = o.pos.x;
                sphere.pose.position.y = o.pos.y;
                sphere.pose.position.z = 0.10;
                const double d = std::max(min_d, 2.0 * std::max(0.0, o.radius));
                sphere.scale.x = d;
                sphere.scale.y = d;
                sphere.scale.z = d;
                sphere.color.a = 0.85;
                sphere.color.r = 0.95;
                sphere.color.g = 0.15;
                sphere.color.b = 0.15;
                marker_pub_.publish(sphere);

                if (enable_lidar_polygon_markers_) {
                    visualization_msgs::Marker poly_del = base;
                    poly_del.ns = "vo_lidar_polygons";
                    poly_del.action = visualization_msgs::Marker::DELETE;
                    marker_pub_.publish(poly_del);
                }
            }

            if (!enable_lidar_velocity_markers_) {
                continue;
            }

            visualization_msgs::Marker vel;
            vel.header = base.header;
            vel.ns = "vo_lidar_cluster_vel";
            vel.id = static_cast<int>(i);
            vel.type = visualization_msgs::Marker::ARROW;
            vel.action = visualization_msgs::Marker::ADD;
            vel.pose.orientation.w = 1.0;
            geometry_msgs::Point p0;
            p0.x = o.pos.x;
            p0.y = o.pos.y;
            p0.z = 0.20;
            geometry_msgs::Point p1;
            p1.x = o.pos.x + o.vel.x;
            p1.y = o.pos.y + o.vel.y;
            p1.z = 0.20;
            vel.points.push_back(p0);
            vel.points.push_back(p1);
            vel.scale.x = 0.03;
            vel.scale.y = 0.06;
            vel.scale.z = 0.06;
            vel.color.a = 0.9;
            vel.color.r = 0.15;
            vel.color.g = 0.95;
            vel.color.b = 0.15;
            vel.lifetime = ttl;
            marker_pub_.publish(vel);
        }
    }

    void subgoalCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        if (has_subgoal_) {
            last_subgoal_pos_ = subgoal_pos_;
            has_last_subgoal_ = true;
        }
        subgoal_pos_.x = msg->point.x;
        subgoal_pos_.y = msg->point.y;
        has_subgoal_ = true;
    }

    void voStateCallback(const std_msgs::Int32::ConstPtr& msg) {
        const int new_state = msg->data;
        last_state_msg_time_ = ros::Time::now();
        // External control received: treat as authoritative for now and
        // reset internal stuck detector so we don't immediately override.
        stuck_counter_ = 0;
        // 如果上层发布为 UNHEALTHY（1）并且之前为 NORMAL，则捕获回退方向并切换
        if (new_state == vo_navigation::STATE_ODOM_UNHEALTHY && vo_state_ == vo_navigation::STATE_NORMAL) {
            if (has_last_subgoal_) {
                double dx = subgoal_pos_.x - last_subgoal_pos_.x;
                double dy = subgoal_pos_.y - last_subgoal_pos_.y;
                fallback_path_heading_ = std::atan2(dy, dx);
                has_fallback_path_heading_ = true;
            }
            vo_state_ = vo_navigation::STATE_ODOM_UNHEALTHY;
            return;
        }

        // 如果上层显式发布 NORMAL（0），则恢复
        if (new_state == vo_navigation::STATE_NORMAL) {
            vo_state_ = vo_navigation::STATE_NORMAL;
            return;
        }

        // 对于其他值，直接设置（保留扩展可能性）
        vo_state_ = new_state;
    }

    void obstacleCallback(const vo_navigation::ObstacleArray::ConstPtr& msg) {
        // Transform incoming obstacles into `odom_frame_id_` (if needed) then store.
        const double match_dist = 0.6; // meters
        const double match_dist2 = match_dist * match_dist;
        std::vector<ObstacleState> prev = last_lidar_obstacles_;
        std::vector<bool> prev_used(prev.size(), false);

        const std::string source_frame = msg->header.frame_id;
        const ros::Time stamp = msg->header.stamp;
        geometry_msgs::TransformStamped tf_stamped;
        bool need_tf = !source_frame.empty() && source_frame != odom_frame_id_;
        if (need_tf) {
            try {
                tf_stamped = tf_buffer_.lookupTransform(odom_frame_id_, source_frame, stamp, ros::Duration(0.05));
            } catch (const tf2::TransformException& ex) {
                try {
                    tf_stamped = tf_buffer_.lookupTransform(odom_frame_id_, source_frame, ros::Time(0), ros::Duration(0.05));
                } catch (const tf2::TransformException& ex2) {
                    ROS_WARN_THROTTLE(1.0, "TF lookup failed (%s -> %s): %s", source_frame.c_str(), odom_frame_id_.c_str(), ex2.what());
                    need_tf = false;
                }
            }
        }

        lidar_obstacles_.clear();
        lidar_obstacles_.reserve(msg->obstacles.size());
        for (size_t i = 0; i < msg->obstacles.size(); ++i) {
            const auto& o = msg->obstacles[i];
            ObstacleState obs;
            obs.radius = std::max(0.0, static_cast<double>(o.radius));
            obs.polygon.clear();

            double sx = 0.0;
            double sy = 0.0;
            // transform polygon points (if any)
            for (size_t k = 0; k < o.polygon.points.size(); ++k) {
                double px = static_cast<double>(o.polygon.points[k].x);
                double py = static_cast<double>(o.polygon.points[k].y);
                if (need_tf) {
                    geometry_msgs::PointStamped in_pt, out_pt;
                    in_pt.header.stamp = stamp;
                    in_pt.header.frame_id = source_frame;
                    in_pt.point.x = px;
                    in_pt.point.y = py;
                    in_pt.point.z = static_cast<double>(o.polygon.points[k].z);
                    tf2::doTransform(in_pt, out_pt, tf_stamped);
                    px = out_pt.point.x;
                    py = out_pt.point.y;
                }
                obs.polygon.push_back(Vec2{px, py});
                sx += px;
                sy += py;
            }

            if (!obs.polygon.empty()) {
                obs.pos.x = sx / static_cast<double>(obs.polygon.size());
                obs.pos.y = sy / static_cast<double>(obs.polygon.size());

                double poly_radius = 0.0;
                for (size_t k = 0; k < obs.polygon.size(); ++k) {
                    poly_radius = std::max(poly_radius, norm(sub(obs.polygon[k], obs.pos)));
                }
                obs.radius = std::max(obs.radius, poly_radius);
            } else {
                // no polygon, transform center position if needed
                double cx = static_cast<double>(o.position.x);
                double cy = static_cast<double>(o.position.y);
                if (need_tf) {
                    geometry_msgs::PointStamped in_pt, out_pt;
                    in_pt.header.stamp = stamp;
                    in_pt.header.frame_id = source_frame;
                    in_pt.point.x = cx;
                    in_pt.point.y = cy;
                    in_pt.point.z = static_cast<double>(o.position.z);
                    tf2::doTransform(in_pt, out_pt, tf_stamped);
                    cx = out_pt.point.x;
                    cy = out_pt.point.y;
                }
                obs.pos = Vec2{cx, cy};
            }

            // transform velocity vector (rotate only) if needed
            double vx = static_cast<double>(o.velocity.x);
            double vy = static_cast<double>(o.velocity.y);
            if (need_tf) {
                tf2::Quaternion q;
                q.setX(tf_stamped.transform.rotation.x);
                q.setY(tf_stamped.transform.rotation.y);
                q.setZ(tf_stamped.transform.rotation.z);
                q.setW(tf_stamped.transform.rotation.w);
                tf2::Vector3 vin(vx, vy, 0.0);
                tf2::Vector3 vout = tf2::quatRotate(q, vin);
                vx = vout.x();
                vy = vout.y();
            }
            obs.vel = Vec2{vx, vy};

            // match to previous obstacles by nearest distance
            int match_idx = -1;
            double best_d2 = std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < prev.size(); ++j) {
                if (prev_used[j]) continue;
                const double dx = obs.pos.x - prev[j].pos.x;
                const double dy = obs.pos.y - prev[j].pos.y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < best_d2 && d2 <= match_dist2) {
                    best_d2 = d2;
                    match_idx = static_cast<int>(j);
                }
            }
            if (match_idx >= 0) {
                obs.id = prev[match_idx].id;
                prev_used[match_idx] = true;
            } else {
                obs.id = next_obstacle_id_++;
            }

            lidar_obstacles_.push_back(obs);
        }

        // save for next-frame association
        last_lidar_obstacles_ = lidar_obstacles_;
    }

    void buildLaneSegmentsFromPath(const nav_msgs::Path::ConstPtr& msg,
                                   std::vector<LaneSegment>& out_segments) const {
        out_segments.clear();
        if (msg->poses.size() < 2) {
            return;
        }

        const size_t step = static_cast<size_t>(std::max(1, lane_downsample_step_));
        std::vector<Vec2> sampled;
        sampled.reserve(msg->poses.size() / step + 2);
        for (size_t i = 0; i < msg->poses.size(); i += step) {
            sampled.push_back(Vec2{msg->poses[i].pose.position.x, msg->poses[i].pose.position.y});
        }

        const auto& last = msg->poses.back().pose.position;
        if (sampled.empty() ||
            std::abs(sampled.back().x - last.x) > 1e-6 ||
            std::abs(sampled.back().y - last.y) > 1e-6) {
            sampled.push_back(Vec2{last.x, last.y});
        }

        if (sampled.size() < 2) {
            return;
        }

        out_segments.reserve(sampled.size() - 1);
        for (size_t i = 0; i + 1 < sampled.size(); ++i) {
            const Vec2 d = sub(sampled[i + 1], sampled[i]);
            if (norm(d) < 1e-6) {
                continue;
            }
            out_segments.push_back(LaneSegment{sampled[i], sampled[i + 1]});
        }
    }

    void leftLaneCallback(const nav_msgs::Path::ConstPtr& msg) {
        buildLaneSegmentsFromPath(msg, left_lane_segments_);
    }

    void rightLaneCallback(const nav_msgs::Path::ConstPtr& msg) {
        buildLaneSegmentsFromPath(msg, right_lane_segments_);
    }

    std::string resolveCsvPath(const std::string& csv_path) const {
        if (csv_path.empty() || csv_path[0] == '/') {
            return csv_path;
        }
        const std::string pkg_path = ros::package::getPath("vo_navigation");
        if (pkg_path.empty()) {
            return csv_path;
        }
        return pkg_path + "/" + csv_path;
    }

    void loadAngularLimitCsv(const std::string& csv_path) {
        speed_to_yaw_rate_limits_.clear();
        std::ifstream ifs(csv_path.c_str());
        if (!ifs.is_open()) {
            ROS_WARN("Cannot open angular limit csv: %s, fallback=%.3f", csv_path.c_str(), fallback_max_yaw_rate_);
            return;
        }

        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            if (line.find("linear_speed") != std::string::npos) {
                continue;
            }
            std::stringstream ss(line);
            std::string v_str;
            std::string w_str;
            if (!std::getline(ss, v_str, ',')) {
                continue;
            }
            if (!std::getline(ss, w_str)) {
                continue;
            }
            try {
                const double v = std::stod(v_str);
                const double w = std::stod(w_str);
                if (v >= 0.0 && w > 0.0) {
                    speed_to_yaw_rate_limits_.push_back(std::make_pair(v, w));
                }
            } catch (...) {
            }
        }

        std::sort(speed_to_yaw_rate_limits_.begin(), speed_to_yaw_rate_limits_.end(),
                  [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                      return a.first < b.first;
                  });

        if (speed_to_yaw_rate_limits_.empty()) {
            ROS_WARN("No valid rows in angular limit csv: %s, fallback=%.3f", csv_path.c_str(), fallback_max_yaw_rate_);
            return;
        }

        ROS_INFO("Loaded %zu angular limit rows from %s", speed_to_yaw_rate_limits_.size(), csv_path.c_str());
    }

    double lookupMaxYawRate(double linear_speed) const {
        if (speed_to_yaw_rate_limits_.empty()) {
            return std::max(0.01, fallback_max_yaw_rate_);
        }

        const double v = std::max(0.0, linear_speed);
        if (v < speed_to_yaw_rate_limits_.front().first) {
            return 1.0;
        }
        if (v > speed_to_yaw_rate_limits_.back().first) {
            return speed_to_yaw_rate_limits_.back().second;
        }

        for (size_t i = 1; i < speed_to_yaw_rate_limits_.size(); ++i) {
            const auto& p0 = speed_to_yaw_rate_limits_[i - 1];
            const auto& p1 = speed_to_yaw_rate_limits_[i];
            if (v <= p1.first) {
                const double span = std::max(1e-6, p1.first - p0.first);
                const double r = (v - p0.first) / span;
                return p0.second + r * (p1.second - p0.second);
            }
        }
        return speed_to_yaw_rate_limits_.back().second;
    }

    Vec2 computeArcEndPoint(double start_heading,
                            double speed,
                            double omega,
                            double horizon,
                            int arc_side_sign) const {
        const Vec2 arc_start = computeArcStart(start_heading, arc_side_sign);
        if (std::abs(omega) < 1e-6) {
            return Vec2{arc_start.x + speed * horizon * std::cos(start_heading),
                        arc_start.y + speed * horizon * std::sin(start_heading)};
        }

        const double h = start_heading + omega * horizon;
        return Vec2{arc_start.x + (speed / omega) * (std::sin(h) - std::sin(start_heading)),
                    arc_start.y - (speed / omega) * (std::cos(h) - std::cos(start_heading))};
    }

    static double cross2D(const Vec2& a, const Vec2& b, const Vec2& c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }

    static double distancePointToSegment(const Vec2& p, const Vec2& a, const Vec2& b) {
        const Vec2 ab = sub(b, a);
        const double ab2 = dot(ab, ab);
        if (ab2 < 1e-9) {
            return norm(sub(p, a));
        }
        const double t = std::max(0.0, std::min(1.0, dot(sub(p, a), ab) / ab2));
        const Vec2 proj = add(a, mul(ab, t));
        return norm(sub(p, proj));
    }

    static bool pointInPolygon(const Vec2& p, const std::vector<Vec2>& poly) {
        if (poly.size() < 3) {
            return false;
        }

        bool inside = false;
        for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
            const Vec2& pi = poly[i];
            const Vec2& pj = poly[j];
            const bool cond = ((pi.y > p.y) != (pj.y > p.y));
            if (!cond) {
                continue;
            }
            const double denom = (pj.y - pi.y);
            if (std::abs(denom) < 1e-9) {
                continue;
            }
            const double x_intersect = (pj.x - pi.x) * (p.y - pi.y) / denom + pi.x;
            if (p.x < x_intersect) {
                inside = !inside;
            }
        }
        return inside;
    }

    double obstacleClearanceAtPoint(const ObstacleState& obs, const Vec2& p) const {
        const double expand = robot_radius_ + obstacle_radius_inflation_;
        if (obs.polygon.empty()) {
            return norm(sub(p, obs.pos)) - (expand + obs.radius);
        }

        if (obs.polygon.size() == 1) {
            return norm(sub(p, obs.polygon.front())) - expand;
        }

        double min_dist = std::numeric_limits<double>::infinity();
        if (obs.polygon.size() == 2) {
            min_dist = distancePointToSegment(p, obs.polygon[0], obs.polygon[1]);
        } else {
            for (size_t i = 0; i < obs.polygon.size(); ++i) {
                const Vec2& a = obs.polygon[i];
                const Vec2& b = obs.polygon[(i + 1) % obs.polygon.size()];
                min_dist = std::min(min_dist, distancePointToSegment(p, a, b));
            }
            if (pointInPolygon(p, obs.polygon)) {
                return -(min_dist + expand);
            }
        }
        return min_dist - expand;
    }

    bool pointInTrapezoid(const Vec2& p,
                          const Vec2& p0,
                          const Vec2& p1,
                          const Vec2& p2,
                          const Vec2& p3) const {
        const double c0 = cross2D(p0, p1, p);
        const double c1 = cross2D(p1, p2, p);
        const double c2 = cross2D(p2, p3, p);
        const double c3 = cross2D(p3, p0, p);
        const bool non_neg = (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0 && c3 >= 0.0);
        const bool non_pos = (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0);
        return non_neg || non_pos;
    }

    bool pointInOrNearTrapezoid(const Vec2& p,
                                const Vec2& p0,
                                const Vec2& p1,
                                const Vec2& p2,
                                const Vec2& p3,
                                double expand) const {
        if (pointInTrapezoid(p, p0, p1, p2, p3)) {
            return true;
        }

        const double d0 = distancePointToSegment(p, p0, p1);
        const double d1 = distancePointToSegment(p, p1, p2);
        const double d2 = distancePointToSegment(p, p2, p3);
        const double d3 = distancePointToSegment(p, p3, p0);
        const double d = std::min(std::min(d0, d1), std::min(d2, d3));
        return d <= expand;
    }

    // ----------------- Recovery planner (body-frame A*) -----------------
    struct GridIndex { int x; int y; };

    inline void worldToBody(const Vec2& w, Vec2& b) const {
        // translate then rotate by -robot_yaw_
        const double dx = w.x - robot_pos_.x;
        const double dy = w.y - robot_pos_.y;
        const double c = std::cos(-robot_yaw_);
        const double s = std::sin(-robot_yaw_);
        b.x = c * dx - s * dy;
        b.y = s * dx + c * dy;
    }

    inline void bodyToWorld(const Vec2& b, Vec2& w) const {
        const double c = std::cos(robot_yaw_);
        const double s = std::sin(robot_yaw_);
        w.x = robot_pos_.x + c * b.x - s * b.y;
        w.y = robot_pos_.y + s * b.x + c * b.y;
    }

    bool isCellOccupied(const std::vector<std::vector<uint8_t>>& grid, int gx, int gy) const {
        if (gx < 0 || gy < 0) return true;
        if (gx >= static_cast<int>(grid.size()) || gy >= static_cast<int>(grid[0].size())) return true;
        return grid[gx][gy] != 0;
    }

    std::vector<Vec2> planRecoveryPath(const Vec2& goal_world) {
        // Build occupancy in body frame centered at robot (robot at 0,0)
        const double res = recovery_grid_res_;
        const double R = recovery_grid_radius_;
        const int half = static_cast<int>(std::ceil(R / res));
        const int size = half * 2 + 1;

        std::vector<std::vector<uint8_t>> grid(size, std::vector<uint8_t>(size, 0));

        // helper to convert body coord to grid index
        auto bodyToIdx = [&](const Vec2& p)->GridIndex{
            int ix = static_cast<int>(std::round(p.x / res)) + half;
            int iy = static_cast<int>(std::round(p.y / res)) + half;
            return GridIndex{ix, iy};
        };

        // mark lidar obstacles
        for (const auto& o : lidar_obstacles_) {
            // transform obs pos to body
            Vec2 pos_b;
            worldToBody(o.pos, pos_b);
            // radius expansion (robot + inflation)
            const double r = std::max(0.0, o.radius) + robot_radius_ + obstacle_radius_inflation_;
            const int rad_cells = static_cast<int>(std::ceil(r / res));
            const GridIndex c = bodyToIdx(pos_b);
            for (int dx = -rad_cells; dx <= rad_cells; ++dx) {
                for (int dy = -rad_cells; dy <= rad_cells; ++dy) {
                    int gx = c.x + dx;
                    int gy = c.y + dy;
                    if (gx < 0 || gy < 0 || gx >= size || gy >= size) continue;
                    const double cx = (gx - half) * res;
                    const double cy = (gy - half) * res;
                    const double dist = std::hypot(cx - pos_b.x, cy - pos_b.y);
                    if (dist <= r) grid[gx][gy] = 255;
                }
            }
        }

        // mark lane segments as occupied
        auto markSegment = [&](const LaneSegment& s) {
                // sample along segment
                const Vec2 a = s.p0;
                const Vec2 b = s.p1;
                const double len = norm(sub(b, a));
                const int samples = std::max(1, static_cast<int>(std::ceil(len / (res * 0.5))));
                for (int i = 0; i <= samples; ++i) {
                    const double t = static_cast<double>(i) / static_cast<double>(samples);
                    Vec2 wp{a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)};
                    Vec2 pb;
                    worldToBody(wp, pb);
                    GridIndex gi = bodyToIdx(pb);
                    if (gi.x >= 0 && gi.y >= 0 && gi.x < size && gi.y < size) {
                        grid[gi.x][gi.y] = 255;
                    }
                }
            };
        for (const auto& seg : left_lane_segments_) markSegment(seg);
        for (const auto& seg : right_lane_segments_) markSegment(seg);

        // A* on grid: start at robot (half,half), goal = goal in body frame
        Vec2 goal_b;
        worldToBody(goal_world, goal_b);
        GridIndex gidx = bodyToIdx(goal_b);
        GridIndex start{half, half};
        if (gidx.x < 0 || gidx.y < 0 || gidx.x >= size || gidx.y >= size) {
            return {}; // goal out of grid
        }
        if (isCellOccupied(grid, gidx.x, gidx.y)) {
            // try to find nearest free cell around goal
            bool found = false;
            for (int r = 1; r <= std::max(size, size) && !found; ++r) {
                for (int dx = -r; dx <= r && !found; ++dx) {
                    for (int dy = -r; dy <= r && !found; ++dy) {
                        int nx = gidx.x + dx;
                        int ny = gidx.y + dy;
                        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
                        if (!isCellOccupied(grid, nx, ny)) { gidx.x = nx; gidx.y = ny; found = true; }
                    }
                }
            }
            if (!found) return {};
        }

        struct Node { int x,y; double g,h; int px,py; };
        std::vector<std::vector<bool>> closed(size, std::vector<bool>(size,false));
        std::vector<std::vector<double>> gscore(size, std::vector<double>(size, 1e9));
        auto hfunc = [&](int x,int y)->double {
            double dx = (x - gidx.x) * res;
            double dy = (y - gidx.y) * res;
            return std::hypot(dx, dy);
        };

        using PQItem = std::tuple<double,int,int,int,int>; // f,x,y,px,py
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        gscore[start.x][start.y] = 0.0;
        pq.emplace(hfunc(start.x,start.y), start.x, start.y, -1, -1);
        std::vector<std::vector<int>> parentx(size, std::vector<int>(size, -1));
        std::vector<std::vector<int>> parity(size, std::vector<int>(size, -1));

        const int dxs[8] = {1,-1,0,0,1,1,-1,-1};
        const int dys[8] = {0,0,1,-1,1,-1,1,-1};

        bool path_found = false;
        while (!pq.empty()) {
            PQItem itm = pq.top(); pq.pop();
            double f = std::get<0>(itm);
            int x = std::get<1>(itm);
            int y = std::get<2>(itm);
            int px = std::get<3>(itm);
            int py = std::get<4>(itm);
            if (closed[x][y]) continue;
            closed[x][y] = true;
            parentx[x][y] = px;
            parity[x][y] = py;
            if (x == gidx.x && y == gidx.y) { path_found = true; break; }
            for (int k=0;k<8;++k) {
                int ddx = dxs[k];
                int ddy = dys[k];
                // Enforce forward-only (linear velocity >= 0) differential-drive approximation
                // In body frame robot forward == +x, so disallow moves with negative ddx (backwards).
                if (ddx < 0) continue;
                int nx = x + ddx;
                int ny = y + ddy;
                if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
                if (isCellOccupied(grid,nx,ny)) continue;
                double cost = ((k<4)?1.0:1.41421356) * res;
                double ng = gscore[x][y] + cost;
                if (ng + 1e-9 < gscore[nx][ny]) {
                    gscore[nx][ny] = ng;
                    double hf = ng + hfunc(nx,ny);
                    pq.emplace(hf, nx, ny, x, y);
                }
            }
        }

        if (!path_found) return {};

        // reconstruct path
        std::vector<Vec2> path_body;
        int cx = gidx.x, cy = gidx.y;
        while (!(cx == start.x && cy == start.y)) {
            double wx = (cx - half) * res;
            double wy = (cy - half) * res;
            path_body.push_back(Vec2{wx, wy});
            int px = parentx[cx][cy];
            int py = parity[cx][cy];
            if (px < 0 || py < 0) break;
            cx = px; cy = py;
        }
        std::reverse(path_body.begin(), path_body.end());

        std::vector<Vec2> path_world;
        for (const auto& pb : path_body) {
            Vec2 w; bodyToWorld(pb, w); path_world.push_back(w);
        }
        return path_world;
    }

    bool findNearestObstacleInForwardSlopeZone(const std::vector<ObstacleState>& obstacles,
                                               double heading,
                                               double lookahead_dist,
                                               double reference_speed,
                                               ObstacleState& nearest,
                                               double& nearest_clear_dist,
                                               bool include_lane_segments = true,
                                               bool use_obstacle_inflation = true) const {
        const Vec2 fwd = unitFromHeading(heading);
        const Vec2 left{-fwd.y, fwd.x};
        const double ref_v = std::max(0.1, collision_slope_ref_speed_);
        const double v_ratio = std::max(0.0, std::min(1.0, std::max(0.0, reference_speed) / ref_v));
        const double slope_deg = std::max(0.0, collision_slope_deg_) * v_ratio;
        const double slope_rad = slope_deg * M_PI / 180.0;
        const double slope_k = std::tan(slope_rad);

        Vec2 p0, p1, p2, p3;
        computeSlopeTrapezoidCorners(heading, lookahead_dist, reference_speed, p0, p1, p2, p3);

        bool found = false;
        nearest_clear_dist = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < obstacles.size(); ++i) {
            const ObstacleState& o = obstacles[i];
            const Vec2 rel = sub(o.pos, robot_pos_);
            const double along = dot(rel, fwd);
            if (along < 0.0) {
                continue;
            }

            const double inflation = use_obstacle_inflation ? obstacle_radius_inflation_ : 0.0;
            const double expand = robot_radius_ + o.radius + inflation;
            if (along > lookahead_dist + expand) {
                continue;
            }

            const double lat = std::abs(dot(rel, left));
            const double bound = robot_radius_ + slope_k * std::max(0.0, along) + expand;
            if (lat > bound) {
                continue;
            }
            if (!pointInOrNearTrapezoid(o.pos, p0, p1, p2, p3, expand)) {
                continue;
            }

            const double clear = along - expand;
            if (clear < nearest_clear_dist) {
                nearest_clear_dist = clear;
                nearest = o;
                found = true;
            }
        }

        if (include_lane_segments) {
            const double lane_half_width = std::max(0.0, lane_obstacle_width_ * 0.5);
            auto processLaneSegments = [&](const std::vector<LaneSegment>& segs) {
                for (size_t i = 0; i < segs.size(); ++i) {
                    const Vec2 a = segs[i].p0;
                    const Vec2 b = segs[i].p1;
                    const double len = norm(sub(b, a));
                    const int samples = std::max(2, static_cast<int>(std::ceil(len / 0.2)));
                    // Keep lane data uninflated: only robot radius + lane intrinsic width.
                    const double expand = robot_radius_ + lane_half_width;

                    for (int k = 0; k <= samples; ++k) {
                        const double t = static_cast<double>(k) / static_cast<double>(samples);
                        const Vec2 p = add(a, mul(sub(b, a), t));
                        const Vec2 rel = sub(p, robot_pos_);
                        const double along = dot(rel, fwd);
                        if (along < 0.0 || along > lookahead_dist + expand) {
                            continue;
                        }

                        const double lat = std::abs(dot(rel, left));
                        const double bound = robot_radius_ + slope_k * std::max(0.0, along) + expand;
                        if (lat > bound) {
                            continue;
                        }
                        if (!pointInOrNearTrapezoid(p, p0, p1, p2, p3, expand)) {
                            continue;
                        }

                        const double clear = along - expand;
                        if (clear < nearest_clear_dist) {
                            nearest_clear_dist = clear;
                            nearest.pos = p;
                            nearest.vel = Vec2{0.0, 0.0};
                            nearest.radius = lane_half_width;
                            nearest.polygon.clear();
                            found = true;
                        }
                    }
                }
            };

            processLaneSegments(left_lane_segments_);
            processLaneSegments(right_lane_segments_);
        }
        return found;
    }

    bool findNearestObstacleInForwardRectZone(const std::vector<ObstacleState>& obstacles,
                                              double heading,
                                              double lookahead_dist,
                                              ObstacleState& nearest,
                                              double& nearest_clear_dist,
                                              bool include_lane_segments = true,
                                              bool use_obstacle_inflation = true) const {
        const Vec2 fwd = unitFromHeading(heading);
        const Vec2 left{-fwd.y, fwd.x};

        bool found = false;
        nearest_clear_dist = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < obstacles.size(); ++i) {
            const ObstacleState& o = obstacles[i];
            const Vec2 rel = sub(o.pos, robot_pos_);
            const double along = dot(rel, fwd);
            if (along < 0.0) {
                continue;
            }

            const double inflation = use_obstacle_inflation ? obstacle_radius_inflation_ : 0.0;
            const double expand = robot_radius_ + o.radius + inflation;
            if (along > lookahead_dist + expand) {
                continue;
            }

            const double lat = std::abs(dot(rel, left));
            const double bound = robot_radius_ + expand;
            if (lat > bound) {
                continue;
            }

            const double clear = along - expand;
            if (clear < nearest_clear_dist) {
                nearest_clear_dist = clear;
                nearest = o;
                found = true;
            }
        }

        if (include_lane_segments) {
            const double lane_half_width = std::max(0.0, lane_obstacle_width_ * 0.5);
            auto processLaneSegments = [&](const std::vector<LaneSegment>& segs) {
                for (size_t i = 0; i < segs.size(); ++i) {
                    const Vec2 a = segs[i].p0;
                    const Vec2 b = segs[i].p1;
                    const double len = norm(sub(b, a));
                    const int samples = std::max(2, static_cast<int>(std::ceil(len / 0.2)));
                    const double expand = robot_radius_ + lane_half_width;

                    for (int k = 0; k <= samples; ++k) {
                        const double t = static_cast<double>(k) / static_cast<double>(samples);
                        const Vec2 p = add(a, mul(sub(b, a), t));
                        const Vec2 rel = sub(p, robot_pos_);
                        const double along = dot(rel, fwd);
                        if (along < 0.0 || along > lookahead_dist + expand) {
                            continue;
                        }

                        const double lat = std::abs(dot(rel, left));
                        const double bound = robot_radius_ + expand;
                        if (lat > bound) {
                            continue;
                        }

                        const double clear = along - expand;
                        if (clear < nearest_clear_dist) {
                            nearest_clear_dist = clear;
                            nearest.pos = p;
                            nearest.vel = Vec2{0.0, 0.0};
                            nearest.radius = lane_half_width;
                            nearest.polygon.clear();
                            found = true;
                        }
                    }
                }
            };

            processLaneSegments(left_lane_segments_);
            processLaneSegments(right_lane_segments_);
        }

        return found;
    }

    ArcHitResult simulateArcHit(const ObstacleState& obs,
                                double start_heading,
                                double speed,
                                double omega,
                                double horizon,
                                int arc_side_sign) const {
        ArcHitResult result;
        const int steps = std::max(20, arc_sim_steps_);
        const double dt = horizon / static_cast<double>(steps);

        // 按要求设置圆弧起点：
        // 左弧(arc_side_sign=+1)从“右交点”起步；右弧(arc_side_sign=-1)从“左交点”起步。
        const Vec2 fwd = unitFromHeading(start_heading);
        const Vec2 left{-fwd.y, fwd.x};
        const Vec2 arc_start = add(robot_pos_, mul(left, -static_cast<double>(arc_side_sign) * robot_radius_));

        for (int i = 1; i <= steps; ++i) {
            const double t = dt * static_cast<double>(i);
            Vec2 p;
            if (std::abs(omega) < 1e-6) {
                p.x = arc_start.x + speed * t * std::cos(start_heading);
                p.y = arc_start.y + speed * t * std::sin(start_heading);
            } else {
                const double h = start_heading + omega * t;
                p.x = arc_start.x + (speed / omega) * (std::sin(h) - std::sin(start_heading));
                p.y = arc_start.y - (speed / omega) * (std::cos(h) - std::cos(start_heading));
            }

            if (obstacleClearanceAtPoint(obs, p) <= 0.0) {
                result.hit = true;
                result.hit_time = t;
                result.hit_heading = normalizeAngle(start_heading + omega * t);
                result.hit_point = p;
                return result;
            }
        }
        return result;
    }

    SideTangentResult findHitTangentCandidate(const ObstacleState& obs,
                                               double start_heading,
                                               double speed,
                                               double horizon,
                                               double omega_max,
                                               int sign,
                                               double fallback_heading) const {
        SideTangentResult out;
        out.heading = normalizeAngle(fallback_heading);
        out.side_sign = sign;

        const double ratio = std::max(0.001, tangent_omega_step_ratio_);
        const double step = std::max(1e-4, std::abs(omega_max) * ratio);
        for (double om = omega_max; om >= 0.0; om -= step) {
            const double omega = static_cast<double>(sign) * om;
            ArcHitResult hit = simulateArcHit(obs, start_heading, speed, omega, horizon, sign);
            if (hit.hit) {
                out.has_hit = true;
                out.hit_point = hit.hit_point;
                // 采用“命中圆弧在命中点的切线方向”：即命中时刻的圆弧航向。
                out.heading = normalizeAngle(hit.hit_heading);
                out.hit_omega = omega;
                return out;
            }
        }
        return out;
    }

    double chooseOptimalHeading(const std::vector<ObstacleState>& obstacles,
        double desired_heading,
                                double arc_heading,
        double current_speed,
                                double target_speed,
                                int* chosen_arc_side_sign,
                                bool* used_avoidance,
                                Vec2* selected_hit_point,
                                bool* has_selected_hit_point) {
        if (chosen_arc_side_sign) {
            *chosen_arc_side_sign = 0;
        }
        if (used_avoidance) {
            *used_avoidance = false;
        }
        if (has_selected_hit_point) {
            *has_selected_hit_point = false;
        }
        if (selected_hit_point) {
            *selected_hit_point = Vec2{0.0, 0.0};
        }

        const double plan_speed = std::max(0.3, std::max(current_speed, target_speed));
        const double horizon = std::max(0.5, avoidance_range_time_sec_);
        const double omega_max = lookupMaxYawRate(plan_speed);
        const double lookahead = plan_speed * horizon;

        ObstacleState nearest_obs;
        double nearest_clear = std::numeric_limits<double>::infinity();
        const bool has_front_obs = findNearestObstacleInForwardSlopeZone(obstacles,
                                                                          arc_heading,
                                                                          lookahead,
                                                                          plan_speed,
                                                                          nearest_obs,
                                                                          nearest_clear);
        if (used_avoidance) {
            *used_avoidance = true;
        }

        const SideTangentResult left = findHitTangentCandidate(nearest_obs,
                                                                arc_heading,
                                                                plan_speed,
                                                                horizon,
                                                                omega_max,
                                                                +1,
                                                                desired_heading);
        const SideTangentResult right = findHitTangentCandidate(nearest_obs,
                                                                 arc_heading,
                                                                 plan_speed,
                                                                 horizon,
                                                                 omega_max,
                                                                 -1,
                                                                 desired_heading);

        if (!left.has_hit && !right.has_hit) {
            ObstacleState goal_nearest;
            double goal_clear = std::numeric_limits<double>::infinity();
            // Fallback step: only use lidar obstacles for goal-direction blocking check.
            const bool goal_blocked = findNearestObstacleInForwardSlopeZone(obstacles,
                                                                             desired_heading,
                                                                             lookahead,
                                                                             plan_speed,
                                                                             goal_nearest,
                                                                             goal_clear,
                                                                             false);
            double angle_diff = std::abs(shortestAngularDistance(arc_heading, desired_heading));
            if (goal_blocked && angle_diff < M_PI_2) { // 夹角小于90度
                if (used_avoidance) {
                    *used_avoidance = true;
                }
                if (has_last_subgoal_) {
                    double dx = subgoal_pos_.x - last_subgoal_pos_.x;
                    double dy = subgoal_pos_.y - last_subgoal_pos_.y;
                    return normalizeAngle(std::atan2(dy, dx));
                }
                return normalizeAngle(arc_heading);
            }
            if (used_avoidance) {
                *used_avoidance = false;
            }
            return normalizeAngle(desired_heading);
        }

        auto accept = [&](const SideTangentResult& c) -> double {
            if (chosen_arc_side_sign) {
                *chosen_arc_side_sign = c.side_sign;
            }
            if (has_selected_hit_point) {
                *has_selected_hit_point = c.has_hit;
            }
            if (selected_hit_point) {
                *selected_hit_point = c.hit_point;
            }
            if (c.has_hit) {
                has_last_hit_heading_ = true;
                last_hit_heading_ = normalizeAngle(c.heading);
            }
            return normalizeAngle(c.heading);
        };

        // 单侧命中，选取未命中的那一侧最靠近目标点的方向（即目标点方向障碍物检测逻辑）
        if (left.has_hit && !right.has_hit) {
            // 只右侧可行，返回右侧圆弧范围内最接近目标点方向的方向，若目标方向也被障碍物阻挡且夹角小于90度则进入回退
            double min_heading = normalizeAngle(arc_heading - omega_max * horizon);
            double max_heading = normalizeAngle(arc_heading);
            double target = normalizeAngle(desired_heading);
            double delta_min = shortestAngularDistance(min_heading, target);
            double delta_max = shortestAngularDistance(max_heading, target);
            double delta_arc = shortestAngularDistance(min_heading, max_heading);
            bool in_range = false;
            if (delta_arc > 0) {
                in_range = (delta_min >= 0 && delta_max <= 0);
            } else {
                in_range = (delta_min >= 0 || delta_max <= 0);
            }
            double chosen_heading = 0.0;
            if (in_range) {
                chosen_heading = target;
            } else {
                double d_to_min = std::abs(shortestAngularDistance(target, min_heading));
                double d_to_max = std::abs(shortestAngularDistance(target, max_heading));
                chosen_heading = (d_to_min < d_to_max) ? min_heading : max_heading;
            }
            // 检查目标方向是否被障碍物阻挡且夹角小于90度
            ObstacleState goal_nearest;
            double goal_clear = std::numeric_limits<double>::infinity();
            bool goal_blocked = findNearestObstacleInForwardSlopeZone(obstacles,
                target,
                plan_speed * horizon,
                plan_speed,
                goal_nearest,
                goal_clear,
                false);
            double angle_diff = std::abs(shortestAngularDistance(arc_heading, target));
            if (goal_blocked && angle_diff < M_PI_2) { // 夹角小于90度
                if (used_avoidance) {
                    *used_avoidance = true;
                }
                if (has_last_subgoal_) {
                    double dx = subgoal_pos_.x - last_subgoal_pos_.x;
                    double dy = subgoal_pos_.y - last_subgoal_pos_.y;
                    return normalizeAngle(std::atan2(dy, dx));
                }
                return normalizeAngle(arc_heading);
            }
            if (used_avoidance) *used_avoidance = true;
            return normalizeAngle(chosen_heading);
        }
        if (!left.has_hit && right.has_hit) {
            // 只左侧可行，返回左侧圆弧范围内最接近目标点方向的方向，若目标方向也被障碍物阻挡且夹角小于90度则进入回退
            double min_heading = normalizeAngle(arc_heading);
            double max_heading = normalizeAngle(arc_heading + omega_max * horizon);
            double target = normalizeAngle(desired_heading);
            double delta_min = shortestAngularDistance(min_heading, target);
            double delta_max = shortestAngularDistance(max_heading, target);
            double delta_arc = shortestAngularDistance(min_heading, max_heading);
            bool in_range = false;
            if (delta_arc > 0) {
                in_range = (delta_min >= 0 && delta_max <= 0);
            } else {
                in_range = (delta_min >= 0 || delta_max <= 0);
            }
            double chosen_heading = 0.0;
            if (in_range) {
                chosen_heading = target;
            } else {
                double d_to_min = std::abs(shortestAngularDistance(target, min_heading));
                double d_to_max = std::abs(shortestAngularDistance(target, max_heading));
                chosen_heading = (d_to_min < d_to_max) ? min_heading : max_heading;
            }
            // 检查目标方向是否被障碍物阻挡且夹角小于90度
            ObstacleState goal_nearest;
            double goal_clear = std::numeric_limits<double>::infinity();
            bool goal_blocked = findNearestObstacleInForwardSlopeZone(obstacles,
                target,
                plan_speed * horizon,
                plan_speed,
                goal_nearest,
                goal_clear,
                false);
            double angle_diff = std::abs(shortestAngularDistance(arc_heading, target));
            if (goal_blocked && angle_diff < M_PI_2) {
                if (used_avoidance) {
                    *used_avoidance = true;
                }
                if (has_last_subgoal_) {
                    double dx = subgoal_pos_.x - last_subgoal_pos_.x;
                    double dy = subgoal_pos_.y - last_subgoal_pos_.y;
                    return normalizeAngle(std::atan2(dy, dx));
                }
                return normalizeAngle(arc_heading);
            }
            if (used_avoidance) *used_avoidance = true;
            return normalizeAngle(chosen_heading);
        }

        int selected = 0;
        // 判断障碍物类型
        const double TIE_EPS = 1e-6;
        if (nearest_obs.polygon.empty()) {
            // 车道线障碍物：比较角度差，若接近相等则选靠近子目标一侧
            double d_left = std::abs(shortestAngularDistance(desired_heading, left.heading));
            double d_right = std::abs(shortestAngularDistance(desired_heading, right.heading));
            if (std::abs(d_left - d_right) > TIE_EPS) {
                selected = (d_left < d_right) ? +1 : -1;
            } else {
                // tie: use subgoal side if available
                if (has_subgoal_) {
                    const Vec2 rel = sub(subgoal_pos_, robot_pos_);
                    const Vec2 fwd = unitFromHeading(desired_heading);
                    const Vec2 left_dir{-fwd.y, fwd.x};
                    const double side_val = dot(rel, left_dir);
                    selected = (side_val > 0.0) ? +1 : -1;
                } else {
                    selected = -1; // fallback to right when unknown
                }
            }
        } else {
            // 激光雷达障碍物：比较 |hit_omega|，若接近相等则用命中点到子目标的距离决定
            double abs_left_omega = std::abs(left.hit_omega);
            double abs_right_omega = std::abs(right.hit_omega);
            if (std::abs(abs_left_omega - abs_right_omega) > TIE_EPS) {
                selected = (abs_left_omega < abs_right_omega) ? +1 : -1;
            } else {
                // tie: prefer the side whose hit point is closer to subgoal (if subgoal exists)
                if (has_subgoal_) {
                    bool left_has = left.has_hit;
                    bool right_has = right.has_hit;
                    if (left_has && right_has) {
                        const double dl = norm(sub(subgoal_pos_, left.hit_point));
                        const double dr = norm(sub(subgoal_pos_, right.hit_point));
                        selected = (dl < dr) ? +1 : -1;
                    } else if (left_has) {
                        selected = +1;
                    } else if (right_has) {
                        selected = -1;
                    } else {
                        // no hit points: fallback to side of subgoal relative to robot
                        const Vec2 rel = sub(subgoal_pos_, robot_pos_);
                        const Vec2 fwd = unitFromHeading(desired_heading);
                        const Vec2 left_dir{-fwd.y, fwd.x};
                        const double side_val = dot(rel, left_dir);
                        selected = (side_val > 0.0) ? +1 : -1;
                    }
                } else {
                    selected = -1; // fallback to right
                }
            }
        }
        // 仅在side_sign发生切换时做积分抑制，圆弧内方向判断不限制
        if (last_selected_side_sign_ == +1 || last_selected_side_sign_ == -1) {
            if (selected != last_selected_side_sign_) {
                if (selected == last_side_sign_candidate_) {
                    side_switch_counter_++;
                } else {
                    side_switch_counter_ = 1;
                    last_side_sign_candidate_ = selected;
                }
                if (side_switch_counter_ >= side_switch_counter_thresh_) {
                    last_selected_side_sign_ = selected;
                    last_side_change_time_ = ros::Time::now();
                    side_switch_counter_ = 0;
                } else {
                    selected = last_selected_side_sign_;
                }
            } else {
                // 没有切换，重置计数器
                side_switch_counter_ = 0;
                last_side_sign_candidate_ = selected;
            }
        } else {
            // 首次赋值
            last_selected_side_sign_ = selected;
            last_side_sign_candidate_ = selected;
            side_switch_counter_ = 0;
        }

        const SideTangentResult& chosen = (selected == +1) ? left : right;
        const double out_heading = accept(chosen);
        if (last_selected_side_sign_ != selected) {
            last_selected_side_sign_ = selected;
            last_side_change_time_ = ros::Time::now();
        }
        return out_heading;
    }

    Vec2 sampleTrajectoryPoint(double start_heading,
                               double speed,
                               double omega,
                               double t,
                               int arc_side_sign) const {
        if (std::abs(omega) < 1e-6) {
            return Vec2{robot_pos_.x + speed * t * std::cos(start_heading),
                        robot_pos_.y + speed * t * std::sin(start_heading)};
        }

        const Vec2 arc_start = computeArcStart(start_heading, arc_side_sign);
        const double h = start_heading + omega * t;
        return Vec2{arc_start.x + (speed / omega) * (std::sin(h) - std::sin(start_heading)),
                    arc_start.y - (speed / omega) * (std::cos(h) - std::cos(start_heading))};
    }

    double evaluateTrajectoryMinClearance(const std::vector<ObstacleState>& obstacles,
                                          double start_heading,
                                          double speed,
                                          double omega,
                                          double horizon,
                                          int arc_side_sign,
                                          bool include_lane_segments = true) const {
        const bool has_lane_segments = !left_lane_segments_.empty() || !right_lane_segments_.empty();
        if (obstacles.empty() && (!include_lane_segments || !has_lane_segments)) {
            return std::numeric_limits<double>::infinity();
        }

        const int steps = std::max(12, speed_arc_sim_steps_);
        const double dt = horizon / static_cast<double>(steps);
        double min_clear = std::numeric_limits<double>::infinity();

        for (int i = 1; i <= steps; ++i) {
            const double t = dt * static_cast<double>(i);
            const Vec2 p = sampleTrajectoryPoint(start_heading, speed, omega, t, arc_side_sign);

            for (size_t k = 0; k < obstacles.size(); ++k) {
                const ObstacleState& o = obstacles[k];
                const double clear = obstacleClearanceAtPoint(o, p);
                if (clear < min_clear) {
                    min_clear = clear;
                }
            }

            if (include_lane_segments) {
                const double lane_half_width = std::max(0.0, lane_obstacle_width_ * 0.5);
                const double lane_expand = robot_radius_ + lane_half_width;
                auto process_lane = [&](const std::vector<LaneSegment>& segs) {
                    for (size_t s = 0; s < segs.size(); ++s) {
                        const double clear = distancePointToSegment(p, segs[s].p0, segs[s].p1) - lane_expand;
                        if (clear < min_clear) {
                            min_clear = clear;
                        }
                    }
                };
                process_lane(left_lane_segments_);
                process_lane(right_lane_segments_);
            }
        }

        return min_clear;
    }

    double evaluateBestArcClearanceForSpeed(const std::vector<ObstacleState>& obstacles,
                                            double start_heading,
                                            double speed,
                                            double horizon) const {
        const double straight_clear = evaluateTrajectoryMinClearance(obstacles,
                                                                     start_heading,
                                                                     speed,
                                                                     0.0,
                                                                     horizon,
                                                                     +1,
                                                                     true);
        if (speed <= 1e-3) {
            return straight_clear;
        }

        const double omega_max = lookupMaxYawRate(speed);
        if (omega_max <= 1e-6) {
            return straight_clear;
        }

        const double left_clear = evaluateTrajectoryMinClearance(obstacles,
                                                                 start_heading,
                                                                 speed,
                                                                 +omega_max,
                                                                 horizon,
                                                                 +1,
                                                                 true);
        const double right_clear = evaluateTrajectoryMinClearance(obstacles,
                                                                  start_heading,
                                                                  speed,
                                                                  -omega_max,
                                                                  horizon,
                                                                  -1,
                                                                  true);
        return std::max(straight_clear, std::max(left_clear, right_clear));
    }

    double computeTargetLinearSpeed(const std::vector<ObstacleState>& obstacles,
                                    double motion_heading,
                                    double current_speed,
                                    double& nearest_clear_out) const {
        const bool has_lane_segments = !left_lane_segments_.empty() || !right_lane_segments_.empty();
        if (obstacles.empty() && !has_lane_segments) {
            nearest_clear_out = std::numeric_limits<double>::infinity();
            return max_linear_vel_;
        }

        const int n = std::max(4, speed_arc_sample_count_);
        const double horizon = std::max(0.5, avoidance_range_time_sec_);
        const double safe_clear = std::max(0.0, speed_arc_safety_clearance_);

        double chosen_speed = 0.0;
        double chosen_clear = -std::numeric_limits<double>::infinity();
        bool has_feasible = false;

        for (int i = n; i >= 0; --i) {
            const double v = max_linear_vel_ * static_cast<double>(i) / static_cast<double>(n);
            const double best_clear = evaluateBestArcClearanceForSpeed(obstacles,
                                                                        motion_heading,
                                                                        v,
                                                                        horizon);

            if (best_clear > chosen_clear) {
                chosen_clear = best_clear;
            }
            if (best_clear >= safe_clear) {
                chosen_speed = v;
                chosen_clear = best_clear;
                has_feasible = true;
                break;
            }
        }

        if (!has_feasible) {
            if (!std::isfinite(chosen_clear) || chosen_clear <= 0.0) {
                chosen_speed = 0.0;
            } else {
                const double ratio = std::max(0.0, std::min(1.0, chosen_clear / std::max(1e-3, safe_clear)));
                chosen_speed = max_linear_vel_ * ratio;
            }
        }

        nearest_clear_out = chosen_clear;
        return std::max(0.0, std::min(max_linear_vel_, chosen_speed));
    }

    double clampHeadingByYawRate(double heading, double linear_speed) const {
        const double omega_max = lookupMaxYawRate(linear_speed);
        const double max_delta = std::max(0.0, omega_max) * std::max(0.01, control_period_sec_);
        const double delta = shortestAngularDistance(robot_yaw_, heading);
        const double clamped = std::max(-max_delta, std::min(max_delta, delta));
        return normalizeAngle(robot_yaw_ + clamped);
    }

    double smoothTargetLinearSpeed(double raw_target,
                                   double current_speed,
                                   double nearest_clear) {
        const double raw = std::max(0.0, std::min(max_linear_vel_, raw_target));
        if (!enable_speed_smoothing_) {
            smoothed_speed_cmd_ = raw;
            has_smoothed_speed_cmd_ = true;
            return raw;
        }

        if (!has_smoothed_speed_cmd_) {
            smoothed_speed_cmd_ = std::max(0.0, std::min(max_linear_vel_, current_speed));
            has_smoothed_speed_cmd_ = true;
        }

        if (std::isfinite(nearest_clear) && nearest_clear <= 0.0) {
            smoothed_speed_cmd_ = 0.0;
            return smoothed_speed_cmd_;
        }

        const double alpha = std::max(0.0, std::min(1.0, speed_lpf_alpha_));
        const double blended = alpha * raw + (1.0 - alpha) * smoothed_speed_cmd_;

        const double dt = std::max(0.01, control_period_sec_);
        const double rise_step = std::max(0.0, speed_rise_limit_mps2_) * dt;
        const double fall_step = std::max(0.0, speed_fall_limit_mps2_) * dt;

        double delta = blended - smoothed_speed_cmd_;
        if (delta > rise_step) {
            delta = rise_step;
        }
        if (delta < -fall_step) {
            delta = -fall_step;
        }

        smoothed_speed_cmd_ = std::max(0.0, std::min(max_linear_vel_, smoothed_speed_cmd_ + delta));
        return smoothed_speed_cmd_;
    }

    double mapOutputLinearSpeed(double v) const {
        if (v <= 0.0) {
            return 0.0;
        }
        if (v < 0.3) {
            return 0.3;
        }
        return v;
    }

    void timerCallback(const ros::TimerEvent&) {
        const std::vector<ObstacleState> all_obstacles = lidar_obstacles_;
        // If external /vo/state has not been updated recently, auto-recover to NORMAL.
        const ros::Time now = ros::Time::now();
        if ((now - last_state_msg_time_).toSec() > vo_state_timeout_sec_) {
            if (vo_state_ != vo_navigation::STATE_NORMAL) {
                vo_state_ = vo_navigation::STATE_NORMAL;
                std_msgs::Int32 s; s.data = vo_state_; vo_state_pub_.publish(s);
                ROS_INFO("/vo/state timeout: reverting to STATE_NORMAL");
            }
        }
        // If this is true, we recently received an external /vo/state and should
        // avoid overriding it with internal auto-detection.
        const bool external_control_active = ((now - last_state_msg_time_).toSec() <= vo_state_timeout_sec_);
        publishClusteredLidarObstacleMarkers(all_obstacles);

        const double current_speed = norm(robot_vel_);
        const double motion_heading = (current_speed > 0.05) ? std::atan2(robot_vel_.y, robot_vel_.x) : robot_yaw_;

        // base desired heading (may be overridden in RECOVER)
        double desired_heading = 0.0;
        if (vo_state_ == vo_navigation::STATE_ODOM_UNHEALTHY) {
            if (has_fallback_path_heading_) {
                desired_heading = fallback_path_heading_;
            } else if (has_last_subgoal_) {
                desired_heading = std::atan2(subgoal_pos_.y - last_subgoal_pos_.y,
                                             subgoal_pos_.x - last_subgoal_pos_.x);
            } else if (has_subgoal_) {
                desired_heading = std::atan2(subgoal_pos_.y - robot_pos_.y, subgoal_pos_.x - robot_pos_.x);
            } else {
                desired_heading = robot_yaw_;
            }
        } else if (vo_state_ == vo_navigation::STATE_RECOVER_OUTSIDE_LANES) {
            // If in recovery and have a planned path, set desired towards next waypoint
            if (!recovery_path_.empty() && recovery_index_ < recovery_path_.size()) {
                const Vec2& wp = recovery_path_[recovery_index_];
                desired_heading = std::atan2(wp.y - robot_pos_.y, wp.x - robot_pos_.x);
            } else if (has_subgoal_) {
                desired_heading = std::atan2(subgoal_pos_.y - robot_pos_.y, subgoal_pos_.x - robot_pos_.x);
            } else {
                desired_heading = robot_yaw_;
            }
        } else {
            desired_heading = has_subgoal_ ? std::atan2(subgoal_pos_.y - robot_pos_.y, subgoal_pos_.x - robot_pos_.x) : robot_yaw_;
        }

        const double subgoal_distance = has_subgoal_ ? norm(sub(subgoal_pos_, robot_pos_)) : -1.0;
        // Only consider goal reached when in NORMAL state (avoid treating subgoal reached during recovery/odom-unhealthy)
        const bool reached_goal = (vo_state_ == vo_navigation::STATE_NORMAL) && has_subgoal_ && std::isfinite(subgoal_distance) && subgoal_distance <= std::max(0.0, goal_stop_distance_);

        double nearest_clear = std::numeric_limits<double>::infinity();
        const double raw_target_linear_x = computeTargetLinearSpeed(all_obstacles, motion_heading, current_speed, nearest_clear);
        const double smoothed_target_linear_x = smoothTargetLinearSpeed(raw_target_linear_x, current_speed, nearest_clear);
        double target_linear_x = mapOutputLinearSpeed(smoothed_target_linear_x);

        bool avoidance_active = false;
        Vec2 selected_hit_point{0.0, 0.0};
        bool has_selected_hit_point = false;
        double optimal_heading = desired_heading;

        if (reached_goal) {
            target_linear_x = 0.0;
            has_smoothed_speed_cmd_ = true;
            smoothed_speed_cmd_ = 0.0;
            optimal_heading = robot_yaw_;
            clearOptimalDirectionMarkers();
        }

        // choose heading for current desired_heading (may be a recovery waypoint)
        const double plan_speed = std::max(0.3, std::max(current_speed, target_linear_x));
        const double horizon = std::max(0.5, avoidance_range_time_sec_);
        const double lookahead = plan_speed * horizon;
        const double omega_max = lookupMaxYawRate(plan_speed);
        publishPlanningGeometryMarkers(robot_yaw_, plan_speed, horizon, omega_max);
        publishHeadingTrapezoidMarker(desired_heading, lookahead, current_speed, 20, 0.2f, 1.0f, 0.2f);

        double current_speed_for_heading = (vo_state_ == vo_navigation::STATE_ODOM_UNHEALTHY) ? target_linear_x : current_speed;

        optimal_heading = chooseOptimalHeading(all_obstacles,
                                               desired_heading,
                                               robot_yaw_,
                                               current_speed_for_heading,
                                               target_linear_x,
                                               nullptr,
                                               &avoidance_active,
                                               &selected_hit_point,
                                               &has_selected_hit_point);
        publishOptimalDirectionMarkers(optimal_heading, avoidance_active && has_selected_hit_point, selected_hit_point);

        // stuck detection: low actual speed while avoidance active (conservative)
        // Only perform internal auto-detection when no recent external /vo/state control.
        if (!external_control_active && vo_state_ != vo_navigation::STATE_RECOVER_OUTSIDE_LANES) {
            if (current_speed < v_stuck_threshold_ && avoidance_active && target_linear_x > 0.01) {
                stuck_counter_++;
            } else {
                stuck_counter_ = 0;
            }
            if (stuck_counter_ >= stuck_threshold_) {
                // enter recovery state
                vo_state_ = vo_navigation::STATE_RECOVER_OUTSIDE_LANES;
                recovery_start_time_ = ros::Time::now();
                // plan recovery path to last known subgoal (prefer last_subgoal if available)
                Vec2 goal = has_last_subgoal_ ? last_subgoal_pos_ : subgoal_pos_;
                if (!has_subgoal_ && has_last_subgoal_) goal = last_subgoal_pos_;
                if (has_subgoal_ || has_last_subgoal_) {
                    recovery_path_ = planRecoveryPath(goal);
                    recovery_index_ = 0;
                    last_replan_time_ = ros::Time::now();
                }
                // publish state
                std_msgs::Int32 s; s.data = vo_state_;
                vo_state_pub_.publish(s);

                ROS_WARN("Auto-entered STATE_RECOVER_OUTSIDE_LANES (stuck_counter=%d)", stuck_counter_);

                // recompute desired_heading towards recovery waypoint (if any) and re-run chooser once
                if (!recovery_path_.empty() && recovery_index_ < recovery_path_.size()) {
                    const Vec2& wp = recovery_path_[recovery_index_];
                    desired_heading = std::atan2(wp.y - robot_pos_.y, wp.x - robot_pos_.x);
                    target_linear_x = recovery_v_;
                    current_speed_for_heading = current_speed; // still use current_speed for arc sim unless odom unhealthy
                    optimal_heading = chooseOptimalHeading(all_obstacles,
                                                           desired_heading,
                                                           robot_yaw_,
                                                           current_speed_for_heading,
                                                           target_linear_x,
                                                           nullptr,
                                                           &avoidance_active,
                                                           &selected_hit_point,
                                                           &has_selected_hit_point);
                }
            }
        }

        // If in recovery state, follow recovery_path_ as temporary subgoals
        if (vo_state_ == vo_navigation::STATE_RECOVER_OUTSIDE_LANES) {
            // Abort recovery if it exceeds allowed duration
            if (!recovery_start_time_.isZero() && (ros::Time::now() - recovery_start_time_).toSec() > recovery_timeout_sec_) {
                vo_state_ = vo_navigation::STATE_NORMAL;
                recovery_path_.clear();
                recovery_index_ = 0;
                recovery_start_time_ = ros::Time(0);
                std_msgs::Int32 s; s.data = vo_state_; vo_state_pub_.publish(s);
                ROS_WARN("Recovery aborted after %.1fs timeout, returning to STATE_NORMAL", recovery_timeout_sec_);
            }
            if (recovery_path_.empty()) {
                // if no path yet and we have a subgoal, try planning
                if (has_subgoal_ || has_last_subgoal_) {
                    Vec2 goal = has_last_subgoal_ ? last_subgoal_pos_ : subgoal_pos_;
                    recovery_path_ = planRecoveryPath(goal);
                    recovery_index_ = 0;
                    last_replan_time_ = ros::Time::now();
                }
            }

            if (!recovery_path_.empty() && recovery_index_ < recovery_path_.size()) {
                const Vec2& wp = recovery_path_[recovery_index_];
                const double dx = wp.x - robot_pos_.x;
                const double dy = wp.y - robot_pos_.y;
                const double dist = std::hypot(dx, dy);
                desired_heading = std::atan2(dy, dx);
                target_linear_x = recovery_v_;

                // if reached this waypoint, advance
                if (dist <= recovery_reach_radius_) {
                    recovery_index_++;
                    if (recovery_index_ >= recovery_path_.size()) {
                        // finished recovery
                        vo_state_ = vo_navigation::STATE_NORMAL;
                        std_msgs::Int32 s; s.data = vo_state_; vo_state_pub_.publish(s);
                        ROS_INFO("Recovery complete, returning to STATE_NORMAL");
                        recovery_path_.clear();
                        recovery_index_ = 0;
                        recovery_start_time_ = ros::Time(0);
                    }
                }
            }
        }

        if (apply_yaw_limit_on_output_) {
            optimal_heading = clampHeadingByYawRate(optimal_heading, std::max(target_linear_x, current_speed));
        }

        speed_controller::SpeedCommand cmd;
        cmd.desired_yaw = static_cast<float>(optimal_heading);
        cmd.desired_linear_x = static_cast<float>(target_linear_x);

        cmd_pub_.publish(cmd);
        writeCycleLog(current_speed,
                      target_linear_x,
                      subgoal_distance,
                      desired_heading,
                      optimal_heading,
                      motion_heading,
                      nearest_clear,
                      avoidance_active,
                      has_selected_hit_point,
                      selected_hit_point,
                      all_obstacles.size());

        if (enable_debug_output_) {
            ROS_INFO_THROTTLE(0.5,
                              "[vo_new] v=%.2f v_raw=%.2f v_cmd=%.2f desired=%.2f optimal=%.2f clear=%.2f obs=%zu stuck=%d state=%d",
                              current_speed, raw_target_linear_x, target_linear_x, desired_heading, optimal_heading,
                              std::isfinite(nearest_clear) ? nearest_clear : -1.0, all_obstacles.size(), stuck_counter_, vo_state_);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "vo_node");
    VONode node;
    ros::spin();
    return 0;
}
