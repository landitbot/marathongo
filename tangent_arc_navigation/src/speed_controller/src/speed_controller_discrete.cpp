#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <speed_controller/SpeedCommand.h>
#include <tf2/utils.h>
#include <angles/angles.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <ctime>
#include <cmath>

// 日志数据结构
struct LogData
{
    ros::Time stamp;
    double odom_linear_x, odom_angular_z, odom_yaw;
    double target_yaw;
    double desired_linear_x;
    double error;               // 航向误差（归一化到 [-pi, pi]）
    double cmd_linear_x, cmd_angular_z;
};

class SpeedController
{
public:
    SpeedController(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : stop_logger_(false)
    {
        // 参数获取（全局命名空间）
        nh.param<double>("speed_controller/kp", kp_, 1.0);
        nh.param<double>("speed_controller/ki", ki_, 0.1);
        nh.param<double>("speed_controller/kd", kd_, 0.05);
        nh.param<double>("speed_controller/max_angular_vel", max_angular_vel_, 1.0);
        nh.param<double>("speed_controller/integral_limit", integral_limit_, 1.0);
        nh.param<double>("speed_controller/steady_state_error_deg", steady_state_error_deg_, 5.0);
        nh.param<double>("speed_controller/max_linear_vel", max_linear_vel_, 1.0);
        nh.param<double>("speed_controller/linear_deadband", linear_deadband_, 0.3);
        nh.param<bool>("speed_controller/enable_turn_speed_limit", enable_turn_speed_limit_, true);
        nh.param<double>("speed_controller/turn_speed_limit_error_deg", turn_speed_limit_error_deg_, 20.0);
        nh.param<double>("speed_controller/turn_speed_limit_linear_vel", turn_speed_limit_linear_vel_, 1.2);

        // 误差阈值转弧度
        steady_state_error_rad_ = steady_state_error_deg_ * M_PI / 180.0;
        turn_speed_limit_error_rad_ = turn_speed_limit_error_deg_ * M_PI / 180.0;

        // 订阅
        odom_sub_ = nh.subscribe("/odometry", 1, &SpeedController::odomCallback, this);
        cmd_sub_ = nh.subscribe("/speed_command", 1, &SpeedController::commandCallback, this);

        // 发布
        cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/fuzzy_cmd_vel", 1);

        // 初始化日志目录
        initLogDirectory();

        // 启动日志线程
        logger_thread_ = std::thread(&SpeedController::logWriterThread, this);

        ROS_INFO("SpeedController initialized with PID gains: kp=%.2f, ki=%.2f, kd=%.2f", kp_, ki_, kd_);
           ROS_INFO("steady_state_error_deg=%.2f, max_linear_vel=%.2f, linear_deadband=%.3f", 
               steady_state_error_deg_, max_linear_vel_, linear_deadband_);
        ROS_INFO("turn_speed_limit: enabled=%s, error_deg=%.2f, max_linear=%.2f",
             enable_turn_speed_limit_ ? "true" : "false",
             turn_speed_limit_error_deg_,
             turn_speed_limit_linear_vel_);
    }

    ~SpeedController()
    {
        stop_logger_ = true;
        queue_cv_.notify_one();
        if (logger_thread_.joinable())
            logger_thread_.join();
        if (log_file_.is_open())
            log_file_.close();
    }

private:
    void initLogDirectory()
    {
        const char* home = getenv("HOME");
        if (!home) {
            ROS_WARN("Cannot get HOME directory, logging disabled");
            return;
        }
        log_dir_ = std::string(home) + "/catkin_ws/logs/speed_controller";
        struct stat st;
        if (stat(log_dir_.c_str(), &st) != 0) {
            std::string cmd = "mkdir -p " + log_dir_;
            int ret = system(cmd.c_str());
            if (ret != 0) {
                ROS_WARN("Failed to create log directory %s", log_dir_.c_str());
                return;
            }
        }

        // 生成带时间戳的文件名
        time_t now = time(nullptr);
        struct tm *tm_info = localtime(&now);
        std::stringstream ss;
        ss << log_dir_ << "/speed_control_" << std::put_time(tm_info, "%Y%m%d_%H%M%S") << ".csv";
        log_filename_ = ss.str();

        log_file_.open(log_filename_.c_str(), std::ios::out);
        if (log_file_.is_open()) {
            log_file_ << "timestamp,odom_linear_x,odom_angular_z,odom_yaw,target_yaw,desired_linear_x,error,cmd_linear_x,cmd_angular_z\n";
            log_file_.flush();
            ROS_INFO("Logging to %s", log_filename_.c_str());
        } else {
            ROS_WARN("Failed to open log file %s", log_filename_.c_str());
        }
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        // 提取当前航向角
        double current_yaw = tf2::getYaw(msg->pose.pose.orientation);

        std::lock_guard<std::mutex> lock(mutex_);
        if (!command_received_)
        {
            ROS_WARN_THROTTLE(2.0, "No speed command received yet.");
            return;
        }

        // 计算角度误差（归一化到[-pi, pi])
        double yaw_error = angles::shortest_angular_distance(current_yaw, desired_yaw_);

        // ===== 稳态误差允许阈值：如果误差小于设定值，视为0，并清零积分 =====
        if (std::abs(yaw_error) < steady_state_error_rad_)
        {
            yaw_error = 0.0;
            integral_ = 0.0;  // 清除积分，避免输出非零
        }

        // ===== 角速度离散控制：0 / ±最大角速度 =====
        double angular_z = 0.0;
        double abs_yaw_error = std::abs(yaw_error);
        if (abs_yaw_error >= steady_state_error_rad_) {
            angular_z = (yaw_error > 0.0 ? max_angular_vel_ : -max_angular_vel_);
        }

        // 限幅
        if (angular_z > max_angular_vel_) angular_z = max_angular_vel_;
        if (angular_z < -max_angular_vel_) angular_z = -max_angular_vel_;

        // 保存误差供下次使用
        prev_error_ = yaw_error;
        last_time_ = ros::Time::now();

        // 构建速度指令
        geometry_msgs::Twist cmd;
        cmd.linear.x = desired_linear_x_;
        if (enable_turn_speed_limit_ && std::abs(yaw_error) > turn_speed_limit_error_rad_) {
            cmd.linear.x = std::min(cmd.linear.x, turn_speed_limit_linear_vel_);
        }
        cmd.angular.z = angular_z;

        // 发布
        cmd_vel_pub_.publish(cmd);

        // 记录本次指令（用于日志）
        last_cmd_ = cmd;

        // 可选日志
        ROS_DEBUG_THROTTLE(0.5, "Current yaw: %.2f, Desired: %.2f, Error: %.2f, Angular cmd: %.2f",
                           current_yaw, desired_yaw_, yaw_error, angular_z);

        // 将数据放入日志队列
        if (log_file_.is_open())
        {
            LogData data;
            data.stamp = ros::Time::now();
            data.odom_linear_x = msg->twist.twist.linear.x;
            data.odom_angular_z = msg->twist.twist.angular.z;
            data.odom_yaw = current_yaw;
            data.target_yaw = desired_yaw_;
            data.desired_linear_x = desired_linear_x_;
            data.error = yaw_error;
            data.cmd_linear_x = last_cmd_.linear.x;
            data.cmd_angular_z = last_cmd_.angular.z;

            std::lock_guard<std::mutex> qlock(queue_mutex_);
            log_queue_.push(data);
            queue_cv_.notify_one();
        }
    }

    void commandCallback(const speed_controller::SpeedCommand::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        desired_yaw_ = msg->desired_yaw;

        const double desired_linear_from_msg = static_cast<double>(msg->desired_linear_x);
        if (std::isfinite(desired_linear_from_msg) && desired_linear_from_msg >= 0.0) {
            desired_linear_x_ = desired_linear_from_msg;
        } else {
            desired_linear_x_ = 0.0;
        }

        if (desired_linear_x_ > max_linear_vel_) desired_linear_x_ = max_linear_vel_;
        if (desired_linear_x_ < 0.0) desired_linear_x_ = 0.0;

        // ========== 线速度死区（可配置）：如果 |v| < deadband 则置零 ==========
        if (std::abs(desired_linear_x_) < linear_deadband_) {
            desired_linear_x_ = 0.0;
        }

        command_received_ = true;
        integral_ = 0.0;
        prev_error_ = 0.0;
        last_time_ = ros::Time::now();
        ROS_INFO("Received new command: yaw=%.2f, msg_linear=%.2f, linear_x=%.2f",
             desired_yaw_, desired_linear_from_msg, desired_linear_x_);
    }

    void logWriterThread()
    {
        ROS_INFO("Log writer thread started");
        while (!stop_logger_)
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] { return !log_queue_.empty() || stop_logger_; });

            while (!log_queue_.empty() && !stop_logger_)
            {
                LogData data = log_queue_.front();
                log_queue_.pop();

                // 临时解锁，允许新数据入队
                lock.unlock();

                if (log_file_.is_open())
                {
                    log_file_ << std::fixed << std::setprecision(6)
                              << data.stamp.toSec() << ","
                              << data.odom_linear_x << ","
                              << data.odom_angular_z << ","
                              << data.odom_yaw << ","
                              << data.target_yaw << ","
                              << data.desired_linear_x << ","
                              << data.error << ","
                              << data.cmd_linear_x << ","
                              << data.cmd_angular_z << "\n";
                    log_file_.flush();
                }

                lock.lock();
            }
        }
        ROS_INFO("Log writer thread stopped");
    }

    // ROS句柄
    ros::Subscriber odom_sub_;
    ros::Subscriber cmd_sub_;
    ros::Publisher cmd_vel_pub_;

    // PID参数
    double kp_, ki_, kd_;
    double max_angular_vel_;
    double integral_limit_;
    double dt_ = 0.02;  // 假设控制周期50Hz，实际由时间差计算

    // 可配置阈值
    double steady_state_error_deg_;
    double steady_state_error_rad_;
    double max_linear_vel_;
    double linear_deadband_;
    bool enable_turn_speed_limit_;
    double turn_speed_limit_error_deg_;
    double turn_speed_limit_error_rad_;
    double turn_speed_limit_linear_vel_;

    // 状态变量
    double desired_yaw_ = 0.0;
    double desired_linear_x_ = 0.0;
    double integral_ = 0.0;
    double prev_error_ = 0.0;
    ros::Time last_time_;
    bool command_received_ = false;
    geometry_msgs::Twist last_cmd_;

    std::mutex mutex_;

    // 日志相关
    std::string log_dir_;
    std::string log_filename_;
    std::ofstream log_file_;
    std::thread logger_thread_;
    std::atomic<bool> stop_logger_;
    std::queue<LogData> log_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "speed_controller");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    SpeedController controller(nh, pnh);

    ros::spin();
    return 0;
}