#include <ros/ros.h>
#include <costmap_converter/ObstacleArrayMsg.h>
#include <vo_navigation/Obstacle.h>
#include <vo_navigation/ObstacleArray.h>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point32.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

class LidarObstacleNode {
 public:
  LidarObstacleNode() : nh_("~"), tf_listener_(tf_buffer_) {
    nh_.param<std::string>("input_topic", input_topic_, "/costmap_converter/costmap_obstacles");
    nh_.param<std::string>("output_topic", output_topic_, "/obstacles");
    nh_.param<std::string>("target_frame", target_frame_, "map");
    nh_.param("default_obstacle_radius", default_obstacle_radius_, 0.25);
    nh_.param("max_obstacles", max_obstacles_, 200);

    obstacle_sub_ = nh_.subscribe(input_topic_, 1, &LidarObstacleNode::obstacleCallback, this);
    obstacle_pub_ = nh_.advertise<vo_navigation::ObstacleArray>(output_topic_, 1);

    ROS_INFO("lidar_obstacle_node (costmap_converter) started: input=%s output=%s target_frame=%s",
             input_topic_.c_str(), output_topic_.c_str(), target_frame_.c_str());
  }

 private:
  ros::NodeHandle nh_;
  ros::Subscriber obstacle_sub_;
  ros::Publisher obstacle_pub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string input_topic_;
  std::string output_topic_;
  std::string target_frame_;

  double default_obstacle_radius_;
  int max_obstacles_;

  bool transformPoint(double x,
                      double y,
                      double z,
                      const std::string& source_frame,
                      const ros::Time& stamp,
                      bool need_tf,
                      const geometry_msgs::TransformStamped& tf_stamped,
                      double& out_x,
                      double& out_y) const {
    if (!need_tf) {
      out_x = x;
      out_y = y;
      return true;
    }

    geometry_msgs::PointStamped in_pt;
    in_pt.header.stamp = stamp;
    in_pt.header.frame_id = source_frame;
    in_pt.point.x = x;
    in_pt.point.y = y;
    in_pt.point.z = z;

    geometry_msgs::PointStamped out_pt;
    tf2::doTransform(in_pt, out_pt, tf_stamped);
    out_x = out_pt.point.x;
    out_y = out_pt.point.y;
    return true;
  }

  void convertAndPublish(const costmap_converter::ObstacleArrayMsg::ConstPtr& msg) {
    vo_navigation::ObstacleArray out;
    out.header.stamp = msg->header.stamp;
    out.header.frame_id = target_frame_;

    const std::string source_frame = msg->header.frame_id;
    const ros::Time stamp = msg->header.stamp;

    geometry_msgs::TransformStamped tf_stamped;
    bool need_tf = !source_frame.empty() && source_frame != target_frame_;
    if (need_tf) {
      try {
        tf_stamped = tf_buffer_.lookupTransform(target_frame_, source_frame, stamp, ros::Duration(0.05));
      } catch (const tf2::TransformException& ex) {
        try {
          tf_stamped = tf_buffer_.lookupTransform(target_frame_, source_frame, ros::Time(0), ros::Duration(0.05));
        } catch (const tf2::TransformException& ex2) {
          ROS_WARN_THROTTLE(1.0,
                            "TF lookup failed (%s -> %s): %s",
                            source_frame.c_str(),
                            target_frame_.c_str(),
                            ex2.what());
          obstacle_pub_.publish(out);
          return;
        }
      }
    }

    if (msg->obstacles.empty()) {
      obstacle_pub_.publish(out);
      return;
    }

    struct Candidate {
      double cx = 0.0;
      double cy = 0.0;
      double radius = 0.0;
      double vx = 0.0;
      double vy = 0.0;
      std::vector<geometry_msgs::Point32> polygon;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(msg->obstacles.size());

    for (const auto& obs_msg : msg->obstacles) {
      const auto& pts = obs_msg.polygon.points;
      if (pts.empty()) {
        continue;
      }

      std::vector<geometry_msgs::Point32> polygon_pts;
      polygon_pts.reserve(pts.size());

      double sx = 0.0;
      double sy = 0.0;
      for (const auto& p : pts) {
        double px = static_cast<double>(p.x);
        double py = static_cast<double>(p.y);
        transformPoint(px, py, static_cast<double>(p.z), source_frame, stamp, need_tf, tf_stamped, px, py);

        geometry_msgs::Point32 out_p;
        out_p.x = static_cast<float>(px);
        out_p.y = static_cast<float>(py);
        out_p.z = 0.0f;
        polygon_pts.push_back(out_p);

        sx += px;
        sy += py;
      }

      const double cx = sx / static_cast<double>(polygon_pts.size());
      const double cy = sy / static_cast<double>(polygon_pts.size());

      double rad = std::max(0.0, static_cast<double>(obs_msg.radius));
      for (const auto& p : polygon_pts) {
        const double dx = static_cast<double>(p.x) - cx;
        const double dy = static_cast<double>(p.y) - cy;
        rad = std::max(rad, std::hypot(dx, dy));
      }

        Candidate cand;
        cand.cx = cx;
        cand.cy = cy;
        cand.radius = std::max(default_obstacle_radius_, rad);
        cand.vx = obs_msg.velocities.twist.linear.x;
        cand.vy = obs_msg.velocities.twist.linear.y;
        cand.polygon = polygon_pts;
        candidates.push_back(cand);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                const double da = a.cx * a.cx + a.cy * a.cy;
                const double db = b.cx * b.cx + b.cy * b.cy;
                return da < db;
              });

    const std::size_t limit = static_cast<std::size_t>(std::max(1, max_obstacles_));
    const std::size_t n = std::min(limit, candidates.size());
    out.obstacles.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
      const Candidate& c = candidates[i];

      vo_navigation::Obstacle obs;
      obs.position.x = c.cx;
      obs.position.y = c.cy;
      obs.position.z = 0.0;
      obs.velocity.x = c.vx;
      obs.velocity.y = c.vy;
      obs.velocity.z = 0.0;
      obs.radius = static_cast<float>(std::max(default_obstacle_radius_, c.radius));
      obs.polygon.points = c.polygon;
      out.obstacles.push_back(obs);
    }

    obstacle_pub_.publish(out);
  }

  void obstacleCallback(const costmap_converter::ObstacleArrayMsg::ConstPtr& msg) {
    convertAndPublish(msg);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lidar_obstacle_node");
  LidarObstacleNode node;
  ros::spin();
  return 0;
}
