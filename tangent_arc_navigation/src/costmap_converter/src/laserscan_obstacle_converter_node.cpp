/*********************************************************************
 * LaserScan -> ObstacleArrayMsg in the laser frame (no TF, no costmap).
 *********************************************************************/

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <costmap_converter/ObstacleArrayMsg.h>
#include <costmap_converter/ObstacleMsg.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TwistWithCovariance.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

namespace
{

struct Point2d
{
  double x;
  double y;
};

static double cross(const Point2d& O, const Point2d& A, const Point2d& B)
{
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

static void regionQuery(const std::vector<Point2d>& points, std::size_t idx, double eps_sq, std::vector<int>& neigh)
{
  neigh.clear();
  const Point2d& p = points[idx];
  for (std::size_t j = 0; j < points.size(); ++j)
  {
    const Point2d& q = points[j];
    const double dx = p.x - q.x;
    const double dy = p.y - q.y;
    if (dx * dx + dy * dy <= eps_sq)
      neigh.push_back(static_cast<int>(j));
  }
}

/** DBSCAN: labels[i] = -2 unvisited, -1 noise, >=1 cluster id (1-based). */
static void dbscan(const std::vector<Point2d>& points, double eps, int min_pts, std::vector<int>& labels)
{
  const int LAB_UNDEF = -2;
  const int LAB_NOISE = -1;
  const std::size_t n = points.size();
  labels.assign(n, LAB_UNDEF);
  if (n == 0)
    return;

  const double eps_sq = eps * eps;
  int cluster_id = 0;
  std::vector<int> neigh;

  for (std::size_t i = 0; i < n; ++i)
  {
    if (labels[i] != LAB_UNDEF)
      continue;
    regionQuery(points, i, eps_sq, neigh);
    if (static_cast<int>(neigh.size()) < min_pts)
    {
      labels[i] = LAB_NOISE;
      continue;
    }
    ++cluster_id;
    labels[i] = cluster_id;
    std::deque<int> seeds(neigh.begin(), neigh.end());
    while (!seeds.empty())
    {
      const int q = seeds.front();
      seeds.pop_front();
      if (labels[static_cast<std::size_t>(q)] == LAB_NOISE)
        labels[static_cast<std::size_t>(q)] = cluster_id;
      if (labels[static_cast<std::size_t>(q)] != LAB_UNDEF)
        continue;
      labels[static_cast<std::size_t>(q)] = cluster_id;
      regionQuery(points, static_cast<std::size_t>(q), eps_sq, neigh);
      if (static_cast<int>(neigh.size()) >= min_pts)
      {
        for (int p : neigh)
          seeds.push_back(p);
      }
    }
  }
}

/** Andrew monotone chain; removes duplicate last point if closed. */
static bool convexHullMonotoneChain(std::vector<Point2d> pts, geometry_msgs::Polygon& poly)
{
  poly.points.clear();
  if (pts.size() < 1)
    return false;
  std::sort(pts.begin(), pts.end(), [](const Point2d& a, const Point2d& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  pts.erase(std::unique(pts.begin(), pts.end(),
                        [](const Point2d& a, const Point2d& b) { return a.x == b.x && a.y == b.y; }),
            pts.end());
  if (pts.size() == 1)
  {
    poly.points.resize(1);
    poly.points[0].x = static_cast<float>(pts[0].x);
    poly.points[0].y = static_cast<float>(pts[0].y);
    poly.points[0].z = 0.f;
    return true;
  }
  if (pts.size() == 2)
  {
    poly.points.resize(2);
    for (int k = 0; k < 2; ++k)
    {
      poly.points[k].x = static_cast<float>(pts[k].x);
      poly.points[k].y = static_cast<float>(pts[k].y);
      poly.points[k].z = 0.f;
    }
    return true;
  }

  std::vector<Point2d> lower;
  for (const Point2d& p : pts)
  {
    while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), p) <= 0)
      lower.pop_back();
    lower.push_back(p);
  }
  std::vector<Point2d> upper;
  for (int i = static_cast<int>(pts.size()) - 1; i >= 0; --i)
  {
    const Point2d& p = pts[static_cast<std::size_t>(i)];
    while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), p) <= 0)
      upper.pop_back();
    upper.push_back(p);
  }
  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());

  poly.points.resize(lower.size());
  for (std::size_t i = 0; i < lower.size(); ++i)
  {
    poly.points[i].x = static_cast<float>(lower[i].x);
    poly.points[i].y = static_cast<float>(lower[i].y);
    poly.points[i].z = 0.f;
  }
  return true;
}

static void fillDefaultObstacleFields(costmap_converter::ObstacleMsg& o)
{
  o.radius = 0.;
  o.orientation.w = 1.;
  o.orientation.x = o.orientation.y = o.orientation.z = 0.;
  o.velocities.twist.linear.x = o.velocities.twist.linear.y = o.velocities.twist.linear.z = 0.;
  o.velocities.twist.angular.x = o.velocities.twist.angular.y = o.velocities.twist.angular.z = 0.;
  const std::size_t n = o.velocities.covariance.size();
  for (std::size_t i = 0; i < n; ++i)
    o.velocities.covariance[i] = 0.;
}

static bool validRange(float r, float rmin, float rmax)
{
  if (!std::isfinite(r))
    return false;
  return r >= rmin && r <= rmax;
}

class LaserScanObstacleConverter
{
public:
  LaserScanObstacleConverter() : nh_("~")
  {
    std::string scan_topic = "scan";
    nh_.param("scan_topic", scan_topic, scan_topic);
    nh_.param("obstacles_topic", obstacles_topic_, obstacles_topic_);
    nh_.param("cluster_eps", cluster_eps_, 0.15);
    nh_.param("cluster_min_points", cluster_min_points_, 2);
    nh_.param("range_min", range_min_override_, -1.0);
    nh_.param("range_max", range_max_override_, -1.0);
    nh_.param("point_obstacle_radius", point_obstacle_radius_, 0.05);
    nh_.param("publish_noise_points", publish_noise_points_, false);

    if (cluster_eps_ <= 0.)
    {
      ROS_WARN("cluster_eps must be > 0, using 0.15");
      cluster_eps_ = 0.15;
    }
    if (cluster_min_points_ < 1)
      cluster_min_points_ = 1;

    scan_sub_ = nh_.subscribe(scan_topic, 1, &LaserScanObstacleConverter::scanCallback, this);
    obstacle_pub_ = nh_.advertise<costmap_converter::ObstacleArrayMsg>(obstacles_topic_, 10);

    ROS_INFO("laserscan_obstacle_converter: scan_topic=%s, output=%s, frame=laser (no TF)",
             scan_topic.c_str(), obstacles_topic_.c_str());
  }

private:
  void scanCallback(const sensor_msgs::LaserScanConstPtr& msg)
  {
    const float eff_min = (range_min_override_ > 0.) ? static_cast<float>(range_min_override_) : msg->range_min;
    const float eff_max = (range_max_override_ > 0.) ? static_cast<float>(range_max_override_) : msg->range_max;

    std::vector<Point2d> points;
    points.reserve(msg->ranges.size());
    for (std::size_t i = 0; i < msg->ranges.size(); ++i)
    {
      const float r = msg->ranges[i];
      if (!validRange(r, eff_min, eff_max))
        continue;
      const float ang = msg->angle_min + static_cast<float>(i) * msg->angle_increment;
      Point2d p;
      p.x = static_cast<double>(r) * std::cos(static_cast<double>(ang));
      p.y = static_cast<double>(r) * std::sin(static_cast<double>(ang));
      points.push_back(p);
    }

    costmap_converter::ObstacleArrayMsg out;
    out.header = msg->header;

    if (points.empty())
    {
      obstacle_pub_.publish(out);
      return;
    }

    std::vector<int> labels;
    dbscan(points, cluster_eps_, cluster_min_points_, labels);

    const int max_label =
        labels.empty() ? 0 : *std::max_element(labels.begin(), labels.end());
    std::vector<std::vector<std::size_t>> clusters_by_id;
    if (max_label >= 1)
      clusters_by_id.resize(static_cast<std::size_t>(max_label) + 1);

    for (std::size_t i = 0; i < labels.size(); ++i)
    {
      const int lab = labels[i];
      if (lab >= 1)
        clusters_by_id[static_cast<std::size_t>(lab)].push_back(i);
    }

    int obs_id = 0;
    for (std::size_t cid = 1; cid < clusters_by_id.size(); ++cid)
    {
      const std::vector<std::size_t>& idcs = clusters_by_id[cid];
      if (idcs.empty())
        continue;
      std::vector<Point2d> cpts;
      cpts.reserve(idcs.size());
      for (std::size_t ix : idcs)
        cpts.push_back(points[ix]);

      costmap_converter::ObstacleMsg o;
      o.header = out.header;
      o.id = obs_id++;
      fillDefaultObstacleFields(o);

      if (cpts.size() == 1)
      {
        o.polygon.points.resize(1);
        o.polygon.points[0].x = static_cast<float>(cpts[0].x);
        o.polygon.points[0].y = static_cast<float>(cpts[0].y);
        o.polygon.points[0].z = 0.f;
        o.radius = point_obstacle_radius_;
      }
      else if (cpts.size() == 2)
      {
        o.polygon.points.resize(2);
        for (int k = 0; k < 2; ++k)
        {
          o.polygon.points[k].x = static_cast<float>(cpts[static_cast<std::size_t>(k)].x);
          o.polygon.points[k].y = static_cast<float>(cpts[static_cast<std::size_t>(k)].y);
          o.polygon.points[k].z = 0.f;
        }
      }
      else
      {
        convexHullMonotoneChain(std::move(cpts), o.polygon);
      }
      out.obstacles.push_back(o);
    }

    if (publish_noise_points_)
    {
      for (std::size_t i = 0; i < labels.size(); ++i)
      {
        if (labels[i] != -1)
          continue;
        costmap_converter::ObstacleMsg o;
        o.header = out.header;
        o.id = obs_id++;
        fillDefaultObstacleFields(o);
        o.polygon.points.resize(1);
        o.polygon.points[0].x = static_cast<float>(points[i].x);
        o.polygon.points[0].y = static_cast<float>(points[i].y);
        o.polygon.points[0].z = 0.f;
        o.radius = point_obstacle_radius_;
        out.obstacles.push_back(o);
      }
    }

    obstacle_pub_.publish(out);
  }

  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Publisher obstacle_pub_;
  std::string obstacles_topic_{"obstacles"};

  double cluster_eps_{0.15};
  int cluster_min_points_{2};
  double range_min_override_{-1.};
  double range_max_override_{-1.};
  double point_obstacle_radius_{0.05};
  bool publish_noise_points_{false};
};

}  // namespace

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserscan_obstacle_converter");
  LaserScanObstacleConverter node;
  ros::spin();
  return 0;
}
