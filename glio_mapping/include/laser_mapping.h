#ifndef LASER_MAPPING_H
#define LASER_MAPPING_H

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "imu_processing.h"
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "gnss/gps_processor.h"
#include "back/pose_graph_manager.h"
 
namespace glio_mapping {
 
#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)
 
bool& opt_with_gnss();
bool& opt_with_wheel();
bool& opt_with_angle();
nav_msgs::Path& gnss_path();
 
// Signal handler declaration
void SigHandle(int sig);
 
// Logging function
void dump_lio_state_to_log(FILE *fp);
 
// Coordinate transformation functions
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s);
void pointBodyToWorld(PointType const * const pi, PointType * const po);
 
template<typename T>
void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
 
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);
 
// Points cache collection
void points_cache_collect();
 
// Map FOV segmentation
void lasermap_fov_segment();
 
// ROS callback functions
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg);
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
void gnss_cbk(const sensor_msgs::NavSatFixConstPtr& msg_in);
 
// Package synchronization
bool sync_packages(common::MeasureGroup &meas);
 
// Incremental mapping 
void map_incremental();
 
// Publishing functions
void publish_frame_world(const ros::Publisher & pubLaserCloudFull);
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body);
void publish_effect_world(const ros::Publisher & pubLaserCloudEffect);
void publish_map(const ros::Publisher & pubLaserCloudMap);
 
template<typename T>
void set_posestamp(T & out);
 
void publish_odometry(const ros::Publisher & pubOdomAftMapped);
void publish_gnss_path(const ros::Publisher pubPath);
void publish_back_path(const ros::Publisher pub_path);
void publish_path(const ros::Publisher pubPath);
 
// EKF shared model
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
 
/**
 * @brief LaserMapping class - Main class for LiDAR odometry and mapping
 */
class LaserMapping {
public:
    /**
     * @brief Constructor
     * @param nh ROS node handle
     * @param config_path Configuration file path
     */
    LaserMapping(ros::NodeHandle& nh, const std::string& config_path = "");
    
    /**
     * @brief Destructor
     */
    ~LaserMapping();
    
    /**
     * @brief Run the mapping process
     */
    void run();
    
    /**
     * @brief Initialize the mapping system
     */
    void init();
    
    /**
     * @brief Shutdown the mapping system
     */
    void shutdown();
 
private:
    ros::NodeHandle* nh_;
    std::string config_path_;
    
    // ROS subscribers
    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Subscriber sub_gnss_;
    
    // ROS publishers
    ros::Publisher pubLaserCloudFull_;
    ros::Publisher pubLaserCloudFull_body_;
    ros::Publisher pubLaserCloudEffect_;
    ros::Publisher pubLaserCloudMap_;
    ros::Publisher pubOdomAftMapped_;
    ros::Publisher pubPath_;
    ros::Publisher pub_gnss_path_, pub_back_path_;
    
    // Debug files
    FILE* fp_;
    std::ofstream fout_pre_;
    std::ofstream fout_out_;
    std::ofstream fout_dbg_;
    
    // Internal methods
    void readParameters();
    void initROSCommunications();
    void initDebugFiles();
    void mainLoop();
};
} // namespace glio_mapping
 
using namespace glio_mapping;
 
#endif // LASER_MAPPING_H
