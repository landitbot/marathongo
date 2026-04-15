#ifndef POSE_GRAPH_MANAGER_H 
#define POSE_GRAPH_MANAGER_H 
 
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "gtsam_solver/gtsam_solver.h"
#include "use-ikfom.hpp"
#include "common_lib.h"
#include <nav_msgs/Path.h>
#include "rotation.h"
#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"
#include "loop/euclidean_loop.h"

/**
 * 6D位姿点云结构定义
*/
struct PointXYZIQT
{
    PCL_ADD_POINT4D     
    PCL_ADD_INTENSITY;  
    float qw;         
    float qx;
    float qy;
    float qz;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   
} EIGEN_ALIGN16;                    

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIQT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, qw, qw) (float, qx, qx) (float, qy, qy) (float, qz, qz)
                                   (double, time, time))

typedef PointXYZIQT  PointTypeQPose;

/**
 * @brief 位姿图管理类，负责关键帧保存、因子图优化和位姿校正 
 */
namespace glio_mapping {
  
class PoseGraphManager : private ModuleBase {
public:
    using Ptr = std::shared_ptr<PoseGraphManager>;
    
    /**
     * @brief 构造函数 
     */
    PoseGraphManager(const std::string &config_file_path="");
    
    /**
     * @brief 析构函数 
     */
    ~PoseGraphManager();
    
    /**
     * @brief 初始化函数 
     * @param solver_ptr GTSAM求解器指针 
     * @param slam_back_ptr SLAM后端指针 
     * @param map_ptr 地图指针 
     */
    void initialize();
    
    /**
     * @brief 保存关键帧并添加因子 
     * @param kf ESEKF对象引用 
     * @param state_point 当前状态点 
     * @param feats_down_body 当前帧降采样点云 
     * @param lidar_end_time 激光雷达结束时间 
     * @param transformTobeMapped 待映射的变换 
     * @param aLoopIsClosed 是否有闭环 
     */
    void saveKeyFramesAndFactor(
        esekfom::esekf<state_ikfom, 12, input_ikfom>& kf,
        state_ikfom& state_point,
        const pcl::PointCloud<PointType>::Ptr& feats_down_body,
        double lidar_end_time);
    
    /**
     * @brief 校正位姿 
     * @param aLoopIsClosed 是否有闭环 
     */
    void correctPoses();
    
    /**
     * @brief 设置关键帧阈值参数 
     */
    void setKeyFrameThresholds(
        float dist_threshold,
        float angle_threshold);
    
    /**
     * @brief 设置GPS参数 
     */
    void setGPSParams(
        bool use_gps_elevation,
        float gps_cov_threshold,
        float pose_cov_threshold);
    
    /**
     * @brief 获取关键帧位姿3D 
     */
    pcl::PointCloud<PointType>::Ptr getCloudKeyPoses3D() const;
    
    /**
     * @brief 获取关键帧位姿6D 
     */
    pcl::PointCloud<PointTypeQPose>::Ptr getCloudKeyPoses6D() const;
    
    /**
     * @brief 获取全局路径 
     */
    nav_msgs::Path getGlobalPath() const;
    
    /**
     * @brief 设置闭环队列 
     */
    void setLoopQueues(
        std::vector<std::pair<int, int>>& loop_index_queue,
        std::vector<gtsam::Pose3>& loop_pose_queue,
        std::vector<gtsam::noiseModel::Diagonal::shared_ptr>& loop_noise_queue);
    Eigen::Affine3d PointToAffine3d(const PointTypeQPose& point);
    void AddNewPose(float x, float y, float z, const int key_frame_index, const double time, pcl::PointCloud<PointTypeQPose>::Ptr cloud_pose, float qw=1, float qx=0, float qy=0, float qz=0) {
          PointTypeQPose point;
          point.x = x;
          point.y = y;
          point.z = z;
          point.intensity = key_frame_index;
          point.time = time;   // HINT float 无法精确表示所有整数
          point.qw = qw;
          point.qx = qx;
          point.qy = qy;
          point.qz = qz;
          // 添加到位姿集合 
          cloud_pose->push_back(point);
    }
    void AddGNSSPose(float x, float y, float z, const int key_frame_index, const double time, float qw=1, float qx=0, float qy=0, float qz=0) {
        AddNewPose(x,y,z, key_frame_index, time, gnss_cloudKeyPoses6D_, qw, qx, qy, qz);
    }
    void setGnssHeadingNeedInit(bool val) { gnss_heading_need_init_ = val;}
    bool Savetrajectory();
    bool saveGlobalMap(std::string map_name, float mapLeafSize);
    bool copySlamData(std::string &map_name);
    using GnssPathCallback = std::function<nav_msgs::Path&(void)>;
    void GetGnssPathCallback(GnssPathCallback cb) { gnss_path = cb; }
    GnssPathCallback gnss_path;
private:
    /**
     * 点到坐标系原点距离
     */
    template<typename PointT>
    float pointDistance(PointT p)
    {
        return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }

    /**
     * 两点之间距离
     */
    template<typename PointT>
    float pointDistance(PointT p1, PointT p2)
    {
        return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
    }
    /**
     * @brief 判断是否保存当前帧为关键帧 
     */
    bool saveFrame();
    
    /**
     * @brief 添加里程计因子 
     */
    void addOdomFactor();
    
    /**
     * @brief 添加GPS因子 
     */
    void addGPSFactor(double lidar_end_time);
    
    /**
     * @brief 添加闭环因子 
     */
    void addLoopFactor();
    
    /**
     * @brief 更新路径 
     */
    void updatePath(const PointTypeQPose& pose_in);
    
    /**
     * @brief 重建IKd树 
     */
    void recontructIKdTree();
    
    /**
     * @brief 欧拉角转四元数 
     */
    Eigen::Quaterniond eulerToQuat(float roll, float pitch, float yaw);
    void visualizeLoopClosure(
    const ros::Publisher& publisher,
    const std::vector<std::pair<int, int>>& loop_container,
    const std::map<int, std::pair<double, Eigen::Affine3d>> corrections);
    void PublishDescPairs(const ros::Publisher &pubSTD, const ros::Publisher &pubDescri);
 
private:
    // 核心组件指针 
    GTSAMSolver::Ptr solver_ptr_;
    
    // 关键帧位姿存储 
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D_;
    pcl::PointCloud<PointTypeQPose>::Ptr cloudKeyPoses6D_;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D_;
    pcl::PointCloud<PointTypeQPose>::Ptr copy_cloudKeyPoses6D_;
    
    // 未优化位姿存储 
    pcl::PointCloud<PointTypeQPose>::Ptr fastlio_unoptimized_cloudKeyPoses6D_;
    
    // GPS位姿存储 
    pcl::PointCloud<PointTypeQPose>::Ptr gnss_cloudKeyPoses6D_;
    
    // 全局路径 
    nav_msgs::Path globalPath_;
    
    // 关键帧阈值参数 
    float surroundingkeyframeAddingDistThreshold_;
    float surroundingkeyframeAddingAngleThreshold_;
    
    // GPS参数 
    bool useGpsElevation_, gnss_heading_need_init_, aLoopIsClosed_, gnss_back_;
    float gpsCovThreshold_, tolerance_time_;
    float poseCovThreshold_;
    
    // 闭环相关队列 
    std::vector<std::pair<int, int>>* loopIndexQueue_;
    std::vector<gtsam::Pose3>* loopPoseQueue_;
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr>* loopNoiseQueue_;
    
    // 闭环索引容器 
    std::map<int, int> loopIndexContainer_;
    
    // 位姿协方差 
    Eigen::MatrixXd poseCovariance_;
    
    // ISAM2当前估计 
    gtsam::Values isamCurrentEstimate_;
    
    // 重建Kd树相关 
    bool recontructKdTree_;
    int updateKdtreeCount_;
    
    // 可视化参数 
    float globalMapVisualizationSearchRadius_;
    float globalMapVisualizationPoseDensity_;
    float globalMapVisualizationLeafSize_;
    
    Eigen::Affine3d cur_pose;
    common::V3D gnss_obs_R;
    bool pcd_save_en_, trajectory_save_en_;
    // loop
    LoopBase::Ptr loop_manager_ptr;
    std::string loop_closure_method_;
    std::shared_ptr<Logger> log_optimized_, log_unoptimized_, log_gnss_;
    ConfigBaseSetting config_setting_;
    // 互斥锁 
    std::mutex mtx_;
};
}
#endif // POSE_GRAPH_MANAGER_H 