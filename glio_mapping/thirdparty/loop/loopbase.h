#pragma once
#include "file_manager.hpp"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <glog/logging.h>
#include "ros/ros.h"
#include "common_lib.h"
#include "timestamp.h"
// 配置参数
typedef struct ConfigBaseSetting {
    /* 基础参数（所有模块共享） */
    double ds_size_ = 0.1;      // 点云下采样分辨率（m）
    int skip_near_num_ = 50;     // 邻近帧跳过数量 
    int sub_frame_num_ = 1;     // 关键帧间隔帧数 
} ConfigBaseSetting;
class LoopBase {
protected:
  ConfigBaseSetting config_setting_;
  int extend_frame_num_ = 25;
  bool pcd_save_en_;
  // if print debug info
  bool print_debug_info_ = 0;
private:
  std::vector<Eigen::Affine3d> origin_pose_vec_;
  std::vector<double> time_vec_;
  mutable std::mutex loop_pair_mtx, origin_pose_mtx, times_mtx, pcd_mtx;
  std::vector<std::pair<int, int>> loop_container_;
  std::vector<Eigen::Affine3d> loop_pose_;
  std::vector<double> loop_score_;
public:
  std::string key_frames_path_ = "";
  std::string trajectory_path_ = "";
  std::string global_path_ = "";
  std::string split_map_path_ = "";
  using Ptr = std::shared_ptr<LoopBase>;
  explicit LoopBase(bool pcd_save_en=false, std::string root_dir=std::string(ROOT_DIR)):pcd_save_en_(pcd_save_en)
  {
    InitDataPath(root_dir);
  }
  virtual ~LoopBase() = default;
 
  //============= 必须实现的纯虚函数 (核心功能) =============//
  /**
   * @brief 从输入点云/图像生成描述符
   * @param input_data 输入数据（点云指针或图像）
   * @param descriptors 输出描述符容器 
   */
  virtual void GenerateDescriptors(CloudPtr &input_cloud, Eigen::Affine3d &pose, const int submap_id, const double time=0) = 0;  // 替换DescriptorType为实际类型 
 
  /**
   * @brief 搜索闭环候选帧 
   * @param descriptors 当前帧描述符
   * @param loop_result 返回最佳匹配帧ID和置信度
   * @param transform 返回匹配的位姿变换 
   * @param matched_pairs 成功匹配的描述符对 
   */
  virtual void SearchLoop( std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d>& loop_transform, double lidar_end_time=0) = 0;
 
  //============= 可选实现的虚函数 (扩展功能) =============//
  /**
   * @brief 添加描述符到数据库（默认空实现）
   */
  virtual void AddDescriptors() {}
 
  /**
   * @brief 几何优化（如ICP、RANSAC）
   * @param source 源点云
   * @param target 目标点云
   * @param transform 优化后的变换 
   */
  virtual bool GeometricOptimization(int from_id, int to_id, std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform, double &score) {
    throw std::runtime_error("GeometricOptimization not implemented");
  }
  virtual const ConfigBaseSetting& getConfigSetting() { 
      return config_setting_; 
  }
  virtual void PublishDescPairs(const ros::Publisher &std_publisher, const ros::Publisher &std_publisher1){}
  bool InitDataPath(std::string root_dir=ROOT_DIR) {
      LOG(INFO) << "InitDataPath:" << root_dir;
      std::string data_path = root_dir;
      if (!FileManager::CreateDirectory(data_path + "slam_data"))
          return false;
      key_frames_path_ = data_path + "slam_data/key_frames/";
      trajectory_path_ = data_path + "slam_data/trajectory/";
      global_path_ = data_path + "slam_data/global_map/";
      split_map_path_ = data_path + "slam_data/split_map/";
      if(pcd_save_en_){
        if (!FileManager::InitDirectory(key_frames_path_, "Point Cloud Key Frames"))
          return false;
        if (!FileManager::InitDirectory(trajectory_path_, "Estimated Trajectory"))
            return false;
        if (!FileManager::InitDirectory(global_path_, "Estimated Trajectory"))
            return false;
        if (!FileManager::InitDirectory(split_map_path_, "Point Cloud split_map Frames"))
            return false;
      }
      return true;
  }
  bool savePointCloud(const CloudPtr& input_scan, std::string file_path) {
      if(!pcd_save_en_ || input_scan->points.size() == 0) return false;
      // 2. 保存点云文件
      input_scan->height = 1;
      input_scan->width = input_scan->points.size();
      std::lock_guard<std::mutex> lock(pcd_mtx);
      if (pcl::io::savePCDFileBinaryCompressed(file_path, *input_scan) == -1) {
          std::cerr << "PCD save failed to " << file_path << std::endl;
          return false;
      }
      return true;
  }
  bool savePointCloud(const CloudPtr& input_scan, int lidar_id) {
      if(!pcd_save_en_ || input_scan->points.size() == 0) return false;
      // 1. 构造完整目录路径 
      const std::string file_path = key_frames_path_ + std::to_string(lidar_id) + ".pcd";
  
      // 2. 保存点云文件
      input_scan->height = 1;
      input_scan->width = input_scan->points.size();
      std::lock_guard<std::mutex> lock(pcd_mtx);
      if (pcl::io::savePCDFileBinaryCompressed(file_path, *input_scan) == -1) {
          std::cerr << "PCD save failed to " << file_path << std::endl;
          return false;
      }
      return true;
  }
  bool loadPointCloud(CloudPtr& output_cloud, int lidar_id) 
  {
      // 1. 构造文件路径
      const std::string file_path = key_frames_path_ + std::to_string(lidar_id) + ".pcd";
      // 2. 检查文件是否存在 
      if (!boost::filesystem::exists(file_path)) {
          std::cerr << "Error: PCD file not found at " << file_path << std::endl;
          return false;
      }
      // 3. 加载点云
      std::lock_guard<std::mutex> lock(pcd_mtx);
      if (pcl::io::loadPCDFile<PointType>(file_path, *output_cloud) == -1) {
          std::cerr << "Error: Failed to load PCD file from " << file_path << std::endl;
          output_cloud.reset(); // 清空指针 
          return false;
      }
      // 4. 验证数据完整性 
      if (output_cloud->points.empty()) {
          std::cerr << "Warning: Loaded empty point cloud from " << file_path << std::endl;
          return false;
      }
      return true;
  }
  bool JointLocalMap(const int submap_id, CloudPtr& map_cloud_ptr, const int searchNum=25) {
      auto origin_pose_vec = getOriginPoses();
      CloudPtr cloud_ptr(new PointCloudType());
      for (int i = submap_id - searchNum; i <= submap_id + searchNum; ++i) {
          if(i < 0 || i >= origin_pose_vec.size()) continue; 
          // a. load back surrounding key scan:
          if(!loadPointCloud(cloud_ptr, i)){
            LOG(INFO) << "size:" <<origin_pose_vec.size();
            return false;
          }
          // b. transform surrounding key scan to map frame:
          pcl::transformPointCloud(*cloud_ptr, *cloud_ptr,origin_pose_vec.at(i));
          *map_cloud_ptr += *cloud_ptr;
      }
      // 2. 体素滤波（降采样）
      pcl::VoxelGrid<PointType> voxel_filter;
      voxel_filter.setInputCloud(map_cloud_ptr);
      voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
      voxel_filter.filter(*map_cloud_ptr);
      return true;
  }

  bool JointScan(const int submap_id, CloudPtr& scan_cloud_ptr) {
      return JointLocalMap(submap_id, scan_cloud_ptr, 0);
  }
  // bool JointCloudMap(CloudPtr& map_cloud_ptr, std::deque<Pose> poses) {
  //     CloudPtr cloud_ptr(new PointCloudType());
  //     for (size_t i = 0; i < poses.size(); ++i) {
  //         // a. load back surrounding key scan:
  //         if(!loadPointCloud(cloud_ptr, i))
  //           return false;
  //         // b. transform surrounding key scan to map frame:
  //         pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, poses[i].position.cast<float>(), poses[i].rotation.cast<float>());
  //         *map_cloud_ptr += *cloud_ptr;
  //     }
  //     return true;
  // }
  bool JointCloudMap(CloudPtr& map_cloud_ptr) {
      // CloudPtr cloud_ptr(new PointCloudType());
      // for (int i = 0; i < origin_pose_vec_.size(); ++i) {
      //     // a. load back surrounding key scan:
      //     if(!loadPointCloud(cloud_ptr, i))
      //       return false;
      //     // b. transform surrounding key scan to map frame:
      //     pcl::transformPointCloud(*cloud_ptr, *cloud_ptr,origin_pose_vec_.at(i));
      //     *map_cloud_ptr += *cloud_ptr;
      // }
      // return true;
  }
  void printDuration(const std::string& tag, int64_t start_us, int64_t end_us) {
      double duration_ms = (end_us - start_us) / 1000.0;
      std::cout << "[TimeProfiler] " << tag << ": " << duration_ms << " ms" << std::endl;
  }
  // 写入单个位姿
  void addOriginPose(const Eigen::Affine3d& pose) {
      std::lock_guard<std::mutex> lock(origin_pose_mtx);
      origin_pose_vec_.push_back(pose);
  }

  // 读取位姿（拷贝返回避免引用失效）
  std::vector<Eigen::Affine3d> getOriginPoses() const {
      std::lock_guard<std::mutex> lock(origin_pose_mtx);
      return origin_pose_vec_;
  }

  // ================ time_vec 操作接口 ================
  // 写入单个时间戳 
  void addTime(double timestamp) {
      std::lock_guard<std::mutex> lock(times_mtx);
      time_vec_.push_back(timestamp);
  }

  // 读取时间戳 
  std::vector<double> getTimes() const {
      std::lock_guard<std::mutex> lock(times_mtx);
      return time_vec_;
  }

  // ================ loop_container_ 操作接口 ================
  // 写入单个闭环对 
  void addLoopPair(int frame1, int frame2, double score, Eigen::Affine3d pose) {
      std::lock_guard<std::mutex> lock(loop_pair_mtx);
      loop_container_.emplace_back(frame1, frame2);
      loop_pose_.push_back(pose);
      loop_score_.emplace_back(score);
  }

  // 读取闭环对
  void getLoopPairs(std::vector<std::pair<int, int>>& loop_containers, std::vector<double>& loop_scores, std::vector<Eigen::Affine3d>& loop_poses) const {
      std::lock_guard<std::mutex> lock(loop_pair_mtx);
      loop_containers = loop_container_;
      loop_scores = loop_score_;
      loop_poses = loop_pose_;
  }
  void clearLoopPairs() {
      std::lock_guard<std::mutex> lock(loop_pair_mtx);
      loop_container_.clear();
      loop_pose_.clear();
      loop_score_.clear();
  }
};