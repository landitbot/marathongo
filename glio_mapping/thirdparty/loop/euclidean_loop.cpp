#include "euclidean_loop.h"
 
EuclideanLoopManager::EuclideanLoopManager(
    const std::string &config_file_path,
    const std::string &prefix,
    bool init,
    std::string root_dir) : 
    LoopBase(init, root_dir), ModuleBase(config_file_path, prefix, "Euclidean Loop Manager"), kdtree(new pcl::KdTreeFLANN<PointType>()) {
    InitLoopClosure();
    cloudKeyPoses3D = pcl::make_shared<PointCloudType>();
    copy_cloudKeyPoses3D = pcl::make_shared<PointCloudType>();
    // 基础参数
    gicp.setMaxCorrespondenceDistance(150);
    gicp.setMaximumIterations(100);
    gicp.setTransformationEpsilon(1e-6);
    gicp.setEuclideanFitnessEpsilon(1e-6);
    gicp.setRANSACIterations(0);
    // GICP特有参数
    gicp.setRotationEpsilon(1e-6);           // 旋转收敛阈值
    gicp.setCorrespondenceRandomness(20);    // 最近邻搜索的随机采样数
    gicp.setMaximumOptimizerIterations(20);  // 优化器最大迭代次数
    icp.setMaximumIterations(20);
}
 
void EuclideanLoopManager::GenerateDescriptors(
    CloudPtr &input_cloud,
    Eigen::Affine3d &pose,
    const int key_frame_index, const double time) {
      static int last_key_id = -1;
    if(key_frame_index > last_key_id){
      AddNewPose(pose, key_frame_index, time);
      // addOriginPose(pose);
      // addTime(time);
    }
    last_key_id = key_frame_index;
}
 
void EuclideanLoopManager::InitLoopClosure() {
    // 接口实现占位
    historyKeyframeSearchRadius = 3;
    readParam("print_debug_info", print_debug_info_, true);
}
 
void EuclideanLoopManager::SearchLoop(
    std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform, double lidar_end_time) {
    static int loopKeyPre = 0;
    // 当前关键帧帧
    loop_result.first = -1;
    {
      std::lock_guard<std::mutex> lock(kdtree_mutex_);
      *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    }
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;                        //  候选关键帧索引
    std::vector<float> pointSearchSqDisLoop;                    //  候选关键帧距离
    if (copy_cloudKeyPoses3D->empty() || !kdtree) {
        // ROS_ERROR("Invalid inputs for KD-Tree");
        return;
    }
    kdtree->setInputCloud(copy_cloudKeyPoses3D); //  历史帧构建kdtree
    kdtree->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(lidar_end_time - copy_cloudKeyPoses3D->at(id).curvature) > 30 && abs(loopKeyPre - id) > 10)
        {
            if(print_debug_info_)
              LOG(INFO) << "loopKeyCur:" << cloudKeyPoses3D->back().intensity << "->id:" << id;
            loop_result.first = id;
            loop_result.second = pointSearchSqDisLoop[id];
            loopKeyPre = id;
            break;
        }
    }
    loop_transform.first = Eigen::Vector3d::Identity();
    loop_transform.second = Eigen::Matrix3d::Identity();
}
 
bool EuclideanLoopManager::GeometricOptimization(
    int from_id,
    int to_id,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform, double &score) {
    const int64_t t_total_start = GetCurrentTime::nowUs();
    Eigen::Matrix4d initial_transform = Eigen::Matrix4d::Identity();
    initial_transform.block<3, 3>(0, 0) = transform.second;
    initial_transform.block<3, 1>(0, 3) = transform.first;
    
    CloudPtr scan_cloud_ptr(new PointCloudType());
    CloudPtr map_cloud_ptr(new PointCloudType());
    // 阶段1: 数据准备
    const int64_t t_data_start = GetCurrentTime::nowUs();
    JointScan(from_id, scan_cloud_ptr);
    JointLocalMap(to_id, map_cloud_ptr);
    if(print_debug_info_){
      LOG(INFO) << "SCAN:" << scan_cloud_ptr->size();
      LOG(INFO) << "MAP:" << map_cloud_ptr->size();
      printDuration("1.DataPreparation", t_data_start, GetCurrentTime::nowUs());
    }
    // 如果特征点较少，返回
    if (scan_cloud_ptr->size() < 300 || map_cloud_ptr->size() < 1000)
        return false;
    // 阶段2: ICP配准
    const int64_t t_icp_start = GetCurrentTime::nowUs();
    // icp.setInputSource(scan_cloud_ptr);
    // icp.setInputTarget(map_cloud_ptr);
    // PointCloudType result;
    // icp.align(result, initial_transform.cast<float>());
    // // scan-to-map匹配
    gicp.setInputSource(scan_cloud_ptr);
    gicp.setInputTarget(map_cloud_ptr);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    gicp.align(*unused_result);
    if(print_debug_info_)
      printDuration("2.ICPRegistration", t_icp_start, GetCurrentTime::nowUs());
 
    // 阶段3: 结果处理 
    const int64_t t_post_start = GetCurrentTime::nowUs();
    Eigen::Matrix4f match_result = gicp.getFinalTransformation();
    transform.first = match_result.block<3, 1>(0, 3).cast<double>();
    transform.second = match_result.block<3, 3>(0, 0).cast<double>();
    score = gicp.getFitnessScore();
    if(print_debug_info_){
      printDuration("3.PostProcessing", t_post_start, GetCurrentTime::nowUs());
      // 总耗时
      printDuration("TotalTime", t_total_start, GetCurrentTime::nowUs()); 
      LOG(INFO) << "score:" << score;
    }
    return gicp.hasConverged() == true && gicp.getFitnessScore() < 3.0;
}
void EuclideanLoopManager::AddNewPose(const Eigen::Affine3d& new_pose, const int key_frame_index, const double time) {
    std::lock_guard<std::mutex> lock(kdtree_mutex_);
    PointType point;
    point.x = new_pose.translation().x();
    point.y = new_pose.translation().y();
    point.z = new_pose.translation().z();
    point.intensity = key_frame_index;
    point.curvature = time;   // HINT float 无法精确表示所有整数
    // 添加到位姿集合 
    cloudKeyPoses3D->push_back(point);
}