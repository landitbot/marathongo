#include "pose_graph_manager.h"
#include <pcl/common/transforms.h>
#include <execution>
namespace glio_mapping {
// 构造函数 
PoseGraphManager::PoseGraphManager(const std::string &config_file_path)
    : cloudKeyPoses3D_(new pcl::PointCloud<PointType>())
    , cloudKeyPoses6D_(new pcl::PointCloud<PointTypeQPose>())
    , copy_cloudKeyPoses3D_(new pcl::PointCloud<PointType>())
    , copy_cloudKeyPoses6D_(new pcl::PointCloud<PointTypeQPose>())
    , fastlio_unoptimized_cloudKeyPoses6D_(new pcl::PointCloud<PointTypeQPose>())
    , gnss_cloudKeyPoses6D_(new pcl::PointCloud<PointTypeQPose>())
    , surroundingkeyframeAddingDistThreshold_(0.3f)
    , surroundingkeyframeAddingAngleThreshold_(0.1f)
    , gpsCovThreshold_(10.0f), gnss_heading_need_init_(true)
    , poseCovThreshold_(0.0f), aLoopIsClosed_(false)
    , recontructKdTree_(false)
    , updateKdtreeCount_(0)
    , globalMapVisualizationSearchRadius_(1e3)
    , globalMapVisualizationPoseDensity_(10.0)
    , globalMapVisualizationLeafSize_(1.0), ModuleBase(config_file_path, "back", "Back")
{
    std::vector<double> gnss_cov;
    readParam<std::vector<double>>("gnss_cov", gnss_cov, std::vector<double>(3, 0.0001));
    readParam("gnss_back", gnss_back_, true);
    readParam("tolerance_time", tolerance_time_, 0.1f);
    readParam("useGpsElevation", useGpsElevation_, true);
    readParam("pcd_save/pcd_save_en", pcd_save_en_, false);
    readParam("pcd_save/trajectory_save_en", trajectory_save_en_, false);
    readParam<std::string>("loop_closure_method", loop_closure_method_, "euclidean_loop");
    
    // 初始化位姿协方差矩阵 
    poseCovariance_ = Eigen::MatrixXd::Identity(6, 6);
    solver_ptr_ = std::make_shared<GTSAMSolver>(GTSAMSolver::ISAM2);

    gnss_obs_R = common::VecFromArray<double>(gnss_cov);
    globalPath_.header.stamp = ros::Time::now();
    globalPath_.header.frame_id = "camera_init";
    if (loop_closure_method_ == "euclidean_loop"){
      loop_manager_ptr = std::make_shared<EuclideanLoopManager>(config_file_path, "", pcd_save_en_, common::ROOT_FILE_DIR());
    }
    config_setting_ = loop_manager_ptr->getConfigSetting();
    if(trajectory_save_en_) {
      log_optimized_ = std::make_shared<Logger>(loop_manager_ptr->trajectory_path_ + "pose.json");
      log_unoptimized_ = std::make_shared<Logger>(loop_manager_ptr->trajectory_path_ + "unopt_pose.json");
      log_gnss_ = std::make_shared<Logger>(loop_manager_ptr->trajectory_path_ + "gnss.json");
    }
    print_table();
}
 
// 析构函数 
PoseGraphManager::~PoseGraphManager() {
    // 清理资源 
}
 
// 初始化函数 
void PoseGraphManager::initialize() 
{
}
Eigen::Affine3d PoseGraphManager::PointToAffine3d(const PointTypeQPose& point) {
    // 1. 构造平移向量
    Eigen::Vector3d translation(point.x, point.y, point.z);
 
    // 2. 构造旋转四元数（注意归一化）
    Eigen::Quaterniond quat(point.qw, point.qx, point.qy, point.qz);
    quat.normalize();  // 确保单位四元数 
 
    // 3. 组合为Affine3d
    Eigen::Affine3d affine = Eigen::Affine3d::Identity();
    affine.translation() = translation;
    affine.linear() = quat.toRotationMatrix();
 
    return affine;
}
// 保存关键帧并添加因子 
void PoseGraphManager::saveKeyFramesAndFactor(
    esekfom::esekf<state_ikfom, 12, input_ikfom>& kf,
    state_ikfom& state_point,
    const pcl::PointCloud<PointType>::Ptr& feats_down_body,
    double lidar_end_time) 
{
    cur_pose.translation() = state_point.pos;
    cur_pose.linear() = state_point.rot.toRotationMatrix();
    //  计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (saveFrame() == false && !gnss_heading_need_init_)
        return;
    // 激光里程计因子(from fast-lio),  输入的是frame_relative pose  帧间位姿(body 系下)
    addOdomFactor();
    // GPS因子 (UTM -> WGS84)
    addGPSFactor(lidar_end_time);
    // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();
    // 执行优化
    solver_ptr_->Compute(aLoopIsClosed_);
    PointType thisPose3D;
    gtsam::Pose3 latestEstimate;

    // 优化结果
    isamCurrentEstimate_ = solver_ptr_->GetCurrentEstimate();
    // 当前帧位姿结果
    latestEstimate = isamCurrentEstimate_.at<gtsam::Pose3>(isamCurrentEstimate_.size() - 1);
    // HINT 关键帧点云保存放在关键帧位姿前面 避免回环线程同步时得不到对应的点云帧
    loop_manager_ptr->savePointCloud(feats_down_body, cloudKeyPoses6D_->size());// 存储关键帧,没有降采样的点云
    // cloudKeyPoses3D加入当前帧位置
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    // 索引
    thisPose3D.intensity = cloudKeyPoses3D_->size(); //  使用intensity作为该帧点云的index
    cloudKeyPoses3D_->push_back(thisPose3D);         //  新关键帧帧放入队列中

    // cloudKeyPoses6D加入当前帧位姿
    gtsam::Quaternion quat = latestEstimate.rotation().toQuaternion();
    AddNewPose(thisPose3D.x, thisPose3D.y, thisPose3D.z,
    thisPose3D.intensity, lidar_end_time, cloudKeyPoses6D_, quat.w(), quat.x(), quat.y(), quat.z()); //  新关键帧帧放入队列中
    // 位姿协方差
    poseCovariance_ = solver_ptr_->GetPoseCovariance(isamCurrentEstimate_.size() - 1);

    // ESKF状态和方差  更新
    state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
    Eigen::Vector3d pos = latestEstimate.translation();
    Eigen::Quaterniond q(latestEstimate.rotation().matrix()); 

    //  更新状态量
    state_updated.pos = pos;
    state_updated.rot =  q;
    state_point = state_updated; // 对state_point进行更新，state_point可视化用到
    // if(aLoopIsClosed == true )
    kf.change_x(state_updated);  //  对cur_pose 进行isam2优化后的修正
    // LOG(INFO) << "state_updated:" << state_updated.pos.transpose();
    // TODO:  P的修正有待考察，按照yanliangwang的做法，修改了p，会跑飞
    // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P(); // 获取当前的状态估计的协方差矩阵
    // P_updated.setIdentity();
    // P_updated(6, 6) = P_updated(7, 7) = P_updated(8, 8) = 0.00001;
    // P_updated(9, 9) = P_updated(10, 10) = P_updated(11, 11) = 0.00001;
    // P_updated(15, 15) = P_updated(16, 16) = P_updated(17, 17) = 0.0001;
    // P_updated(18, 18) = P_updated(19, 19) = P_updated(20, 20) = 0.001;
    // P_updated(21, 21) = P_updated(22, 22) = 0.00001;
    // kf.change_P(P_updated);
    //  save  unoptimized pose
    AddNewPose(thisPose3D.x, thisPose3D.y, thisPose3D.z,
    thisPose3D.intensity, lidar_end_time, fastlio_unoptimized_cloudKeyPoses6D_, quat.w(), quat.x(), quat.y(), quat.z()); //  新关键帧帧放入队列中
    updatePath(cloudKeyPoses6D_->back()); //  可视化update后的path
}
 
// 校正位姿 
void PoseGraphManager::correctPoses() {
    if (cloudKeyPoses3D_->points.empty())
        return;

    if (aLoopIsClosed_ == true)
    {
        // 清空里程计轨迹
        globalPath_.poses.clear();
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate_.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D_->points[i].x = isamCurrentEstimate_.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D_->points[i].y = isamCurrentEstimate_.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D_->points[i].z = isamCurrentEstimate_.at<gtsam::Pose3>(i).translation().z();
            gtsam::Quaternion quat = isamCurrentEstimate_.at<gtsam::Pose3>(i).rotation().toQuaternion();
            
            cloudKeyPoses6D_->points[i].x = cloudKeyPoses3D_->points[i].x;
            cloudKeyPoses6D_->points[i].y = cloudKeyPoses3D_->points[i].y;
            cloudKeyPoses6D_->points[i].z = cloudKeyPoses3D_->points[i].z;
            cloudKeyPoses6D_->points[i].qw = quat.w();
            cloudKeyPoses6D_->points[i].qx = quat.x();
            cloudKeyPoses6D_->points[i].qy = quat.y();
            cloudKeyPoses6D_->points[i].qz = quat.z();
            // 更新里程计轨迹
            updatePath(cloudKeyPoses6D_->points[i]);
        }
        // 清空局部map， reconstruct  ikdtree submap
        recontructIKdTree();
        aLoopIsClosed_ = false;
    }
}
 
// 判断是否保存当前帧为关键帧 
bool PoseGraphManager::saveFrame() {
    if (cloudKeyPoses3D_->points.empty())
        return true;

    // 前一帧位姿
    Eigen::Affine3d transStart = PointToAffine3d(cloudKeyPoses6D_->back());
    // 当前帧位姿
    Eigen::Affine3d transFinal = cur_pose;
                    
    // 位姿变换增量
    Eigen::Affine3d transBetween = transStart.inverse() * transFinal;
    auto euler = Rotation::matrix2euler(transBetween.rotation()); //  获取上一帧 相对 当前帧的 位姿

    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(euler.x()) < surroundingkeyframeAddingAngleThreshold_ &&
        abs(euler.y()) < surroundingkeyframeAddingAngleThreshold_ &&
        abs(euler.z()) < surroundingkeyframeAddingAngleThreshold_ &&
        euler.norm() < surroundingkeyframeAddingDistThreshold_)
        return false;
    return true;
}
 
// 添加里程计因子 
void PoseGraphManager::addOdomFactor() {
    
    solver_ptr_->AddNode(cloudKeyPoses3D_->size(), cur_pose);
    if (cloudKeyPoses3D_->points.empty())
    {
        // 第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) <<1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
        solver_ptr_->SetPriorPose(cloudKeyPoses3D_->size(), cur_pose, priorNoise);
    }
    else
    {
        // // 添加激光里程计因子
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        auto last_pose = PointToAffine3d(cloudKeyPoses6D_->points.back());
        solver_ptr_->AddConstraint(cloudKeyPoses3D_->size() - 1, cloudKeyPoses3D_->size(), last_pose.inverse()*cur_pose, odometryNoise);
    }
}
 
// 添加GPS因子 
void PoseGraphManager::addGPSFactor(double lidar_end_time) {
    if (!gnss_back_)
        return;
    if (gnss_cloudKeyPoses6D_->empty())
        return;
    // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
    if (cloudKeyPoses3D_->points.empty())
        return;
    else
    {
        if (gnss_heading_need_init_)
        {
          LOG(INFO) << "gnss not initialized !";
          return;
        }
        // if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
        //     return;
    }
    // 位姿协方差很小，没必要加入GPS数据进行校正
    if (poseCovariance_(3,3) < poseCovThreshold_ && poseCovariance_(4,4) < poseCovThreshold_)
        return;
    static PointTypeQPose lastGPSPoint;      // 最新的gps数据
    while (!gnss_cloudKeyPoses6D_->empty())
    {
        // 删除无效帧
        // if (gnss_cloudKeyPoses6D_->front().intensity != 1)
        // {
        //     gnss_cloudKeyPoses6D_->erase(gnss_cloudKeyPoses6D_->begin());
        // }
        // 删除当前帧0.1s之前的里程计
        if (gnss_cloudKeyPoses6D_->front().time < lidar_end_time - tolerance_time_)
        {
            gnss_cloudKeyPoses6D_->erase(gnss_cloudKeyPoses6D_->begin());
        }
        // // 超过当前帧0.1s之后，退出
        else if (gnss_cloudKeyPoses6D_->front().time > lidar_end_time + tolerance_time_)
        {
            break;
        }
        else
        {
            LOG(INFO) << "TIME2:" << gnss_cloudKeyPoses6D_->front().time - lidar_end_time;
            PointTypeQPose thisGPS = gnss_cloudKeyPoses6D_->front();
            gnss_cloudKeyPoses6D_->erase(gnss_cloudKeyPoses6D_->begin());
            // GPS噪声协方差太大，不能用
            // float noise_x = thisGPS.roll ;       //  x 方向的协方差
            // float noise_y = thisGPS.pitch;
            // float noise_z = thisGPS.yaw;      //   z(高层)方向的协方差
            // if (noise_x > gpsCovThreshold_ || noise_y > gpsCovThreshold_)
            //     continue;
            // GPS里程计位置
            float gps_x = thisGPS.x;
            float gps_y = thisGPS.y;
            float gps_z = thisGPS.z;
            
            // (0,0,0)无效数据
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                continue;
            // 每隔5m添加一个GPS里程计
            // if (pointDistance(thisGPS, lastGPSPoint) < 5.0)
            //     continue;
            // else
            //     lastGPSPoint = thisGPS;
            // 添加GPS因子
            gtsam::Vector Vector3(3);
            // Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
            if (!useGpsElevation_)           //  是否使用gps的高度
            {
                gps_z = cur_pose.translation().z();
                Vector3 << gnss_obs_R.x(), gnss_obs_R.y(), 0.001;
            } else{
                Vector3 << gnss_obs_R.x(), gnss_obs_R.y(), gnss_obs_R.z();
            }
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            solver_ptr_->AddGpsPositionFactor(cloudKeyPoses3D_->size(), common::V3D(gps_x, gps_y, gps_z), gps_noise);
            aLoopIsClosed_ = true;
            break;
        }
    }
}
 
// 添加闭环因子 
void PoseGraphManager::addLoopFactor() {
    // todo
}
 
// 更新路径 
void PoseGraphManager::updatePath(const PointTypeQPose& pose_in) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = "camera_init";
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    pose_stamped.pose.orientation.x = pose_in.qx;
    pose_stamped.pose.orientation.y = pose_in.qy;
    pose_stamped.pose.orientation.z = pose_in.qz;
    pose_stamped.pose.orientation.w = pose_in.qw;
    globalPath_.poses.push_back(pose_stamped);
}
 
// 重建IKd树 
void PoseGraphManager::recontructIKdTree() {
    // todo
}
 
// 设置关键帧阈值参数 
void PoseGraphManager::setKeyFrameThresholds(
    float dist_threshold,
    float angle_threshold) 
{
    surroundingkeyframeAddingDistThreshold_ = dist_threshold;
    surroundingkeyframeAddingAngleThreshold_ = angle_threshold;
}
 
// 设置GPS参数 
void PoseGraphManager::setGPSParams(
    bool use_gps_elevation,
    float gps_cov_threshold,
    float pose_cov_threshold) 
{
    useGpsElevation_ = use_gps_elevation;
    gpsCovThreshold_ = gps_cov_threshold;
    poseCovThreshold_ = pose_cov_threshold;
}
 
// 获取关键帧位姿3D 
pcl::PointCloud<PointType>::Ptr PoseGraphManager::getCloudKeyPoses3D() const {
    return cloudKeyPoses3D_;
}
 
// 获取关键帧位姿6D 
pcl::PointCloud<PointTypeQPose>::Ptr PoseGraphManager::getCloudKeyPoses6D() const {
    return cloudKeyPoses6D_;
}
 
// 获取全局路径 
nav_msgs::Path PoseGraphManager::getGlobalPath() const {
    
    return globalPath_;
}
 
// 设置闭环队列 
void PoseGraphManager::setLoopQueues(
    std::vector<std::pair<int, int>>& loop_index_queue,
    std::vector<gtsam::Pose3>& loop_pose_queue,
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr>& loop_noise_queue) 
{
    loopIndexQueue_ = &loop_index_queue;
    loopPoseQueue_ = &loop_pose_queue;
    loopNoiseQueue_ = &loop_noise_queue;
}


void PoseGraphManager::visualizeLoopClosure(
    const ros::Publisher& publisher,
    const std::vector<std::pair<int, int>>& loop_container,
    const std::map<int, std::pair<double, Eigen::Affine3d>> corrections) {
  if (loop_container.empty())
    return;

  if (publisher.getNumSubscribers() < 1)
    return;

  visualization_msgs::MarkerArray markerArray;
  visualization_msgs::Marker markerNode;
  markerNode.header.frame_id = "camera_init";
  markerNode.action = visualization_msgs::Marker::ADD;
  markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
  markerNode.ns = "loop_nodes";
  markerNode.id = 0;
  markerNode.pose.orientation.w = 1;
  markerNode.scale.x = 0.3;
  markerNode.scale.y = 0.3;
  markerNode.scale.z = 0.3;
  markerNode.color.r = 0;
  markerNode.color.g = 0.8;
  markerNode.color.b = 1;
  markerNode.color.a = 1;

  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "camera_init";
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = "loop_edges";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.1;
  markerEdge.color.r = 0.9;
  markerEdge.color.g = 0.9;
  markerEdge.color.b = 0;
  markerEdge.color.a = 1;

  for (auto it = loop_container.begin(); it != loop_container.end(); ++it) {
    int key_cur = it->first;
    int key_pre = it->second;
    geometry_msgs::Point p;
    p.x = corrections.at(key_cur).second.translation().x();
    p.y = corrections.at(key_cur).second.translation().y();
    p.z = corrections.at(key_cur).second.translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
    p.x = corrections.at(key_pre).second.translation().x();
    p.y = corrections.at(key_pre).second.translation().y();
    p.z = corrections.at(key_pre).second.translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
  }

  markerArray.markers.push_back(markerNode);
  markerArray.markers.push_back(markerEdge);
  publisher.publish(markerArray);
}
bool PoseGraphManager::saveGlobalMap(std::string map_name, float mapLeafSize) {
    if(!pcd_save_en_) return true;
    CloudPtr nearKeyframes(new PointCloudType());
    nearKeyframes->clear();
    auto surfcloud_keyframes_size = cloudKeyPoses6D_->size() ;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr temp(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::VoxelGrid<PointType> globalMapPtrFilter;
    globalMapPtrFilter.setLeafSize(0.1, 0.1, 0.1);
    for (int i = 0; i < surfcloud_keyframes_size; ++i) {
        // if(!loadPointCloud(temp,i))
        if(!loop_manager_ptr->loadPointCloud(temp,i))
          return false;
        // 降采样
        globalMapPtrFilter.setInputCloud(temp);
        globalMapPtrFilter.filter(*temp);
        pcl::transformPointCloud(*temp, *temp, PointToAffine3d(cloudKeyPoses6D_->points[i]));
        *nearKeyframes += *temp; //  fast-lio 没有进行特征提取，默认点云就是surf
    }
    if (nearKeyframes->empty())
        return false;
    // 降采样
    globalMapPtrFilter.setLeafSize(mapLeafSize, mapLeafSize, mapLeafSize);
    globalMapPtrFilter.setInputCloud(nearKeyframes);
    globalMapPtrFilter.filter(*nearKeyframes);
    if (pcl::io::savePCDFileBinaryCompressed(map_name, *nearKeyframes) == -1)
    {
        if(pcl::io::savePCDFileBinary(map_name, *nearKeyframes) == -1){
          LOG(INFO) << " savePCDFileBinary success, save_path " << map_name << std::endl;
          return true;
        } else {
          LOG(INFO) << " save fail ! " << std::endl;
          return false;
        }
    }else{
        LOG(INFO) << " savePCDFileBinaryCompressed success, save_path " << map_name << std::endl;
    }
    return true;
}
bool PoseGraphManager::copySlamData(std::string &map_name){
    std::string save_dir = map_name.substr(0, map_name.find_last_of('/'));
    std::string mkdir_dir = "mkdir -p " + save_dir;
    std::string rm_rf = "rm -rf " + save_dir + "/*";
    std::system(mkdir_dir.data());
    std::system(rm_rf.data());
    
    std::string command = "cp -r " + loop_manager_ptr->global_path_ + "../../slam_data" + " " + map_name.substr(0, map_name.find_last_of('/'));
    LOG(INFO) << "command:" << command;
    int sys_ret = system(command.c_str());
    if (sys_ret == -1) {
        LOG(ERROR) << "Failed to execute command:" << command;
        return false;
    } else if (WEXITSTATUS(sys_ret) != 0) {
        LOG(ERROR) << "Copy failed with code: " << WEXITSTATUS(sys_ret) << ", command:" << command;
        return false;
    }
    std::string link_path = loop_manager_ptr->global_path_ + "globalmap.pcd";
    if (symlink(map_name.c_str(), link_path.c_str()) != 0) {
        LOG(ERROR) << "Failed to create symlink: " << strerror(errno);
        // return false;
    } else {
        LOG(INFO) << "Symlink created successfully";
    }
    return true;
}
bool PoseGraphManager::Savetrajectory(){
    if(!trajectory_save_en_)  return true;
    for (int i = 0; i < cloudKeyPoses6D_->size(); ++i) {
        log_optimized_->SavePose(cloudKeyPoses6D_->points[i].time, PointToAffine3d(cloudKeyPoses6D_->points[i]).cast<double>());
    }
    for (int i = 0; i < fastlio_unoptimized_cloudKeyPoses6D_->size(); ++i) {
        log_unoptimized_->SavePose(fastlio_unoptimized_cloudKeyPoses6D_->points[i].time, PointToAffine3d(fastlio_unoptimized_cloudKeyPoses6D_->points[i]).cast<double>());
    }
    for (int i = 0; i < gnss_path().poses.size(); ++i) {
      auto& pose = gnss_path().poses[i];
      log_gnss_->SavePose(pose.header.stamp.toSec(), Eigen::Vector3d(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z ), Eigen::Quaterniond(pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z));
    }
    return true;
}
void PoseGraphManager::PublishDescPairs(const ros::Publisher &pubSTD, const ros::Publisher &pubDescri) {
    if(pubSTD.getNumSubscribers() != 0 || pubDescri.getNumSubscribers() != 0){
      loop_manager_ptr->PublishDescPairs(pubSTD, pubDescri);
    }
}
}