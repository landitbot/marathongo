#pragma once
#include <pcl/kdtree/kdtree_flann.h>
#include "loopbase.h"
#include "module_base.h"
#include <pcl/registration/gicp.h>
/**
 * @brief 基于欧式距离的点云回环检测器
 * @details 实现基于点云特征点欧式距离匹配的回环检测功能 
 */
class EuclideanLoopManager : public LoopBase, public ModuleBase {
private:
    CloudPtr cloudKeyPoses3D, copy_cloudKeyPoses3D; // 历史关键帧位姿（位置）
    pcl::KdTreeFLANN<PointType>::Ptr kdtree;
    // save all key clouds, optional
    std::mutex kdtree_mutex_;  // 线程安全锁 
    float loopClosureFrequency; //   回环检测频率
    int surroundingKeyframeSize;
    float historyKeyframeSearchRadius;   // 回环检测 radius kdtree搜索半径
    float historyKeyframeSearchTimeDiff; //  帧间时间阈值
    int historyKeyframeSearchNum;        //   回环时多少个keyframe拼成submap
    float historyKeyframeFitnessScore;   // icp 匹配阈值
    ///GICP
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
    pcl::IterativeClosestPoint<PointType, PointType> icp;
public:
    /**
     * @brief 构造函数 
     * @param config_file_path 配置文件路径 
     * @param prefix 配置参数前缀 
     * @param init 是否立即初始化 
     * @param root_dir 根目录路径 
     */
    explicit EuclideanLoopManager(
        const std::string &config_file_path = "", 
        const std::string &prefix = "euclidean_loop",
        bool init = true,
        std::string root_dir = std::string(ROOT_DIR));
 
    /**
     * @brief 生成点云欧式距离描述符 
     * @param input_cloud 输入点云 
     * @param pose 当前帧位姿
     * @param submap_id 关键帧索引
     */
    void GenerateDescriptors(
        CloudPtr &input_cloud,
        Eigen::Affine3d &pose,
        const int submap_id, const double time=0) override;
 
    /**
     * @brief 初始化回环检测系统
     */
    void InitLoopClosure();
 
    /**
     * @brief 搜索回环候选 
     * @param[out] loop_result 回环检测结果 <候选ID, 匹配得分>
     * @param[out] loop_transform 候选位姿变换
     */
    void SearchLoop(
        std::pair<int, double> &loop_result,
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform, double lidar_end_time=0) override;
 
    /**
     * @brief 几何验证
     * @param from_id 查询帧ID 
     * @param to_id 候选帧ID
     * @param[out] transform 验证后的位姿变换 
     * @return bool 验证是否通过 
     */
    bool GeometricOptimization(
        int from_id,
        int to_id,
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform, double &score) override;

    void AddNewPose(const Eigen::Affine3d& new_pose, const int submap_id, const double time=0);
};