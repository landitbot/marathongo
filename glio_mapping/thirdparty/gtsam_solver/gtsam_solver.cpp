#include "gtsam_solver.h"
#include <gtsam/nonlinear/DoglegOptimizer.h>
using namespace gtsam;
// Eigen::Affine3d转换为gtsam::Pose3
Pose3 affineToPose3(const Eigen::Affine3d& affine) {
    Eigen::Matrix3d rot = affine.rotation();
    Eigen::Vector3d trans = affine.translation();
    return Pose3(Rot3(rot), Point3(trans));
}
 
// gtsam::Pose3转换为Eigen::Affine3d 
Eigen::Affine3d pose3ToAffine(const Pose3& pose) {
    Eigen::Affine3d affine = Eigen::Affine3d::Identity();
    affine.translation() = Eigen::Vector3d(pose.x(), pose.y(), pose.z());
    affine.linear() = pose.rotation().matrix();
    return affine;
}

GTSAMSolver::GTSAMSolver(SolverType type) : solverType_(type) {
    // 3D先验噪声模型
    noiseModel::Diagonal::shared_ptr priorNoise = 
        noiseModel::Diagonal::Sigmas((Vector6() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
    
    // 初始位姿设为原点 
    graph_.addPrior(0, Pose3(), priorNoise);

    // ISAM2 初始化
    if (solverType_ == ISAM2) {
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        parameters.enablePartialRelinearizationCheck = true;
        isam2_ = std::make_unique<gtsam::ISAM2>(parameters);
    }
}
void GTSAMSolver::SetPriorPose(int id, const Eigen::Affine3d& pose, 
                              const Eigen::Matrix<double,6,6>& noise_sigmas) {
    // 确保首次设置时清空历史先验
    if (!initialGuess_.exists(id)) {
        initialGuess_.insert(id, affineToPose3(pose));
    } else {
        initialGuess_.update(id, affineToPose3(pose));
    }
    // 创建噪声模型
    auto noise = noiseModel::Diagonal::Covariance(noise_sigmas);
    // 添加新先验 
    graph_.addPrior(id, affineToPose3(pose), noise);
}
void GTSAMSolver::SetPriorPose(int id, const Eigen::Affine3d& pose, 
                              const gtsam::noiseModel::Base::shared_ptr& noise) {
    // 确保首次设置时清空历史先验
    if (!initialGuess_.exists(id)) {
        initialGuess_.insert(id, affineToPose3(pose));
    } else {
        initialGuess_.update(id, affineToPose3(pose));
    }
    // 添加新先验 
    graph_.addPrior(id, affineToPose3(pose), noise);
}
void GTSAMSolver::AddNode(int id, const Eigen::Affine3d& pose) {
    initialGuess_.insert(id, affineToPose3(pose));
}
 
void GTSAMSolver::AddConstraint(int sourceId, int targetId, 
                              const Eigen::Affine3d& relative_pose,
                              const Eigen::Matrix<double, 6, 6>& covariance) {
    // 创建GTSAM噪声模型 
    noiseModel::Gaussian::shared_ptr noise = 
        noiseModel::Gaussian::Covariance(covariance);
    
    // 添加3D位姿约束 
    graph_.emplace_shared<BetweenFactor<Pose3>>(
        sourceId, targetId, affineToPose3(relative_pose), noise);
    
    // 记录边关系 
    graphEdges_.emplace_back(sourceId, targetId);
}
void GTSAMSolver::AddConstraint(int sourceId, int targetId, 
                              const Eigen::Affine3d& relative_pose,
                              const gtsam::noiseModel::Base::shared_ptr& noise) {
    // 添加3D位姿约束 
    graph_.emplace_shared<BetweenFactor<Pose3>>(
        sourceId, targetId, affineToPose3(relative_pose), noise);
    
    // 记录边关系 
    graphEdges_.emplace_back(sourceId, targetId);
}
void GTSAMSolver::AddConstraint(int sourceId, int targetId, 
                              const Pose3& relative_pose,
                              const gtsam::noiseModel::Base::shared_ptr& noise) {
    // 添加3D位姿约束 
    graph_.emplace_shared<BetweenFactor<Pose3>>(
        sourceId, targetId, relative_pose, noise);
    
    // 记录边关系 
    graphEdges_.emplace_back(sourceId, targetId);
}
void GTSAMSolver::AddGpsPositionFactor(
    int id, 
    const Eigen::Vector3d& gps_position,
    const Eigen::Matrix3d& covariance) 
{
    // 创建噪声模型
    auto noise = gtsam::noiseModel::Gaussian::Covariance(covariance);
    
    // 创建GPS因子 
    gtsam::GPSFactor gpsFactor(
        id, 
        gtsam::Point3(gps_position.x(), gps_position.y(), gps_position.z()),
        noise);
    
    graph_.add(gpsFactor);
}
void GTSAMSolver::AddGpsPositionFactor(
        int id,
        const Eigen::Vector3d& gps_position,
        const gtsam::noiseModel::Base::shared_ptr& noise){
    // 创建GPS因子 
    gtsam::GPSFactor gpsFactor(
        id, 
        gtsam::Point3(gps_position.x(), gps_position.y(), gps_position.z()),
        noise);
    
    graph_.add(gpsFactor);
}
void GTSAMSolver::Compute(bool has_loop_flag) {
    corrections_.clear();
    Values result;
    if (solverType_ == ISAM2) {
        // ISAM2增量优化 
        isam2_->update(graph_, initialGuess_);
        isam2_->update();
        if(has_loop_flag){
          isam2_->update();
          isam2_->update();
          isam2_->update();
          isam2_->update();
          isam2_->update();
        }
        // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
        graph_.resize(0);
        initialGuess_.clear();

        result = isam2_->calculateEstimate();
    } else{
        // 设置优化参数 
        LevenbergMarquardtParams params;
        params.setMaxIterations(100);
        params.setRelativeErrorTol(1e-5);
        
        // 执行优化 
        LevenbergMarquardtOptimizer optimizer(graph_, initialGuess_, params);
        result = optimizer.optimize();
    }
    // 转换结果为Eigen::Affine3d
    for (const auto& kv : result) {
        corrections_.emplace(kv.key, 
            pose3ToAffine(kv.value.cast<Pose3>()));
    }
    isamCurrentEstimate_ = result;
    // for (int i = 0; i < result.size(); ++i) {
    //   auto est = result.at<gtsam::Pose3>(i);
    //   Eigen::Affine3d est_affine3d(est.matrix());
    //   corrections_.push_back(est_affine3d);
    // }
}
void GTSAMSolver::AddHeightFactor(int id, double height, double noise_sigma) {
    // 创建 1 维高斯噪声模型
    // auto noise = gtsam::noiseModel::Isotropic::Sigma(1, noise_sigma);
    auto unaryNoise =
      noiseModel::Diagonal::Sigmas(Vector1(noise_sigma));  // 10cm std on x,y
    // 添加到因子图
    // 注意：这里假设 id 就是 gtsam 的 Key (Symbol)
    graph_.add(HeightFactor(id, height, unaryNoise));
}
void GTSAMSolver::getGraph(std::vector<Eigen::Affine3d>& nodes,
                          std::vector<std::pair<int, int>>& edges) {
    nodes.clear();
    { 
        nodes.reserve(corrections_.size());
        for (const auto& [key, value] : corrections_) {
          int frame_id = key;                           // 键：int类型的帧ID 
          Eigen::Affine3d pose = value;          // 值的第二部分：位姿矩阵 
          nodes.push_back(pose);
        }
    }
    edges = graphEdges_;
}
 
const std::map<int, Eigen::Affine3d>& GTSAMSolver::GetCorrections() const {
    return corrections_;
}
const Values& GTSAMSolver::GetCurrentEstimate() const{
    return isamCurrentEstimate_;
}
Eigen::Matrix<double, 6, 6> GTSAMSolver::GetPoseCovariance(int id) const{
  // 获取最新位姿的协方差矩阵
  gtsam::Matrix covariance;
  try {
      // 计算边际协方差（6x6矩阵，对应Pose3）
      covariance = isam2_->marginalCovariance(id);
      
      // 转换为Eigen矩阵便于使用
      // Eigen::Matrix<double, 6, 6> eigenCov = covariance;
      
      // // 可选的：输出或处理协方差矩阵 
      // std::cout << "Marginal covariance:\n" << eigenCov << std::endl;
  } catch (const std::exception& e) {
      std::cerr << "Error computing marginal covariance: " << e.what() << std::endl;
  }
  return covariance;
}
void GTSAMSolver::Clear()
{
  corrections_.clear();
}
size_t GTSAMSolver::GetNodeNum() const {
    // 返回较大值（包括历史节点）
    if (solverType_ == ISAM2 && isam2_) {
        return isam2_->getVariableIndex().size();  // ISAM2管理的所有变量 
    } else{
        return initialGuess_.size();
    }
}