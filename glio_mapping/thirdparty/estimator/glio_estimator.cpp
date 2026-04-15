#include "glio_estimator.h"
#include <iostream>
#include <cmath>
namespace glio_mapping {
GlioEstimator::GlioEstimator(const std::string &config_path) :ModuleBase(config_path, "estimator", "Glio Estimator"){
    // 初始化噪声协方差矩阵等 
    // 这里的 process_noise_cov 对应 use-ikfom.hpp 中的定义 
    last_gnss_update_t_ = 0.0;
    last_odo_update_t_ = 0.0;
    imucur_.timestamp = 0;
    readParam("useGpsElevation", useGpsElevation_, true);
}
 
void GlioEstimator::init(const state_ikfom& init_state, const Eigen::MatrixXd& init_cov) {
    // kf_.init(init_state, init_cov);
}
 
void GlioEstimator::addIMUData(const IMUData& imu) {
    imupre_ = imucur_;
    imucur_ = imu;
    if (imu_buffer_.size() < 0.2 * params_.imu_rate) {
      imu_buffer_.push_back(imucur_);
    } else {
      imu_buffer_.pop_front();
      imu_buffer_.push_back(imucur_);
    }
}
 
void GlioEstimator::addGNSSData(const GNSSData& gnss) {
    current_gnss_ = gnss;
}
void GlioEstimator::addAngleData(const AngleData& angle){
  current_angle_ = angle;
}
void GlioEstimator::addAngleData(double timestamp, Eigen::Matrix3d& angle){
  AngleData angle_;
  angle_.timestamp = timestamp;
  angle_.angle = angle;
  angle_.isvalid = true;
  addAngleData(angle_);
}
void GlioEstimator::addOdomData(const OdomData& odom) {
    current_odo_ = odom;
}
void GlioEstimator::addIMUData(double timestamp, Eigen::Vector3d& acc, Eigen::Vector3d& gyro){
  IMUData imu_;
  imu_.timestamp = timestamp;
  imu_.acc = acc;
  imu_.gyro = gyro;
  addIMUData(imu_);
}
void GlioEstimator::addIMUData(const sensor_msgs::Imu::ConstPtr& imu) {
    IMUData imu_;
    imu_.timestamp = imu->header.stamp.toSec();
    imu_.acc.x() = imu->linear_acceleration.x;
    imu_.acc.y() = imu->linear_acceleration.y;
    imu_.acc.z() = imu->linear_acceleration.z;
    imu_.gyro.x() = imu->angular_velocity.x;
    imu_.gyro.y() = imu->angular_velocity.y;
    imu_.gyro.z() = imu->angular_velocity.z;
    addIMUData(imu_);
}
 
void GlioEstimator::addGNSSData(const nav_msgs::Odometry::ConstPtr& gnss, const ObsType& type) {
    GNSSData gnss_;
    switch (type)
    {
    case ObsType::GPS:
      gnss_.timestamp = gnss->header.stamp.toSec();
      gnss_.pos.x() = gnss->pose.pose.position.x;
      gnss_.pos.y() = gnss->pose.pose.position.y;
      gnss_.pos.z() = gnss->pose.pose.position.z;
      gnss_.std.x() = gnss->pose.covariance[2];
      gnss_.std.y() = gnss->pose.covariance[7];
      gnss_.std.z() = gnss->pose.covariance[8];
      gnss_.isvalid = gnss->pose.covariance[9] > 0 ? true: false;
      addGNSSData(gnss_);
      break;
    case ObsType::WHEEL_SPEED:{
      OdomData odom_;
      odom_.timestamp = gnss->header.stamp.toSec();
      odom_.vel.x() = gnss->pose.covariance[3];
      // odom_.vel.y() = gnss->pose.covariance[4];
      // odom_.vel.z() = gnss->pose.covariance[5];
      odom_.vel.y() = 0;
      odom_.vel.z() = 0;
      odom_.isvalid = gnss->pose.covariance[9] > 0 ? true: false;
      addOdomData(odom_);
      break;
    }
    default:
      break;
    }
}
void GlioEstimator::addOdomData(double timestamp, Eigen::Vector3d& vel){
    OdomData odom_;
    odom_.timestamp = timestamp;
    odom_.vel = vel;
    odom_.isvalid = true;
    addOdomData(odom_);
}
void GlioEstimator::addOdomData(const nav_msgs::Odometry::ConstPtr& odom, const ObsType& type) {
    OdomData odom_;
    switch (type)
    {
    case ObsType::WHEEL_SPEED:
      odom_.timestamp = odom->header.stamp.toSec();
      odom_.vel.x() = odom->pose.pose.position.x;
      odom_.vel.y() = odom->pose.pose.position.y;
      odom_.vel.z() = odom->pose.pose.position.z;
      odom_.isvalid = true;
      break;
    default:
      break;
    }
    addOdomData(odom_);
}
// 核心处理逻辑：模仿 WheelGINS::newImuProcess 
void GlioEstimator::process(esekf &kf, Eigen::Matrix<double, 12, 12> &Q) {
 
    // 2. 确定更新时间源 (优先 GNSS，其次 ODO)
    double updatetime = -1.0;
    if (current_gnss_.isvalid) updatetime = current_gnss_.timestamp;
    else if (current_odo_.isvalid) updatetime = current_odo_.timestamp;
    else if (current_angle_.isvalid) updatetime = current_angle_.timestamp;
 
    // 3. 判断时间状态 
    int res = isToUpdate(imupre_.timestamp, imucur_.timestamp, updatetime);
    // if(res != 0) LOG(INFO) << "res:" << res;
    if (res == 0) {
        // 仅传播 
        predict(kf, imupre_, imucur_, Q);
    } 
    else if (res == 1) {
        // 更新时间靠近上一时刻：先更新，后传播 
        // 注意：此时 imupre_ 时刻应该已经过去，这里通常意味着数据滞后处理 
        // 简单起见，若 GNSS 时间戳正好在 imupre_，则更新 
        if (params_.if_fuse_gnss && current_gnss_.isvalid) {
            gnssUpdate(kf);
            current_gnss_.isvalid = false;
        } else if (params_.if_fuse_odo && current_odo_.isvalid) {
            odoNHCUpdate(kf);
            current_odo_.isvalid = false;
        } else if (params_.if_fuse_angle && current_angle_.isvalid) {
            angleUpdate(kf);
            current_angle_.isvalid = false;
        }
        predict(kf, imupre_, imucur_, Q);
    } 
    else if (res == 2) {
        // 更新时间靠近当前时刻：先传播，后更新 
        predict(kf, imupre_, imucur_, Q);
        
        if (params_.if_fuse_gnss && current_gnss_.isvalid) {
            gnssUpdate(kf);
            current_gnss_.isvalid = false;
        } else if (params_.if_fuse_odo && current_odo_.isvalid) {
            odoNHCUpdate(kf);
            current_odo_.isvalid = false;
        } else if (params_.if_fuse_angle && current_angle_.isvalid) {
            angleUpdate(kf);
            current_angle_.isvalid = false;
        }
    } 
    else if (res == 3) {
        // 更新时间在两帧 IMU 中间：插值处理 
        IMUData midimu;
        imuInterpolate(imupre_, imucur_, updatetime, midimu);
 
        // 传播前半段 
        predict(kf, imupre_, midimu, Q);
 
        // 执行更新 
        if (params_.if_fuse_gnss && current_gnss_.isvalid) {
            gnssUpdate(kf);
            current_gnss_.isvalid = false;
        } else if (params_.if_fuse_odo && current_odo_.isvalid) {
            odoNHCUpdate(kf);
            current_odo_.isvalid = false;
        } else if (params_.if_fuse_angle && current_angle_.isvalid) {
            angleUpdate(kf);
            current_angle_.isvalid = false;
        }
 
        // 传播后半段 
        // 注意：需要更新 imupre_ 为 midimu 
        predict(kf, midimu, imucur_, Q);
    }
    auto cur_status = kf.get_x();
    if (ZIHR_num_ >= 2 && last_odo_update_t_ == imucur_.timestamp &&
      !if_ZIHR_available_) {
    if_ZIHR_available_ = true;
    auto euler = Rotation::matrix2euler(cur_status.rot.toRotationMatrix());
    zihr_preheading_ = common::rad2deg(euler[2]);
  }
}
 
int GlioEstimator::isToUpdate(double imutime1, double imutime2, double updatetime) const {
    if (updatetime < 0) return 0;
 
    double dt = 1.0 / params_.imu_rate;
    
    // 判断逻辑参考 WheelGINS 
    if (std::abs(imutime1 - updatetime) < dt && std::abs(imutime1 - updatetime) < std::abs(imutime2 - updatetime)) {
        return 1;
    } else if (std::abs(imutime2 - updatetime) <= dt && std::abs(imutime2 - updatetime) < std::abs(imutime1 - updatetime)) {
        return 2;
    } else if (imutime1 < updatetime && updatetime < imutime2) {
        return 3;
    }
    return 0;
}
 
void GlioEstimator::imuInterpolate(const IMUData& imu1, IMUData& imu2, double timestamp, IMUData& midimu) {
    double lambda = (timestamp - imu1.timestamp) / (imu2.timestamp - imu1.timestamp);
    midimu.timestamp = timestamp;
    midimu.acc = imu1.acc * (1 - lambda) + imu2.acc * lambda;
    midimu.gyro = imu1.gyro * (1 - lambda) + imu2.gyro * lambda;
    
    // 修改 imu2 的时间间隔，供后半段传播使用 
    imu2.timestamp = timestamp; // 注意：这里逻辑需谨慎，实际应修改 imu2 的 dt 而非原始数据 
    // 实际实现中通常不修改原始数据，而是计算 dt 传入 predict 
}
 
void GlioEstimator::predict(esekf &kf, const IMUData& imupre, const IMUData& imucur, Eigen::Matrix<double, 12, 12> &Q) {
    // 构建 input_ikfom 
    input_ikfom in;
    in.acc = imucur.acc;
    in.gyro = imucur.gyro;
    double dt;
    if(imupre.timestamp == 0)
        dt = 1.0 / params_.imu_rate;
    else
        dt = imucur.timestamp - imupre.timestamp;
    if (dt <= 0) return;
 
    // 调用 esekfom 的预测 
    // FastLIO 中 predict 需要计算 F, Q 矩阵，通常在 esekfom 内部通过回调函数实现 
    // 这里假设 esekfom::predict 接口为 
    kf.predict(dt, Q, in); 
}
 
void GlioEstimator::gnssUpdate(esekf &kf) {
    opt_with_gnss() = true;
    kf.update_iterated_dyn_share(); // gnss更新
    opt_with_gnss() = false;
    last_gnss_update_t_ = timestamp();
}
void GlioEstimator::ZUPT(esekf &kf) {
  LOG(INFO) << "ZUPT";
  current_odo_.vel = Eigen::Vector3d::Zero();
  odoUpdate(kf);
  if (if_ZIHR_available_) {
    // Initial static alignment
    Eigen::Vector3d mean_acc = std::accumulate(imu_buffer_.begin(), imu_buffer_.end(), 
    Eigen::Vector3d::Zero().eval(), [&](Eigen::Vector3d sum, IMUData& imu) {
        return sum + imu.acc; 
    })/imu_buffer_.size();
    double roll = atan2(-mean_acc[1], -mean_acc[2]);
    double pitch =
        atan2(mean_acc[0], sqrt(mean_acc[1] * mean_acc[1] +
                                    mean_acc[2] * mean_acc[2]));
    if (params_.if_use_static_bias_update) {
      // todo
    }
    // todo 姿态更新{roll,pitch,zihr_preheading_}
    if (params_.if_use_angular_velocity_update) {
      // todo angularVelUpdate();
    }
  }
}
void GlioEstimator::odoNHCUpdate(esekf &kf) {
    // TODO: 实现 NHC (Non-Holonomic Constraint) 更新 
    // 观测：侧向和垂向速度为 0 
    if (imu_buffer_.size() < 0.2 * params_.imu_rate) return;

    if ((imucur_.timestamp - last_odo_update_t_) < params_.odo_update_interval)
      return;
      
    if (params_.if_use_zupt) detectZUPT(current_odo_.vel);

    if (if_ZUPT_available_) {
      ZUPT(kf);
    } else {
      // LOG(INFO) << "odoUpdate";
      // odoUpdate(kf);
    }
}
void GlioEstimator::odoUpdate(esekf &kf) {
    opt_with_wheel() = true;
    kf.update_iterated_dyn_share(); // gnss更新
    opt_with_wheel() = false;
    last_odo_update_t_ = timestamp();
}
void GlioEstimator::angleUpdate(esekf &kf) {
    opt_with_angle() = true;
    kf.update_iterated_dyn_share(); // gnss更新
    opt_with_angle() = false;
    last_angle_update_t_ = timestamp();
}
void GlioEstimator::checkCov() {
    // 检查协方差是否发散 
    // Eigen::MatrixXd P = kf_.get_P();
    // for (int i = 0; i < P.rows(); i++) {
    //     if (P(i, i) < 0) { ... }
    // }
}
void GlioEstimator::h_share_model_wheel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    ekfom_data.z = Eigen::MatrixXd::Zero(3, 1);
    ekfom_data.h_x = Eigen::MatrixXd::Zero(3, 23);
    ekfom_data.h.resize(3);
    ekfom_data.R = Eigen::MatrixXd::Zero(3, 3);
    ekfom_data.h_v = Eigen::MatrixXd::Identity(3, 3);
    // residual body vel
    Eigen::Vector3d res = current_odo_.vel - s.vel;
    // Eigen::Vector3d res = current_odo_.vel - s.rot.toRotationMatrix().transpose() * s.vel;
    // if(current_odo_.vel.z() == 0)
    //   LOG(INFO) << "wheel res:" << res.transpose() << "wheel vel:" << current_odo_.vel.transpose()<< "wheel s.pos:" << ( s.vel).transpose();
    ekfom_data.h(0) = res.x();
    ekfom_data.h(1) = res.y();
    ekfom_data.h(2) = res.z();
    // jacobian
    Eigen::Matrix3d rot_crossmat;
    rot_crossmat << SKEW_SYM_MATRIX(Eigen::Vector3d(s.rot.toRotationMatrix().transpose() * s.vel)); // 当前状态imu系下 点坐标反对称矩阵
    // ekfom_data.h_x.block<3, 3>(0,3) = -rot_crossmat; // diff w.r.t. rot
    ekfom_data.h_x.block<3, 3>(0,12) = -Eigen::Matrix3d::Identity(); // diff w.r.t. vel
    // covariance
    ekfom_data.R(0, 0) = wheel_obs_R_.x();
    ekfom_data.R(1, 1) = wheel_obs_R_.y();
    ekfom_data.R(2, 2) = wheel_obs_R_.z();
}

void GlioEstimator::h_share_model_gnss(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    ekfom_data.z = Eigen::MatrixXd::Zero(3, 1);
    ekfom_data.h_x = Eigen::MatrixXd::Zero(3, 23);
    ekfom_data.h.resize(3);
    ekfom_data.R = Eigen::MatrixXd::Zero(3, 3);
    ekfom_data.h_v = Eigen::MatrixXd::Identity(3, 3);
    // residual 
    auto pos = current_gnss_.pos;
    if(!useGpsElevation_)
        pos.z() = s.pos.z();
    Eigen::Vector3d res = pos - s.pos;
    // LOG(INFO) << "gnss res:" << res.transpose() << "gnss pos:" << pos.transpose()<< "gnss s.pos:" << s.pos.transpose();
    ekfom_data.h(0) = res.x();
    ekfom_data.h(1) = res.y();
    ekfom_data.h(2) = res.z();
    ekfom_data.h_x.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity(); // d_dp
    ekfom_data.R(0, 0) = gnss_obs_R_.x();
    ekfom_data.R(1, 1) = gnss_obs_R_.y();
    ekfom_data.R(2, 2) = gnss_obs_R_.z();
}
void GlioEstimator::h_share_model_angle(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    ekfom_data.z = Eigen::MatrixXd::Zero(3, 1);
    ekfom_data.h_x = Eigen::MatrixXd::Zero(3, 33);
    ekfom_data.h.resize(3);
    ekfom_data.R = Eigen::MatrixXd::Zero(3, 3);
    ekfom_data.h_v = Eigen::MatrixXd::Identity(3, 3);
    // residual
    Eigen::Matrix3d d_R = s.rot.toRotationMatrix().transpose() * current_angle_.angle;
    Eigen::Vector3d res = Log(d_R);
    // LOG(INFO) << "res " << res.transpose();
    ekfom_data.h(0) = res.x();
    ekfom_data.h(1) = res.y();
    ekfom_data.h(2) = res.z();
    ekfom_data.h_x.block<3, 3>(0,3) = -Rotation::JacobianRInv(res) * d_R.transpose(); // diff w.r.t. rot
    // covariance
    ekfom_data.R(0, 0) = angle_obs_R_.x();
    ekfom_data.R(1, 1) = angle_obs_R_.y();
    ekfom_data.R(2, 2) = angle_obs_R_.z();
}
void GlioEstimator::detectZUPT(Eigen::Vector3d& vel) {
  Eigen::Matrix<double, 6, 1> maxIMUdata, minIMUdata;
  maxIMUdata.setZero();
  minIMUdata.setZero();

  maxIMUdata.array() -= 999;
  minIMUdata.array() += 999;

  for (auto it = imu_buffer_.begin(); it != imu_buffer_.end(); ++it) {
    maxIMUdata.head(3) = maxIMUdata.head(3).cwiseMax(it->gyro);
    minIMUdata.head(3) = minIMUdata.head(3).cwiseMin(it->gyro);
    maxIMUdata.tail(3) = maxIMUdata.tail(3).cwiseMax(it->acc);
    minIMUdata.tail(3) = minIMUdata.tail(3).cwiseMin(it->acc);
  }

  double maxmin_angular_velocity =
      (maxIMUdata.head(3) - minIMUdata.head(3)).cwiseAbs().maxCoeff();
  double maxmin_acceleration =
      (maxIMUdata.tail(3) - minIMUdata.tail(3)).cwiseAbs().maxCoeff();
  Eigen::Vector3d mean_acc = std::accumulate(imu_buffer_.begin(), imu_buffer_.end(), 
    Eigen::Vector3d::Zero().eval(), [&](Eigen::Vector3d sum, IMUData& imu) {
        return sum + imu.acc; 
    })/imu_buffer_.size();

  if (maxmin_acceleration < params_.zupt_special_force_threshold &&
      abs(vel.x()) < params_.zupt_velocity_threshold) {
    if_ZUPT_available_ = true;
    if (maxmin_angular_velocity < params_.zupt_angular_velocity_threshold) {
      ZIHR_num_++;
    } else {
      ZIHR_num_ = 0;
    }
  } else {
    ZIHR_num_ = 0;
    if_ZUPT_available_ = false;
    if_ZIHR_available_ = false;
  }
}
}