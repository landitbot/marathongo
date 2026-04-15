#include "imu_processing.h"
 
namespace glio_mapping {
 
ImuProcess::ImuProcess(const std::string &config_path)
    : b_first_frame_(true), imu_need_init_(true), ModuleBase(config_path, "imu_processing", "ImuProcess")
{
  glio_estimaor_ptr_ = std::make_shared<GlioEstimator>(config_path);
  glio_estimaor_ptr_->GetOptWithGnssCallback([&](void) -> bool& {
      return opt_with_gnss();
  });
  glio_estimaor_ptr_->GetOptWithWheelCallback([&](void) -> bool& {
      return opt_with_wheel();
  });
  glio_estimaor_ptr_->GetOptWithAngleCallback([&](void) -> bool& {
      return opt_with_angle();
  });
  double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
  common::V3D lidar_T_wrt_IMU;
  common::M3D lidar_R_wrt_IMU;
  std::vector<double> gnss_cov_,wheel_cov_, angle_cov_, extrinT_, extrinR_;  // lidar-imu rotation
 
  readParam("acc_cov", acc_cov, 0.1);
  readParam("gyr_cov", gyr_cov, 0.1);
  readParam("b_gyr_cov", b_gyr_cov, 0.0001);
  readParam("b_acc_cov", b_acc_cov, 0.0001);
  readParam("extrinsic_T", extrinT_, std::vector<double>({0.065, 0.0, 0.07}));
  readParam("extrinsic_R", extrinR_, std::vector<double>({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  readParam("initUseRtk", initUseRtk_, true);
  readParam("front/gnss_cov", gnss_cov_, std::vector<double>(3, 0.0001));
  readParam("front/gnss_front", gnss_front_, true);
  readParam("front/wheel_cov", wheel_cov_, std::vector<double>(3, 0.0001));
  readParam("front/wheel_front", wheel_front_, false);
  readParam("front/static_wheel_front", static_wheel_front_, false);
  readParam("front/virtual_wheel_front", virtual_wheel_front_, false);
  readParam("front/angle_cov", angle_cov_, std::vector<double>(3, 0.0001));
  readParam("front/angle_front", angle_front_, false);
  readParam("static_rot_method", static_rot_method_, 3);
  readParam("initheading", initheading_, -999);
  readParam("gnss_heading_need_init", gnss_heading_need_init_, true);
  readParam("useGpsElevation", glio_estimaor_ptr_->useGpsElevation(), true);
  print_table();
  lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
  lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);
 
  set_extrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
  set_gyr_cov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
  set_acc_cov(common::V3D(acc_cov, acc_cov, acc_cov));
  set_gyr_bias_cov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  set_acc_bias_cov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
  glio_estimaor_ptr_->gnssObsR() = common::VecFromArray<double>(gnss_cov_);
  glio_estimaor_ptr_->wheelObsR() = common::VecFromArray<double>(wheel_cov_);
  glio_estimaor_ptr_->angleObsR() = common::VecFromArray<double>(angle_cov_);
 
  init_iter_num = 1;
  Q = process_noise_cov();
  mean_acc      = common::V3D(0, 0, -1.0);
  mean_gyr      = common::V3D(0, 0, 0);
  angvel_last     = common::Zero3d;
  last_imu_.reset(new sensor_msgs::Imu());
}
 
ImuProcess::~ImuProcess() {}
 
void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = common::V3D(0, 0, -1.0);
  mean_gyr      = common::V3D(0, 0, 0);
  angvel_last       = common::Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}
 
void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}
 
void ImuProcess::set_extrinsic(const common::V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}
 
void ImuProcess::set_extrinsic(const common::V3D &transl, const common::M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}
 
void ImuProcess::set_gyr_cov(const common::V3D &scaler)
{
  cov_gyr_scale = scaler;
}
 
void ImuProcess::set_acc_cov(const common::V3D &scaler)
{
  cov_acc_scale = scaler;
}
 
void ImuProcess::set_gyr_bias_cov(const common::V3D &b_g)
{
  cov_bias_gyr = b_g;
}
 
void ImuProcess::set_acc_bias_cov(const common::V3D &b_a)
{
  cov_bias_acc = b_a;
}
 
void ImuProcess::IMU_init(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  common::V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }
 
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
 
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;
 
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);
 
    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;
 
    N ++;
  }
  init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * common::G_m_s2);
  // 使用重力做水平面
  common::M3D rot_init; 
  common::V3D ba;
  common::V3D tmp_gravity = - common::G_m_s2 * mean_acc / mean_acc.norm();
  SetInitPose(tmp_gravity, rot_init, ba);
  init_state.ba = ba;
  //state_inout.rot = common::Eye3d; // Exp(mean_acc.cross(common::V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);
 
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();
 
}
void ImuProcess::SetInitPose(common::V3D &tmp_gravity, common::M3D &rot, common::V3D &ba)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  // V3D tmp_gravity = - mean_acc / mean_acc.norm() * G_m_s2; // state_gravity;
  common::V3D gravity_vec = common::V3D(0,0,-1); // 目标重力方向
  double similarity = gravity_vec.dot(mean_acc / mean_acc.norm());
  if (static_rot_method_ == 1){
    common::V3D gravity_vec = common::V3D(0,0,-1); // 目标重力方向
    // 计算将测量重力旋转到目标重力方向的旋转矩阵
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(tmp_gravity, gravity_vec);
    rot = q.toRotationMatrix();
    if (similarity < 0) {
      ba = mean_acc - rot.transpose() * Eigen::Vector3d(0,0, common::G_m_s2);
    } else{
      ba = mean_acc + rot.transpose() * Eigen::Vector3d(0,0, common::G_m_s2);
    }
  } else if(static_rot_method_ == 2){
    common::M3D hat_grav;
    common::V3D _gravity(0,0,-common::G_m_s2);
    hat_grav << 0.0, _gravity(2), -_gravity(1),
                -_gravity(2), 0.0, _gravity(0),
                _gravity(1), -_gravity(0), 0.0;
    double align_norm = (hat_grav * tmp_gravity).norm() / tmp_gravity.norm() / _gravity.norm();
    double align_cos = _gravity.transpose() * tmp_gravity;
    align_cos = align_cos / _gravity.norm() / tmp_gravity.norm();
    if (align_norm < 1e-6)
    {
      if (align_cos > 1e-6)
      {
        rot = common::Eye3d;
      }
      else
      {
        rot = -common::Eye3d;
      }
    }
    else
    {
      common::V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos); 

      rot = Exp<double>(align_angle(0), align_angle(1), align_angle(2));
    }
    if (similarity < 0) {
      ba = mean_acc - rot.transpose() * Eigen::Vector3d(0,0, common::G_m_s2);
    } else{
      ba = mean_acc + rot.transpose() * Eigen::Vector3d(0,0, common::G_m_s2);
    }
  } else{
    if (similarity < 0) {//imu放置Z轴向上
      double roll = Rotation::heading(atan2(-mean_acc[1], -mean_acc[2]) + M_PI);
      double pitch =
          -atan2(mean_acc[0], sqrt(mean_acc[1] * mean_acc[1] +
                                      mean_acc[2] * mean_acc[2]));
      if(initheading_ != -999) {
        rot = Rotation::euler2matrix(common::V3D(roll, pitch, common::deg2rad(initheading_)));
        setGnssHeadingNeedInit(false);
        LOG(INFO) << "euler2:" << common::rad2deg(Rotation::matrix2euler(rot)).transpose()
              << ", ba:" << ba.transpose();
      }
      else
        rot = Rotation::euler2matrix(common::V3D(roll, pitch, 0));
      ba = mean_acc - rot.transpose() * Eigen::Vector3d(0,0, common::G_m_s2);

    } else {//imu放置Z轴向下
      double roll = Rotation::heading(atan2(-mean_acc[1], -mean_acc[2]));
      double pitch =
          atan2(mean_acc[0], sqrt(mean_acc[1] * mean_acc[1] +
                                      mean_acc[2] * mean_acc[2]));
      if(initheading_ != -999) {
        rot = Rotation::euler2matrix(common::V3D(roll, pitch, common::deg2rad(initheading_)));
        setGnssHeadingNeedInit(false);
      }
      else
        rot = Rotation::euler2matrix(common::V3D(roll, pitch, 0));
      ba = mean_acc + rot.transpose() * Eigen::Vector3d(0,0, common::G_m_s2);
    }
  }
}
void ImuProcess::UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  auto curr_gnss = meas.gnss;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
 
  double pcl_beg_time = meas.lidar_beg_time;
  double pcl_end_time = meas.lidar_end_time;
  /*** case when gnss_ meas is in the last lidar scan period but received in this period ***/
  if (gnss_front_ && !curr_gnss.empty())
  {
      double gnss_time = curr_gnss.front()->header.stamp.toSec();
      if (gnss_time < last_lidar_end_time_) // gnss_ is in the last period
      {
          if (!gnss_heading_need_init_)
          {
              glio_estimaor_ptr_->addGNSSData(curr_gnss.front());
              glio_estimaor_ptr_->process(kf_state, Q);
          }else{
              LOG(INFO) << "gnss_ not initialized ";
          }
          curr_gnss.pop_front();
      }
  }
 
    /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;
 
  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(common::set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
 
  /*** forward propagation at each imu point ***/
  common::V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  common::M3D R_imu;
 
  double dt = 0;
 
  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
 
    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
 
    acc_avr     = acc_avr * common::G_m_s2 / mean_acc.norm(); // - state_inout.ba;
 
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    glio_estimaor_ptr_->addIMUData(tail->header.stamp.toSec(), acc_avr, angvel_avr);
    glio_estimaor_ptr_->process(kf_state, Q);
    // kf_state.predict(dt, Q, in);
    /*** case when gnss_ meas is in the last lidar scan period but received in this period ***/
    if (gnss_front_ && !curr_gnss.empty())
    {
        double gnss_time = curr_gnss.front()->header.stamp.toSec();
        if (gnss_time < head->header.stamp.toSec()){
            curr_gnss.pop_front();
        }else{
            if (gnss_time < tail->header.stamp.toSec()){ // gnss位于两个imu之间
                if (!gnss_heading_need_init_)
                {
                  glio_estimaor_ptr_->addGNSSData(curr_gnss.front());
                  glio_estimaor_ptr_->process(kf_state, Q);
                }else{
                    LOG(INFO) << "gnss_ not initialized ";
                }
                curr_gnss.pop_front();
            }
        }
    }
 
    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(common::set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }
 
  /*** calculated the pos and attitude prediction at the frame-end ***/
  // double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  // dt = note * (pcl_end_time - imu_end_time);
  // kf_state.predict(dt, Q, in);
  glio_estimaor_ptr_->addIMUData(pcl_end_time > imu_end_time ? pcl_end_time : imu_end_time, acc_avr, angvel_avr);
  glio_estimaor_ptr_->process(kf_state, Q);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;
 
  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
 
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_imu<<MAT_FROM_ARRAY(head->rot);
      // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
      vel_imu<<VEC_FROM_ARRAY(head->vel);
      pos_imu<<VEC_FROM_ARRAY(head->pos);
      acc_imu<<VEC_FROM_ARRAY(tail->acc);
      angvel_avr<<VEC_FROM_ARRAY(tail->gyr);
 
      for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
      {
          dt = it_pcl->curvature / double(1000) - head->offset_time;
 
          /* Transform to the 'end' frame, using only the rotation
            * Note: Compensation direction is INVERSE of Frame's moving direction
            * So if we want to compensate a point at timestamp-i to the frame-e
            * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
          common::M3D R_i(R_imu * Exp(angvel_avr, dt));
 
          common::V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
          common::V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
          common::V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
 
          // save Undistorted points and their rotation
          it_pcl->x = P_compensate(0);
          it_pcl->y = P_compensate(1);
          it_pcl->z = P_compensate(2);
 
          if (it_pcl == pcl_out.points.begin()) break;
      }
  }
}
 
void ImuProcess::Process(const common::MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();
 
  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);
 
  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);
 
    
    last_imu_   = meas.imu.back();
 
    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      if(initUseRtk_){
        if(meas.gnss.empty()) {
          LOG(ERROR) << "Note: When not using GPS for initialization, initUseRtk must be set to false.";
          return;
        }
        Eigen::Vector3d gnss_pose(meas.gnss.back()->pose.pose.position.x, meas.gnss.back()->pose.pose.position.y, meas.gnss.back()->pose.pose.position.z);
        init_state.pos = gnss_pose;
        if(meas.gnss.back()->pose.covariance[9] == 2){
          double yaw = Rotation::normalizeHeadingDeg(-meas.gnss.back()->pose.covariance[6] + 90);
          init_state.rot = Rotation::euler2matrix(common::V3D(Rotation::matrix2euler(init_state.rot.toRotationMatrix())[0], Rotation::matrix2euler(init_state.rot.toRotationMatrix())[1], common::deg2rad(yaw)));
          setGnssHeadingNeedInit(false);
          LOG(INFO) << "INITIAL gnss HEADING gnss:" << common::rad2deg(Rotation::matrix2euler(init_state.rot.toRotationMatrix())).transpose();
        }
        kf_state.change_x(init_state);
        LOG(INFO) << "init_state =  " << init_state.pos.transpose();
      }
      cov_acc *= pow(common::G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
 
      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done: init_iter_num: %d; Gravity: %.4f %.4f %.4f %.4f; \
        \n state.bias_g: %.4f %.4f %.4f;  state.bias_a: %.4f %.4f %.4f; \
        \n acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f; \
        \n acc bias covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               init_iter_num, init_state.grav[0], init_state.grav[1], init_state.grav[2], mean_acc.norm(), init_state.bg[0], init_state.bg[1], init_state.bg[2], 
               init_state.ba[0], init_state.ba[1], init_state.ba[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2],
               cov_bias_acc[0], cov_bias_acc[1], cov_bias_acc[2], cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2]
              );
      // fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }
 
    return;
  }
  if (gnss_heading_need_init_){
      if (!meas.gnss.empty())
      {
          Eigen::Vector3d gnss_pose(meas.gnss.back()->pose.pose.position.x, meas.gnss.back()->pose.pose.position.y, 0.0);
          state_ikfom state = kf_state.get_x(); // 初始状态量
          common::V3D euler = Rotation::matrix2euler(state.rot.toRotationMatrix());
          Eigen::Vector3d tmp_vec(state.pos.x(), state.pos.y(), 0.0);
          Eigen::Vector3d tmp(init_state.pos.x(), init_state.pos.y(), 0.0);
          ROS_INFO_STREAM("gnss norm" << (gnss_pose - tmp).norm());
          if ( (gnss_pose - tmp).norm() > 3)
          { // gnss 水平位移大于5m
              auto GNSS_Heading = Eigen::Quaterniond::FromTwoVectors(gnss_pose- tmp, tmp_vec- tmp).toRotationMatrix();
              SO3 so3(GNSS_Heading);
              euler = SO3ToEuler(so3);
              ROS_WARN_STREAM("INITIAL gnss HEADING 1" << euler.transpose());
              euler = Rotation::matrix2euler(state.rot.toRotationMatrix());
              ROS_WARN_STREAM("INITIAL GNSS HEADING 2" << euler.transpose() * 180/M_PI);
              common::M3D rot = state.rot.toRotationMatrix() * GNSS_Heading.transpose();
              euler = Rotation::matrix2euler(rot);
              ROS_WARN_STREAM("INITIAL GNSS HEADING 3" << euler.transpose() * 180/M_PI);
              state.rot = Rotation::euler2matrix(euler);
              state.pos = Eigen::Vector3d(meas.gnss.back()->pose.pose.position.x, meas.gnss.back()->pose.pose.position.y, meas.gnss.back()->pose.pose.position.z);
              kf_state.change_x(state); 
              setGnssHeadingNeedInit(false);
              getRebuildIkdtreeFlag() = true;
          }
      }
  }
 
  UndistortPcl(meas, kf_state, *cur_pcl_un_);
 
  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
 
}  // namespace glio_mapping