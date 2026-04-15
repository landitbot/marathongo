#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H
 
#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"
#include "preprocess.h"
#include "module_base.h"
#include "rotation.h"
#include "estimator/glio_estimator.h"
 
/// *************Preconfiguration
namespace glio_mapping {
#define MAX_INI_COUNT (10)
 
inline bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};
 
/// *************IMU Process and undistortion
class ImuProcess : private ModuleBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
  ImuProcess(const std::string &config_path);
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void setGnssHeadingNeedInit(bool val) { gnss_heading_need_init_ = val; if(heading_init_callback_) heading_init_callback_(val);}
  bool& getRebuildIkdtreeFlag() { return rebuild_ikdtree_flag_; }
  void set_extrinsic(const common::V3D &transl, const common::M3D &rot);
  void set_extrinsic(const common::V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const common::V3D &scaler);
  void set_acc_cov(const common::V3D &scaler);
  void set_gyr_bias_cov(const common::V3D &b_g);
  void set_acc_bias_cov(const common::V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const common::MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);
  using HeadingInitCallback = std::function<void(bool)>;
  void SetHeadingInitCallback(HeadingInitCallback cb) { heading_init_callback_ = cb; }
  HeadingInitCallback heading_init_callback_;
  using GetVariableCallback = std::function<bool&(void)>;
  void GetOptWithGnssCallback(GetVariableCallback cb) { opt_with_gnss = cb; }
  void GetOptWithWheelCallback(GetVariableCallback cb) { opt_with_wheel = cb; }
  void GetOptWithAngleCallback(GetVariableCallback cb) { opt_with_angle = cb; }
  GetVariableCallback opt_with_gnss, opt_with_wheel, opt_with_angle;
  ofstream fout_imu;
  common::V3D cov_acc;
  common::V3D cov_gyr;
  common::V3D cov_acc_scale;
  common::V3D cov_gyr_scale;
  common::V3D cov_bias_gyr;
  common::V3D cov_bias_acc;
  double first_lidar_time;
  state_ikfom init_state;
  GlioEstimator::Ptr glio_estimaor_ptr_;
 private:
  void SetInitPose(common::V3D&, common::M3D&, common::V3D &);
  void IMU_init(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);
 
  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<common::Pose6D> IMUpose;
  vector<common::M3D>    v_rot_pcl_;
  common::M3D Lidar_R_wrt_IMU;
  common::V3D Lidar_T_wrt_IMU;
  common::V3D mean_acc;
  common::V3D mean_gyr;
  common::V3D angvel_last;
  common::V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1, static_rot_method_, initheading_;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true, initUseRtk_, gnss_heading_need_init_, gnss_front_, rebuild_ikdtree_flag_;
  bool wheel_front_, static_wheel_front_, virtual_wheel_front_, angle_front_;
};
 
}  // namespace glio_mapping
 
#endif // IMU_PROCESSING_H