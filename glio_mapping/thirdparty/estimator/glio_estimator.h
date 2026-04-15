#ifndef GLIO_ESTIMATOR_H 
#define GLIO_ESTIMATOR_H 
 
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <mutex>
#include "use-ikfom.hpp"
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include "module_base.h"
#include "so3_math.h"
#include "common_lib.h"
#include <glog/logging.h>
#include "rotation.h"

// 定义传感器数据结构 
struct IMUData {
    double timestamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
};

struct GNSSData {
    double timestamp;
    Eigen::Vector3d pos; // 经纬高或 xyz 
    Eigen::Vector3d std; // 标准差 
    bool isvalid = false;
};

struct OdomData {
    double timestamp;
    Eigen::Vector3d vel; // 纵向速度 
    bool isvalid = false;
};
struct AngleData {
    double timestamp;
    Eigen::Matrix3d angle; // 纵向速度 
    bool isvalid = false;
};
namespace glio_mapping {
class GlioEstimator : private ModuleBase {
public:
    using Ptr = std::shared_ptr<GlioEstimator>;
    // 使用 FastLIO 定义的状态和输入类型实例化 esekfom 
    using esekf = esekfom::esekf<state_ikfom, 12, input_ikfom>;
    using measurementModel_dyn_share = std::function<void(state_ikfom &, esekfom::dyn_share_datastruct<double> &)>;
    /// ESKF 观测类型
    enum class ObsType {
        LIDAR,                  // 开源版本只有Lidar
        WHEEL_SPEED,            // 单独的轮速观测
        WHEEL_SPEED_AND_LIDAR,  // 轮速+Lidar
        ACC_AS_GRAVITY,         // 重力作为加计观测量
        GPS,                    // GPS/RTk 六自由度位姿
        BIAS,
    };
    
    // 参数配置结构体 
    struct EstimatorParameters {
        double imu_rate = 200.0;
        double gnss_update_interval = 1.0;
        double odo_update_interval = 0.1;
        double zupt_angular_velocity_threshold = 0.1;
        double zupt_special_force_threshold = 0.4;
        double zupt_velocity_threshold = 0.06;
        
        // 噪声参数 
        double acc_noise = 0.1;
        double gyro_noise = 0.01;
        double gnss_pos_noise = 0.5;
        double nhc_noise = 0.2;
        
        // 开关 
        bool if_fuse_gnss = true;
        bool if_fuse_odo = true;
        bool if_fuse_angle = true;
        bool if_use_zupt = true;
        bool if_use_static_bias_update = false;
        bool if_use_angular_velocity_update = false;
    };
    GlioEstimator(const std::string &config_path);
    ~GlioEstimator() = default;
 
    // 数据输入接口 
    void addIMUData(const IMUData& imu);
    void addGNSSData(const GNSSData& gnss);
    void addOdomData(const OdomData& odom);
    void addAngleData(const AngleData& angle);
    void addAngleData(double timestamp, Eigen::Matrix3d& angle);
    void addIMUData(double timestamp, Eigen::Vector3d& acc, Eigen::Vector3d& gyro);
    void addIMUData(const sensor_msgs::Imu::ConstPtr& imu);
    void addGNSSData(const nav_msgs::Odometry::ConstPtr& gnss, const ObsType& type=ObsType::GPS);
    void addOdomData(const nav_msgs::Odometry::ConstPtr& odom, const ObsType& type=ObsType::WHEEL_SPEED);
    void addOdomData(double timestamp, Eigen::Vector3d& vel);
    // 初始化 
    void init(const state_ikfom& init_state, const Eigen::MatrixXd& init_cov);
 
    // 核心处理循环 (对应 WheelGINS::newImuProcess)
    void process(esekf &kf, Eigen::Matrix<double, 12, 12> &Q);
    void SetMeasurementModelCallback(measurementModel_dyn_share cb) { measurementModel_callback_ = cb; }
    measurementModel_dyn_share measurementModel_callback_;
    inline double timestamp() const { return imucur_.timestamp; }
    // 获取状态 
    state_ikfom getState();
    Eigen::MatrixXd getCovariance();
    void h_share_model_gnss(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    void h_share_model_wheel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    void h_share_model_angle(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    common::V3D& gnssObsR() { return gnss_obs_R_; }
    common::V3D& wheelObsR() { return wheel_obs_R_; }
    common::V3D& angleObsR() { return angle_obs_R_; }
    bool& useGpsElevation() { return useGpsElevation_; }
    using GetVariableCallback = std::function<bool&(void)>;
    void GetOptWithGnssCallback(GetVariableCallback cb) { opt_with_gnss = cb; }
    void GetOptWithWheelCallback(GetVariableCallback cb) { opt_with_wheel = cb; }
    void GetOptWithAngleCallback(GetVariableCallback cb) { opt_with_angle = cb; }
    GetVariableCallback opt_with_gnss, opt_with_wheel, opt_with_angle;
private:
    // 时间对齐判断逻辑 (对应 WheelGINS::isToUpdate)
    // 返回: 0-无需更新, 1-靠近上一时刻, 2-靠近当前时刻, 3-在中间 
    int isToUpdate(double imutime1, double imutime2, double updatetime) const;
 
    // IMU 插值 (对应 WheelGINS::imuInterpolate)
    void imuInterpolate(const IMUData& imu1, IMUData& imu2, double timestamp, IMUData& midimu);
 
    // IMU 预测 (对应 WheelGINS::insPropagation)
    void predict(esekf &kf, const IMUData& imupre, const IMUData& imucur, Eigen::Matrix<double, 12, 12> &Q);
 
    // 传感器更新逻辑 
    void gnssUpdate(esekf &kf);      // 对应 GNSSUpdate 
    void odoNHCUpdate(esekf &kf);    // 对应 odo_nhcUpdate 
    void angleUpdate(esekf &kf);
    // 状态反馈 (esekfom 内部已包含，此处可扩展外参反馈)
    void stateFeedback();
 
    // 辅助函数 
    void checkCov();
    inline void next(void) {
      imupre_ = imucur_;
    }
    void detectZUPT(Eigen::Vector3d& vel);
    void ZUPT(esekf &kf);
    void odoUpdate(esekf &kf);
 
private:
    EstimatorParameters params_;
 
    // 数据缓存 
    std::deque<IMUData> imu_buffer_;
    GNSSData current_gnss_;
    OdomData current_odo_;
    AngleData current_angle_;
    // 当前和上一时刻 IMU 
    IMUData imupre_;
    IMUData imucur_;
    
    // 时间戳记录 
    double last_gnss_update_t_;
    double last_odo_update_t_;
    double last_angle_update_t_;
 
    // 状态维度 (FastLIO 中通常固定，但保留接口以备扩展)
    int state_dim_ = 23; // dim of state_ikfom 
    bool useGpsElevation_;
    common::V3D gnss_obs_R_;
    common::V3D wheel_obs_R_;
    common::V3D angle_obs_R_;
    bool if_ZUPT_available_ = false;
    bool if_ZIHR_available_ = false;
    int ZIHR_num_ = 0;
    double zihr_preheading_ = 0.0;
};
}
#endif // GLIO_ESTIMATOR_H 