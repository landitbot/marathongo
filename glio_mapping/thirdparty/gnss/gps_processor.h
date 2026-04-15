#pragma once
#include "gps_util.h"
#include <cmath>
#include <string>
#include <iomanip>
#include <iostream>
#include "common_lib.h"
#include "module_base.h"
namespace glio_mapping {
class GPSProcessor  : private ModuleBase {
private:
    bool gnss_inited = false;
    
    Eigen::Affine3d Gnss_T_IMU_;
    std::vector<double> extrinT_gnss_{3, 0.0};  // gnss-imu translation
    std::vector<double> extrinR_gnss_{9, 0.0};  // gnss-imu rotation
    std::vector<double> initXYZ_;
public:
    using Ptr = std::shared_ptr<GPSProcessor>;
    GPSProcessor(std::string config_path): ModuleBase(config_path, "gps_processing", "GPSProcessor") {
      common::V3D Gnss_T_wrt_IMU = common::V3D::Zero();  // 平移向量 
      common::M3D Gnss_R_wrt_IMU = common::M3D::Identity();  // 旋转矩阵 
      readParam("extrinsic_T_gnss", extrinT_gnss_, std::vector<double>({0.07, 0.27, 0.12}));
      readParam("extrinsic_R_gnss", extrinR_gnss_, std::vector<double>({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
      readParam("initXYZ", initXYZ_, std::vector<double>({463619.839000113075599,4404452.67, 29.148}));
      
      Gnss_T_wrt_IMU = common::VecFromArray<double>(extrinT_gnss_);
      Gnss_R_wrt_IMU = common::MatFromArray<double>(extrinR_gnss_);
      SetExtrinsic(Gnss_T_wrt_IMU, Gnss_R_wrt_IMU);

      if(!initXYZ_.empty() && initXYZ_[0] != 0){
        SetOriginXYZ(initXYZ_[0], initXYZ_[1], initXYZ_[2]);
        gnss_inited = true;
      }
      print_table();
    }
    
    double time = 0.0;
    double local_x = 0.0;
    double local_y = 0.0;
    double local_z = 0.0;
    
    double origin_x = 0.0;
    double origin_y = 0.0;
    double origin_z = 0.0;

    
 
    // 初始化原点坐标（经纬度 -> 平面坐标）
    void InitOrigin(double lat, double lon, double alt) {
        GPSUtil::ConvertLonLatToXY(lon, lat, origin_x, origin_y);
        origin_z = alt;
        std::cout << "Origin set to: " 
                  << std::fixed << std::setprecision(6) 
                  << origin_x << ", " << origin_y << ", " << origin_z << std::endl;
    }
 
    // 更新当前点的平面坐标（相对原点）
    void UpdatePosition(double lat, double lon, double alt) {
        double x, y;
        GPSUtil::ConvertLonLatToXY(lon, lat, x, y);
        common::V3D gnss_pos = Gnss_T_IMU_ * common::V3D(x - origin_x, y - origin_y, alt - origin_z);
        local_x = gnss_pos.x();
        local_y = gnss_pos.y();
        local_z = gnss_pos.z();

    }
    
    // 直接设置原点平面坐标（测试用）
    void SetOriginXYZ(double x, double y, double z) {
        origin_x = x;
        origin_y = y;
        origin_z = z;
    }
    bool& getGnssInitStatus() { 
        return gnss_inited;  // 返回左值引用 
    }
    void SetExtrinsic(const common::V3D &transl, const common::M3D &rot) {
        Gnss_T_IMU_.translation() = transl;
        Gnss_T_IMU_.linear() = rot;
    }
    void SetExtrinsic(const common::V3D &transl) {
        SetExtrinsic(transl, common::M3D::Identity());
    }
    common::V3D GetLocalPosition() const {
        return common::V3D(local_x, local_y, local_z);
    }
};
}  // namespace glio_mapping