/*
 * @Description: 读写文件管理
 * @Author: Ren Qian
 * @Date: 2020-02-24 19:22:53
 */
#ifndef LIDAR_LOCALIZATION_TOOLS_FILE_MANAGER_HPP_
#define LIDAR_LOCALIZATION_TOOLS_FILE_MANAGER_HPP_

#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
#include <boost/filesystem.hpp>
#include "glog/logging.h"

class FileManager{
  public:
    static bool CreateFile(std::ofstream& ofs, std::string file_path) {
        ofs.close();
        boost::filesystem::remove(file_path.c_str());

        ofs.open(file_path.c_str(), std::ios::out);
        if (!ofs) {
            LOG(WARNING) << "无法生成文件: " << std::endl << file_path << std::endl << std::endl;
            return false;
        }

        return true;
    }

    static bool InitDirectory(std::string directory_path, std::string use_for) {
        if (boost::filesystem::is_directory(directory_path)) {
            boost::filesystem::remove_all(directory_path);
        }

        return CreateDirectory(directory_path, use_for);
    }

    static bool CreateDirectory(std::string directory_path, std::string use_for) {
        if (!boost::filesystem::is_directory(directory_path)) {
            boost::filesystem::create_directory(directory_path);
        }

        if (!boost::filesystem::is_directory(directory_path)) {
            LOG(WARNING) << "CANNOT create directory " << std::endl << directory_path << std::endl << std::endl;
            return false;
        }

        std::cout << use_for << " output path:" << std::endl << directory_path << std::endl << std::endl;
        return true;
    }

    static bool CreateDirectory(std::string directory_path) {
        if (!boost::filesystem::is_directory(directory_path)) {
            boost::filesystem::create_directory(directory_path);
        }

        if (!boost::filesystem::is_directory(directory_path)) {
            LOG(WARNING) << "CANNOT create directory " << std::endl << directory_path << std::endl << std::endl;
            return false;
        }

        return true;
    }
};
class Logger {
public:
    explicit Logger(const std::string& file) {
        std::cerr << "Logger" << std::endl;
        if (!FileManager::CreateFile(ofs, file)) {
            std::cerr << "12312"<< std::endl;
            throw std::runtime_error("Log file creation failed");
        }
    }
    bool SavePose(double current_time, const Eigen::Matrix4d &pose) {
        if (init_start_time < 0) {
            init_start_time = current_time;  // 第一次调用时初始化
        }
        double timestamp = current_time - init_start_time;
        const Eigen::Vector3d translation = pose.block<3,1>(0,3); 
        
        const Eigen::Quaterniond quat(pose.block<3,3>(0,0)); 
        ofs << std::fixed << std::setprecision(15) 
            << timestamp << " "
            << translation.x() << " "
            << translation.y() << " "
            << translation.z() << " "
            << quat.x() << " "
            << quat.y() << " "
            << quat.z() << " "
            << quat.w() << std::endl;
        return ofs.good(); 
    }
    bool SavePose(double current_time, const Eigen::Vector3d translation,const Eigen::Quaterniond quat) {
        if (init_start_time < 0) {
            init_start_time = current_time;  // 第一次调用时初始化
        }
        double timestamp = current_time - init_start_time;
        ofs << std::fixed << std::setprecision(15) 
            << timestamp << " "
            << translation.x() << " "
            << translation.y() << " "
            << translation.z() << " "
            << quat.x() << " "
            << quat.y() << " "
            << quat.z() << " "
            << quat.w() << std::endl;
        return ofs.good(); 
    }
    bool SavePose(double current_time, const Eigen::Affine3d pose) {
        if (init_start_time < 0) {
            init_start_time = current_time;  // 第一次调用时初始化
        }
        double timestamp = current_time - init_start_time;
        Eigen::Quaterniond quat = Eigen::Quaterniond(pose.rotation());
        ofs << std::fixed << std::setprecision(15) 
            << timestamp << " "
            << pose.translation().x() << " "
            << pose.translation().y() << " "
            << pose.translation().z() << " "
            << quat.x() << " "
            << quat.y() << " "
            << quat.z() << " "
            << quat.w() << std::endl;
        return ofs.good(); 
    }
private:
    double init_start_time = -1;
    std::ofstream ofs;
};
#endif
