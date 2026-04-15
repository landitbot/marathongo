/*
 * @Author: ylh 
 * @Date: 2024-04-27 22:07:15 
 * @Last Modified by: ylh 2252512364@qq.com
 * @Last Modified time: 2024-04-27 22:22:13
 */

#ifndef TIMESTAMP_H
#define TIMESTAMP_H
#include <iostream>
#include <chrono>

class GetCurrentTime {
    public:
        // 获取当前时间戳（秒）1
        static int64_t nows() {
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
            return duration.count();
        }

        // 获取当前时间戳（毫秒）1e-3
        static int64_t nowMs() {
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
            return duration.count();
        }

        // 获取当前时间戳（微秒）1e-6
        static int64_t nowUs() {
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
            return duration.count();
        }

        // 获取当前时间戳（纳秒）1e-9
        static int64_t nowNs() {
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
            return duration.count();
        }
};
    
#endif