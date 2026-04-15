/*
 * @Descripttion:
 * @Author: MengKai
 * @version:
 * @Date: 2023-06-08 23:59:37
 * @LastEditors: MengKai
 * @LastEditTime: 2023-06-10 01:21:13
 */
#pragma once
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>

#include <vector>
#include <sstream>

#include "colorful_terminal/colorful_terminal.hpp"

class ModuleBase {
    public:
    using Ptr = std::shared_ptr<ModuleBase>;
    private:
    YAML::Node config_node;
    std::string name;
    std::shared_ptr<ctl::table_out> table_out_ptr;

    public:
    std::vector<std::string> split(const std::string &s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }
    /**
     * @param config_path: 配置文件目录
     * @param prefix: 前缀
     * @param module_name: 模块名称
     */
    ModuleBase(const std::string &config_path, const std::string &prefix,
                const std::string &module_name = "default") {
        name = module_name;
        // colorful terminal print out
        table_out_ptr = std::make_shared<ctl::table_out>(module_name);
        if (config_path != "") {
            try {
                config_node = YAML::LoadFile(config_path);
            } catch (YAML::Exception &e) {
                std::cout << e.msg << std::endl;
            }

            if (prefix != "" && config_node[prefix]) config_node = config_node[prefix];
        }
    }
    /**
     * @param T
     * @param key: 键值
     * @param val: 读取数据到哪个参数
     * @param default_val: 默认值
     */
    template <typename T>
    void readParam(const std::string &key, T &val, T default_val) {
        if (key.find('/') != std::string::npos) { // 处理嵌套路径 
            YAML::Node current_node = YAML::Clone(config_node); 
            auto subkeys = split(key, '/');
            for (const auto &subkey : subkeys) {
                if (!current_node || !current_node[subkey]) {
                    // std::cerr << "[ERROR] Missing node: " << subkey 
                    //       << " in path: " << key << std::endl;
                    val = default_val;
                    // colorful terminal add var
                    table_out_ptr->add_item(key, VAR_NAME(val), val);
                    return;
                }
                current_node = current_node[subkey];
            }
            val = current_node.as<T>();
        } else{
          if (config_node[key]) {
              val = config_node[key].as<T>();
          } else {
              val = default_val;
          }
        }
        // colorful terminal add var
        table_out_ptr->add_item(key, VAR_NAME(val), val);
        // std::cout<<name: <<default_val<<std::endl;
    }
    void print_table() { table_out_ptr->make_table_and_out(); }
};