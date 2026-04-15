# Marathon Tracking
```txt
 _______                    ______ _______ _____  
|   |   |.---.-.----.---.-.|   __ \_     _|     \ 
|       ||  _  |   _|  _  ||    __/_|   |_|  --  |
|__|_|__||___._|__| |___._||___|  |_______|_____/ 
                                                  
```

## 🌟特性
追踪分为三个阶段：全速寻线，全速避障，低速恢复
- 全速寻线：当无障碍物时，机器人会使用类似Bang-Bang控制的方式进行寻线
- 全速避障：当障碍物距机器人>1.5米时候，机器人会进行高速避障，安全范围较大
- 低速恢复：当障碍物距机器人<1.5米时候，机器人会紧急停下，然后使用A*规划出一条轨迹，绕开复杂障碍物，回到原始轨迹上。当机器人稳定后再继续进入全速寻线。

## ✨系统组件
- 地图数据结构：环形hash体素地图
- 障碍物检测：基于相对高度的地面去除（障碍物高度需要＞0.4米）
- 路径规划1：基于3次样条采样的路径规划
- 路径规划2：基于A*的路径规划
- 控制器：PID控制器（仅控制角速度）
- 运动学包络器：用于实现多变量控制，让控制量更加符合机器人可响应范围
- 集成NoVnc远程图形化界面（仅用于调试）

## ⏫系统输入
| 话题 | 介绍 |
|-----------------------------|----------------------------------|
|  /current_scan_body         |  经过去畸变后的body系下的点云数据  |
|  /odometry                  |  全局里程计                       |
|  /central/smoothed_path     |  待跟踪轨迹                       |
|  /left/smoothed_path        |  左边线(障碍物)                    |
|  /right/smoothed_path       |  右边线(障碍物)                    |

## 🕹️启动流程
开机后相关服务将会通过marago.service启动  
marago.service服务：启动/opt/marathon_ws/marago.sh  
marago启动内容：  
- 启动roscore
- python3 python_ws/joy_node.py
- rosrun ros1_sender_general sender_node
- rosrun local_planner local_planner

## 🎮控制指令话题逻辑
|     话题   |   解释   |
|--------------------------------|---------|
|/final_stampd_cmd_vel           | 最终带时间戳的控制，将会被sender_node转发给机器人  |
|/fuzzy_cmd_vel                  | 自动控制器发布的速度，将通过joy_node.py进行路由    |
| /joy                           |  常规手柄遥控器的话题                             |
| 设备：/dev/tty_elrs             |  航模遥控器的串口设备                            |

也就是说，自动控制将控制信息发布到/fuzzy_cmd_vel话题即可

## 📇目录解释

### python_ws
作用：这是一个由Python实现的一系列工具
其中包含：
- 遥控器控制(支持ROS Joy和SBUS协议遥控器)
- 模糊控制器（使用PID代替）

### ros1_sender_general
作用：就是一个普通的sender  
输入：/final_stampd_cmd_vel  
输出：发送控制量给机器人  

### local_planner
作用：继承了路径规划，避障，控制一体  
输入：参考上述表格  
输出：最终控制量  

## ⚒️依赖安装 
```txt
sudo apt install libpcl-dev
sudo apt install libpcl-dev
```

python环境安装
```txt
cd /opt/marathon_ws/src/python_ws
./create_venv.sh
source venv/bin/activate
pip3 install -r ./requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```