# tangent_arc_navigation

#### 介绍
**马拉松项目导航算法：已有全局路径下，使用激光雷达进行路径追踪和避障。自写避障算法，区分于MPC和模糊控制方案**

#### 前置条件

1. 需要局部去畸变点云话题:/cuerrent_scan_body
2. 世界坐标系下的定位话题:/odometry
3. 全局路径
4. 机器人控制话题
5. 动态tf发布

#### 安装教程

1.  配置环境

```
./setup_workspace_env.sh
```

2.  编译

```
catkin_make && source devel/setup.bash
```

#### 使用说明

1.  启动odom转发和tf发布，以及rviz可视化

```
roslaunch sim_interface sim_navigation.launch
```

2.  启动点云地面点滤除、scan转换(需要自写，简单实现思路为：对一定高度范围内点云进行裁剪并转换为scan，发布到/scan话题里面)

```
roslaunch obstacle_detector obstacle_detector.launch
```

3.  启动costmap、costmap_converter、规划节点、控制节点


```
roslaunch vo_navigation vo_navigation.launch
```

> ***注意:*** 这里使用的并不是vo算法，自写算法但是没有更改名字
