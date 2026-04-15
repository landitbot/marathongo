#!/bin/bash

# tools

source /opt/ros/noetic/setup.bash
cd /path/to/workspace

check_port(){
    if lsof -i :$1 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

wait_port(){
    while true; do
        if check_port $1; then
            echo "Found Port: $1"
            break
        else
            echo "Waiting for port: $1 ..."
            sleep 3
        fi
    done
}

wait_roscore(){
    while true; do
        rosnode list > /dev/null 2>&1 && break
        sleep 1
    done
}

# main

start_setup(){
    cp ./99-mara.rules /etc/udev/rules.d
    udevadm control --reload-rules
    udevadm trigger
}


start_roscore(){
    bash -c "
        source /opt/ros/noetic/setup.bash
        roscore
    " &
}

start_joy(){
    bash -c "
        source src/python_ws/venv/bin/activate
        python3 src/python_ws/joy_node.py
    " &
}

start_send_node(){
    bash -c "
        source devel/setup.bash
        rosrun ros_bridge_sender send_node
    " &
}

start_path_process(){
    bash -c "
        source devel/setup.bash
        rosrun path_process path_process_node
    " &
}

start_fuzzy_control(){
    bash -c "
        source src/python_ws/venv/bin/activate
        python3 src/python_ws/fuzzy_control_node.py
    " &
}

start_setup || echo "failed to start_setup"
start_roscore || echo "failed to start_roscore"

wait_roscore && echo "roscore ready"

start_joy || echo "failed to start_joy"
start_send_node || echo "failed to start_send_node"
#start_path_process || echo "failed to start_path_process"
#start_fuzzy_control || echo "failed to start_fuzzy_control"

while true; do
    sleep 1
done
