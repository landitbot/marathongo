import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist, TwistStamped
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from Joy import *
import serial
import time
import threading
import numpy as np
import math

import signal
import sys


def clamp(x, min_, max_):
    if x < min_:
        return min_
    if x > max_:
        return max_
    return x


class ROSJoy(AbstractJOY):
    def __init__(self):
        super().__init__()
        rospy.Subscriber("/joy", Joy, self.handler_joy)

    def handler_joy(self, msg: Joy):
        LX = -msg.axes[0]
        LY = msg.axes[1]
        RX = msg.axes[3]
        RY = msg.axes[4]
        A = msg.buttons[0]
        B = msg.buttons[1]
        X = msg.buttons[2]
        Y = msg.buttons[3]
        LB = msg.buttons[4]  # function
        RB = msg.buttons[5]  # control
        BACK = msg.buttons[6]
        START = msg.buttons[7]
        self.state.vel_forward = clamp(LY, -1, 1)
        self.state.vel_rotation = clamp(RX, -1, 1)
        self.state.btn_init_receiver = LB > 0 and START > 0
        self.state.btn_poweroff = LB > 0 and RB > 0 and X > 0
        self.state.btn_manual_control = RB > 0 and X > 0
        self.state.btn_auto_control = RB > 0 and A > 0
        # self.state.btn_third_control = RB > 0 and B > 0
        self.state.btn_third_control = False
        self.updated()


class SBUSJoy(AbstractJOY):
    def __init__(self, com):
        super().__init__()
        self.receiver = SBUSReceiver()
        self.serial = serial.Serial(com, 100000, parity='E', stopbits=2)
        self.running = True
        self.thread = threading.Thread(target=self.read_serial)
        self.thread.start()
        self.cali = [
            [174, 1811],
            [174, 1811],
            [174, 1811],
            [174, 1811],
            [191, 1792],  # A
            [191, 997, 1792],  # B
            [191, 997, 1792],  # C
            [191, 1792],  # D
            [191, 997, 1792],  # E
            [191, 997, 1792],  # F
        ]

    def stop(self):
        self.running = False
        self.thread.join()

    def read_serial(self):
        while self.running:
            byte = self.serial.read(1)
            if self.receiver.process_byte(byte):
                self.handler_data()

    def handler_data(self):
        chan = self.receiver.parser.get_channels()
        # print(chan[0], chan[1], chan[2], chan[3])
        Ail = ((chan[0] - self.cali[0][0]) /
               (self.cali[0][1] - self.cali[0][0]) - 0.5) * 2
        Ele = ((chan[1] - self.cali[1][0]) /
               (self.cali[1][1] - self.cali[1][0]) - 0.5) * 2
        Thr = ((chan[2] - self.cali[2][0]) /
               (self.cali[2][1] - self.cali[2][0]) - 0.5) * 2
        Rud = ((chan[3] - self.cali[3][0]) /
               (self.cali[3][1] - self.cali[3][0]) - 0.5) * 2

        if abs(Ail) < 0.1:
            Ail = 0
        if abs(Ele) < 0.1:
            Ele = 0
        if abs(Thr) < 0.1:
            Thr = 0
        if abs(Rud) < 0.1:
            Rud = 0
        SA = -100 if chan[4] <= 500 else 100
        SB = -100 if chan[5] <= 500 else 0 if chan[5] <= 1200 else 100
        SC = -100 if chan[6] <= 500 else 0 if chan[6] <= 1200 else 100
        SD = -100 if chan[7] <= 500 else 100
        SE = -100 if chan[8] <= 500 else 0 if chan[8] <= 1200 else 100
        SF = -100 if chan[9] <= 500 else 0 if chan[9] <= 1200 else 100

        self.state.vel_forward = clamp(Thr, 0, 1)
        self.state.vel_rotation = clamp(-Ail, -1, 1) * 2.0
        self.state.btn_init_receiver = SA > 0 and SD > 0 and Thr < 0
        self.state.btn_poweroff = Thr < -0.95 and Rud < - \
            0.95 and Ele < -0.95 and Ail > 0.95
        # self.state.btn_manual_control = SA > 0 and SB == -100 and SC == -100
        # self.state.btn_auto_control = SA > 0 and SB == -100 and SC == 100
        # self.state.btn_third_control = SA > 0 and SB == -100 and SC == 0
        self.state.btn_manual_control = SC == -100
        self.state.btn_auto_control = SC == 100
        self.state.btn_third_control = SC == 0

        # print(Thr, Rud, Ele, Ail)
        # print(Ail, Ele, Thr, Rud)
        # print(SA, SB, SC, SD, SE, SF)
        self.updated()


class Node():
    def __init__(self):
        self._puber_cmd_vel = rospy.Publisher(
            "/final_stampd_cmd_vel", TwistStamped, latch=False, queue_size=1)

        self.suber_fuzzy_cmd_vel = rospy.Subscriber(
            "/fuzzy_cmd_vel", Twist, self.handler_fuzzy_cmd_vel, queue_size=1)

        self.cmd_vel_feed_count = 0
        self.timer_watch_dog = rospy.Timer(
            rospy.Duration(0.1), self.handler_timer_watch_dog)

        self.process_timer = rospy.Timer(
            rospy.Duration(0.02), self.process_state)

        self.state = None  # type: JoyState

        # self.joy_interface = ROSJoy()
        self.joy_interface = SBUSJoy("/dev/tty_elrs")
        self.joy_interface.setCallback(self.on_joy)

        self.control_mode = 'manual'
        self.fuzzy_cmd_vel = None
        self.last_linear_x_vel = 0

    def handler_timer_watch_dog(self, event):
        if self.fuzzy_cmd_vel is not None:
            self.cmd_vel_feed_count = self.cmd_vel_feed_count + 1
            if self.cmd_vel_feed_count > 5:
                self.cmd_vel_feed_count = 9999
                self.fuzzy_cmd_vel.linear.x = 0
                self.fuzzy_cmd_vel.linear.y = 0
                self.fuzzy_cmd_vel.linear.z = 0
                self.fuzzy_cmd_vel.angular.x = 0
                self.fuzzy_cmd_vel.angular.y = 0
                self.fuzzy_cmd_vel.angular.z = 0

    def stop(self):
        self.joy_interface.stop()

    def on_joy(self, state):
        self.state = state

    def handler_fuzzy_cmd_vel(self, msg: Twist):
        self.cmd_vel_feed_count = 0
        msg.linear.x = msg.linear.x / 1.0
        msg.linear.x = clamp(msg.linear.x, -0.5, 1.0)
        msg.angular.z = clamp(msg.angular.z, -3.0, 3.0)
        self.fuzzy_cmd_vel = msg

    def process_state(self, event):
        if self.state is None:
            return

        state = self.state
        if state.btn_manual_control and self.control_mode != 'manual':
            self.control_mode = 'manual'
            state.vel_forward = 0
            state.vel_rotation = 0
            self.call_rc1_mode()
            print("SetControlMode: manual")
            self.smooth_stop()
        elif state.btn_auto_control and self.control_mode != 'auto':
            self.control_mode = 'auto'
            self.call_rc1_mode()
            print("SetControlMode: auto")
        elif state.btn_third_control and self.control_mode != 'third':
            self.control_mode = 'third'
            print("SetControlMode: third")
            self.call_rc0_mode()
            # self.pub_cmd_vel(0, 0)

        if state.btn_init_receiver:
            print("btn_init_receiver")
            self.call_receiver_init()
            self.call_rc1_mode()
            time.sleep(1)

        if state.btn_poweroff:
            print("btn_poweroff")
            # self.call_poweroff()
            # time.sleep(1)

        if self.control_mode == 'manual':
            self.pub_cmd_vel(state.vel_forward, state.vel_rotation)
        if self.control_mode == 'auto':
            if self.fuzzy_cmd_vel is None:
                self.pub_cmd_vel(0, 0)
            else:
                self.pub_cmd_vel(self.fuzzy_cmd_vel.linear.x,
                                 self.fuzzy_cmd_vel.angular.z)

    def call_receiver_init(self):
        try:
            rospy.wait_for_service("/cmd_server/init_udp_receiver", timeout=2)
            client = rospy.ServiceProxy(
                "/cmd_server/init_udp_receiver", Trigger)
            res = client(TriggerRequest())  # type: TriggerResponse
            print(res.message)
        except rospy.ROSException as e:
            print("Cannot find /cmd_server/init_udp_receiver")
    
    def call_rc0_mode(self):
        try:
            rospy.wait_for_service("/cmd_server/rc0_mode", timeout=2)
            client = rospy.ServiceProxy(
                "/cmd_server/rc0_mode", Trigger)
            res = client(TriggerRequest())  # type: TriggerResponse
            print(res.message)
        except rospy.ROSException as e:
            print("Cannot find /cmd_server/rc0_mode")
    
    def call_rc1_mode(self):
        try:
            rospy.wait_for_service("/cmd_server/rc1_mode", timeout=2)
            client = rospy.ServiceProxy(
                "/cmd_server/rc1_mode", Trigger)
            res = client(TriggerRequest())  # type: TriggerResponse
            print(res.message)
        except rospy.ROSException as e:
            print("Cannot find /cmd_server/rc1_mode")

    def call_poweroff(self):
        try:
            rospy.wait_for_service("/cmd_server/poweroff", timeout=2)
            client = rospy.ServiceProxy("/cmd_server/poweroff", Trigger)
            res = client(TriggerRequest())  # type: TriggerResponse
            print(res.message)
        except rospy.ROSException as e:
            print("Cannot find /cmd_server/poweroff")

    def pub_cmd_vel(self, vel_x, vel_z):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = vel_x
        msg.twist.linear.y = 0
        msg.twist.linear.z = 0
        msg.twist.angular.x = 0
        msg.twist.angular.y = 0
        msg.twist.angular.z = vel_z
        self.last_linear_x_vel = vel_x
        self._puber_cmd_vel.publish(msg)

    def smooth_stop(self, interval=0.05, stop_duration=1.0):
        if self.last_linear_x_vel > 0.5:
            for i in np.linspace(2.0, 0.0, int(stop_duration / interval)):
                vel = math.tanh(i) * self.last_linear_x_vel
                self.pub_cmd_vel(vel, 0.0)
                print(vel)
                time.sleep(interval)
        else:
            self.pub_cmd_vel(0.0, 0.0)


def shutdown(a, b):
    rospy.signal_shutdown("normal")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, shutdown)
    try:
        rospy.init_node("py_joy_node")
        node = Node()
        print("Started!")
        rospy.spin()
    except KeyboardInterrupt:
        print("Exiting...")

        node.stop()
    except Exception as e:
        print(e)


main()
