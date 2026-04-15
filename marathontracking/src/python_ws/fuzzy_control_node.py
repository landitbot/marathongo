import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from RLFuzzyTracking import *


class ControlNode():
    def __init__(self):
        self.controller = FuzzyTracking()
        self.puber_cmd_vel = rospy.Publisher(
            "/fuzzy_cmd_vel", Twist, latch=False, queue_size=1)

        self.suber_tracking_info = rospy.Subscriber(
            "/tracking_info", Float64MultiArray, self.handler_tracking_info,
            queue_size=1)

        self.suber_tracking_info = rospy.Subscriber(
            "/local_planner/control_info", Float64MultiArray, self.handler_control_info,
            queue_size=1)

        # self.timer = rospy.Timer(rospy.Duration(0.05), self.compute)

    def handler_tracking_info(self, msg: Float64MultiArray):
        target_x = msg.data[0]
        error_angle = msg.data[1]
        max_curvatrue = msg.data[2]
        max_curvatrue_back = msg.data[3]
        result = self.controller.compute(
            target_x, error_angle, max_curvatrue, max_curvatrue_back)
        self.pub_cmd_vel(result[0], result[1])

    def handler_control_info(self, msg: Float64MultiArray):
        target_x = msg.data[0]
        error_angle = msg.data[1]
        curvatrue = msg.data[2]
        result = self.controller.compute(
            target_x, error_angle, curvatrue)
        self.pub_cmd_vel(result[0] * 3.0, result[1])

    def pub_cmd_vel(self, vel_x, vel_z):
        msg = Twist()
        msg.linear.x = vel_x
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vel_z
        self.puber_cmd_vel.publish(msg)


def main():
    try:
        rospy.init_node("fuzzy_control_node")
        node = ControlNode()
        rospy.spin()
    except Exception as e:
        print(e)


main()
