#!/usr/bin/env python3
import csv
import math
import os

import rospy
from nav_msgs.msg import Odometry


class TwistRecorder:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry")
        self.out_csv = rospy.get_param("~out_csv", "")
        self.flush_every = max(1, int(rospy.get_param("~flush_every", 20)))

        if not self.out_csv:
            ts = int(rospy.Time.now().to_sec())
            self.out_csv = f"/tmp/odom_twist_{ts}.csv"

        out_dir = os.path.dirname(self.out_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        self.file = open(self.out_csv, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "stamp",
            "linear_x",
            "linear_y",
            "linear_speed",
            "angular_z",
        ])

        self.count = 0
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.cb, queue_size=100)
        rospy.on_shutdown(self.close)

        rospy.loginfo("Recording odom twist from %s -> %s", self.odom_topic, self.out_csv)

    def cb(self, msg):
        lin_x = msg.twist.twist.linear.x
        lin_y = msg.twist.twist.linear.y
        ang_z = msg.twist.twist.angular.z
        lin_speed = math.hypot(lin_x, lin_y)

        self.writer.writerow([
            f"{msg.header.stamp.to_sec():.6f}",
            f"{lin_x:.6f}",
            f"{lin_y:.6f}",
            f"{lin_speed:.6f}",
            f"{ang_z:.6f}",
        ])

        self.count += 1
        if self.count % self.flush_every == 0:
            self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.flush()
            self.file.close()
            rospy.loginfo("Saved %d twist samples to %s", self.count, self.out_csv)


if __name__ == "__main__":
    rospy.init_node("record_twist_from_odom")
    TwistRecorder()
    rospy.spin()
