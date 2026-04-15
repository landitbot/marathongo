#!/usr/bin/env python3
import csv
import math
import rospy
from nav_msgs.msg import Odometry

class Recorder:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry")
        self.out_csv = rospy.get_param("~out_csv", "")
        self.min_dist = rospy.get_param("~min_point_distance", 0.2)
        self.max_points = rospy.get_param("~max_points", 20000)

        if not self.out_csv:
            ts = rospy.Time.now().to_sec()
            self.out_csv = f"/tmp/recorded_path_{int(ts)}.csv"

        self.points = []
        self.last_xy = None
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.cb, queue_size=50)

        rospy.on_shutdown(self.save)
        rospy.loginfo("Recording from %s -> %s", self.odom_topic, self.out_csv)

    def cb(self, msg):
        p = msg.pose.pose.position
        x, y, z = p.x, p.y, p.z
        if self.last_xy is not None:
            dx = x - self.last_xy[0]
            dy = y - self.last_xy[1]
            if math.hypot(dx, dy) < self.min_dist:
                return
        self.points.append((x, y, z))
        self.last_xy = (x, y)
        if len(self.points) >= self.max_points:
            rospy.signal_shutdown("Reached max_points")

    def save(self):
        if not self.points:
            rospy.logwarn("No points recorded.")
            return
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "z"])
            w.writerows(self.points)
        rospy.loginfo("Saved %d points to %s", len(self.points), self.out_csv)

if __name__ == "__main__":
    rospy.init_node("record_path_from_odom")
    Recorder()
    rospy.spin()