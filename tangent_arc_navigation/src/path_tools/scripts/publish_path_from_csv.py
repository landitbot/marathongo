#!/usr/bin/env python3
import csv
import math
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion

def yaw_to_quat(yaw):
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    return q

def load_points(csv_file):
    pts = []
    with open(csv_file, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append((float(row["x"]), float(row["y"]), float(row.get("z", 0.0))))
    return pts

def build_path(points, frame_id):
    path = Path()
    path.header.frame_id = frame_id
    n = len(points)
    for i, (x, y, z) in enumerate(points):
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z

        if i + 1 < n:
            nx, ny, _ = points[i + 1]
            yaw = math.atan2(ny - y, nx - x)
        elif i > 0:
            px, py, _ = points[i - 1]
            yaw = math.atan2(y - py, x - px)
        else:
            yaw = 0.0
        ps.pose.orientation = yaw_to_quat(yaw)
        path.poses.append(ps)
    return path

if __name__ == "__main__":
    rospy.init_node("publish_path_from_csv")
    csv_file = rospy.get_param("~csv_file")
    frame_id = rospy.get_param("~frame_id", "map")
    topic = rospy.get_param("~path_topic", "/central/smoothed_path")
    rate_hz = rospy.get_param("~rate", 2.0)

    points = load_points(csv_file)
    path = build_path(points, frame_id)
    pub = rospy.Publisher(topic, Path, queue_size=1, latch=True)

    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        path.header.stamp = now
        for p in path.poses:
            p.header.stamp = now
        pub.publish(path)
        rate.sleep()