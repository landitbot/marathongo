#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math


def compute_offsets(points, offset):
    # points: list of (x,y)
    n = len(points)
    if n == 0:
        return [], []
    left = []
    right = []
    for i in range(n):
        if n == 1:
            # single point: no tangent, choose arbitrary normal
            tx, ty = 1.0, 0.0
        else:
            if i == 0:
                x0, y0 = points[0]
                x1, y1 = points[1]
                tx, ty = x1 - x0, y1 - y0
            elif i == n - 1:
                x0, y0 = points[n - 2]
                x1, y1 = points[n - 1]
                tx, ty = x1 - x0, y1 - y0
            else:
                x_prev, y_prev = points[i - 1]
                x_next, y_next = points[i + 1]
                tx, ty = x_next - x_prev, y_next - y_prev
        # normalize tangent
        norm = math.hypot(tx, ty)
        if norm < 1e-6:
            nx, ny = 0.0, 0.0
        else:
            tx /= norm
            ty /= norm
            # left normal: (-ty, tx)
            nx = -ty
            ny = tx
        px, py = points[i]
        lx = px + nx * offset
        ly = py + ny * offset
        rx = px - nx * offset
        ry = py - ny * offset
        left.append((lx, ly))
        right.append((rx, ry))
    return left, right


class OffsetNode(object):
    def __init__(self):
        rospy.init_node('offset_paths_node')
        self.offset = rospy.get_param('~offset_distance', 3.5)
        self.frame_id = rospy.get_param('~frame_id', '')
        self.left_pub = rospy.Publisher('/left/smoothed_path', Path, queue_size=1)
        self.right_pub = rospy.Publisher('/right/smoothed_path', Path, queue_size=1)
        self.sub = rospy.Subscriber('/central/smoothed_path', Path, self.cb, queue_size=1)
        rospy.loginfo('offset_paths_node started, offset=%.3f', self.offset)

    def cb(self, msg):
        pts = []
        for p in msg.poses:
            pts.append((p.pose.position.x, p.pose.position.y))
        left_pts, right_pts = compute_offsets(pts, self.offset)

        left_path = Path()
        right_path = Path()
        # preserve header/frame
        left_path.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.frame_id
        right_path.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.frame_id
        left_path.header.stamp = rospy.Time.now()
        right_path.header.stamp = rospy.Time.now()

        for i, (x, y) in enumerate(left_pts):
            pose = PoseStamped()
            pose.header.frame_id = left_path.header.frame_id
            pose.header.stamp = left_path.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            left_path.poses.append(pose)
        for i, (x, y) in enumerate(right_pts):
            pose = PoseStamped()
            pose.header.frame_id = right_path.header.frame_id
            pose.header.stamp = right_path.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            right_path.poses.append(pose)

        self.left_pub.publish(left_path)
        self.right_pub.publish(right_path)


if __name__ == '__main__':
    try:
        node = OffsetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
