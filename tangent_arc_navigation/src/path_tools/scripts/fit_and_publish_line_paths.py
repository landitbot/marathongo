#!/usr/bin/env python3
"""
Fit a straight line to recorded path and publish central + left/right offset paths.
Central path: direct line fit from recorded.csv
Left/right paths: offset by ±3.5m perpendicular to the line
"""

import rospy
import csv
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math

def fit_line_to_path(points):
    """
    Fit a 2D line (ax + by + c = 0) to points using least squares.
    Returns: (a, b, c) coefficients and a direction vector along the line.
    """
    x_data = points[:, 0]
    y_data = points[:, 1]
    
    # Center data
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    x_c = x_data - x_mean
    y_c = y_data - y_mean
    
    # PCA: compute covariance and find principal component
    cov = np.cov(x_c, y_c)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # Direction along the line (eigenvector with larger eigenvalue)
    principal_idx = np.argmax(eigvals)
    direction = eigvecs[:, principal_idx]
    
    # Normal to the line (perpendicular)
    normal = np.array([-direction[1], direction[0]])
    
    # Line equation: normal · (p - center) = 0
    # => normal[0] * (x - x_mean) + normal[1] * (y - y_mean) = 0
    # => normal[0] * x + normal[1] * y - (normal[0] * x_mean + normal[1] * y_mean) = 0
    a = normal[0]
    b = normal[1]
    c = -(a * x_mean + b * y_mean)
    
    # Normalize so that a^2 + b^2 = 1 for easier offset calculation
    norm = np.sqrt(a**2 + b**2)
    a, b, c = a/norm, b/norm, c/norm
    
    return a, b, c, direction

def offset_point_perpendicular(point, a, b, offset):
    """
    Offset a point perpendicular to line ax + by + c = 0 by 'offset' distance.
    Offset direction: away from origin side of line normal (a, b).
    """
    # The normal vector is (a, b), already normalized if a^2 + b^2 = 1
    x, y = point
    # Determine sign: we offset in direction of normal or opposite
    x_new = x + offset * a
    y_new = y + offset * b
    return np.array([x_new, y_new])

def create_path_msg(points, frame_id="map", seq=0):
    """
    Create a nav_msgs/Path message from a list of 2D points.
    """
    path = Path()
    path.header.seq = seq
    path.header.frame_id = frame_id
    path.header.stamp = rospy.Time.now()
    
    for i, pt in enumerate(points):
        pose = PoseStamped()
        pose.header.seq = i
        pose.header.frame_id = frame_id
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = pt[0]
        pose.pose.position.y = pt[1]
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    
    return path

def main():
    rospy.init_node('fit_and_publish_line_paths', anonymous=False)
    
    # Load path data
    csv_path = rospy.get_param('~csv_path', 
        '/home/user/catkin_ws/src/path_tools/data/path/recorded.csv')
    offset_dist = rospy.get_param('~offset_distance', 3.5)
    frame_id = rospy.get_param('~frame_id', 'map')
    
    rospy.loginfo(f"Loading path from: {csv_path}")
    points = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x'])
            y = float(row['y'])
            points.append([x, y])
    
    points = np.array(points)
    rospy.loginfo(f"Loaded {len(points)} points")
    
    # Fit line
    a, b, c, direction = fit_line_to_path(points)
    rospy.loginfo(f"Line equation: {a:.6f}*x + {b:.6f}*y + {c:.6f} = 0")
    rospy.loginfo(f"Normal vector: ({a:.6f}, {b:.6f})")
    rospy.loginfo(f"Direction vector: ({direction[0]:.6f}, {direction[1]:.6f})")
    
    # Project points onto the line to get central path
    # For each point, find closest point on line
    central_points = []
    for pt in points:
        x, y = pt
        # Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
        # Since we normalized, it's just |ax + by + c|
        dist = a * x + b * y + c
        # Closest point on line: pt - dist * normal
        closest = pt - dist * np.array([a, b])
        central_points.append(closest)
    
    # Simplify path by keeping only endpoints and midpoints for visualization
    # Or keep original projected points
    central_points = np.array(central_points)
    rospy.loginfo(f"Projected central path has {len(central_points)} points")
    
    # Create left and right offset paths
    left_points = np.array([offset_point_perpendicular(pt, a, b, offset_dist) 
                            for pt in central_points])
    right_points = np.array([offset_point_perpendicular(pt, a, b, -offset_dist) 
                             for pt in central_points])
    
    # Create ROS messages
    central_path = create_path_msg(central_points, frame_id=frame_id, seq=0)
    left_path = create_path_msg(left_points, frame_id=frame_id, seq=1)
    right_path = create_path_msg(right_points, frame_id=frame_id, seq=2)
    
    # Publish
    pub_central = rospy.Publisher('/central/smoothed_path', Path, queue_size=10)
    pub_left = rospy.Publisher('/left/smoothed_path', Path, queue_size=10)
    pub_right = rospy.Publisher('/right/smoothed_path', Path, queue_size=10)
    
    rospy.loginfo("Publishing paths...")
    rate = rospy.Rate(1)
    
    # Publish once per second
    while not rospy.is_shutdown():
        pub_central.publish(central_path)
        pub_left.publish(left_path)
        pub_right.publish(right_path)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
