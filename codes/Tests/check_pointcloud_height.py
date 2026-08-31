#!/usr/bin/env python3
"""
Reads one message from a PointCloud2 topic and reports its Z-range,
compared against the known true floor height, to numerically settle
whether points are sinking below the real ground plane.

Usage:
    python3 check_pointcloud_height.py --topic /depth_cam/left/points_corrected --floor-z -0.0925

--floor-z is the Z coordinate of the true floor, expressed in the SAME
frame the point cloud is published in (left_camera_link_optical, which
gets transformed into whatever Fixed Frame you're viewing in RViz --
here we just read raw message data, transform via TF into base_link,
so pass the floor height relative to base_link: -0.0925 for this robot).
"""
import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import tf2_ros
from tf2_ros import TransformException
import numpy as np
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class HeightChecker(Node):
    def __init__(self, topic, floor_z, target_frame):
        super().__init__('pointcloud_height_checker')
        self.floor_z = floor_z
        self.target_frame = target_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.sub = self.create_subscription(
            PointCloud2, topic, self.callback, qos_profile_sensor_data)
        self.done = False

    def callback(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'TF not ready yet: {ex}')
            return

        cloud_in_target = do_transform_cloud(msg, tf)
        points = np.array([[p[0], p[1], p[2]] for p in
                            pc2.read_points(cloud_in_target, field_names=('x', 'y', 'z'), skip_nans=True)])

        if points.shape[0] == 0:
            self.get_logger().warn('No valid points in this message, waiting for next...')
            return

        z = points[:, 2]
        below_floor = z < (self.floor_z - 0.02)  # 2cm tolerance
        n_below = int(np.sum(below_floor))

        print(f'\n--- Point cloud height report (frame: {self.target_frame}) ---')
        print(f'Total valid points:      {len(z)}')
        print(f'Min Z:                   {z.min():.4f} m')
        print(f'Max Z:                   {z.max():.4f} m')
        print(f'Expected floor Z:        {self.floor_z:.4f} m')
        print(f'Points >2cm below floor: {n_below}  ({100*n_below/len(z):.1f}%)')
        if n_below > 0:
            worst = z[below_floor].min()
            print(f'Worst offender depth:    {worst:.4f} m ({self.floor_z - worst:.4f} m below floor)')
        print('---------------------------------------------------\n')

        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/depth_cam/left/points_corrected')
    parser.add_argument('--floor-z', type=float, default=-0.0925,
                         help='True floor Z in target frame (base_link). Default: -wheel_radius')
    parser.add_argument('--target-frame', default='base_link')
    args = parser.parse_args()

    rclpy.init()
    node = HeightChecker(args.topic, args.floor_z, args.target_frame)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()