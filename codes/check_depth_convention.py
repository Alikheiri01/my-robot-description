#!/usr/bin/env python3
"""
Tests whether Gazebo's rgbd_camera depth image reports Z-DEPTH (forward
distance, what depth_image_proc assumes) or RANGE (Euclidean distance,
what would explain the "sinking below floor" bug).

Uses the floor itself as ground truth: we know the camera's height above
the floor and that it's mounted with zero pitch, so for ANY pixel we can
compute geometrically where its ray should intersect the floor -- both
as a Z-depth value and as a range value -- with no cube or precise robot
placement required. Just needs the robot on open flat ground.

Usage:
    python3 check_depth_convention.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv_bridge

# Known extrinsics (from tf2_echo base_link -> left_camera_link_optical,
# and Component Inspector ground-truth floor height). Update if your
# geometry differs.
CAMERA_ORIGIN_BASE = np.array([0.366, -0.036, 0.560])  # camera optical origin in base_link
FLOOR_Z_BASE = -0.0925  # true floor height in base_link


class ConventionChecker(Node):
    def __init__(self):
        super().__init__('depth_convention_checker')
        self.bridge = cv_bridge.CvBridge()
        self.caminfo = None
        self.done = False
        self.create_subscription(CameraInfo, '/depth_cam/left/camera_info',
                                  self.caminfo_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/depth_cam/left/depth_image',
                                  self.depth_cb, qos_profile_sensor_data)

    def caminfo_cb(self, msg):
        self.caminfo = msg

    def depth_cb(self, msg):
        if self.caminfo is None or self.done:
            return

        K = self.caminfo.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        h, w = depth.shape

        # Sample points: center-bottom (mostly vertical off-axis),
        # and the four corners (max combined off-axis angle, matching
        # the "wing" pixels in your screenshot).
        samples = {
            'center (u=cx, v=3h/4)': (int(cx), int(3 * h / 4)),
            'bottom-left corner': (5, h - 5),
            'bottom-right corner': (w - 5, h - 5),
        }

        print(f'\n--- Depth convention test (fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}) ---')
        for label, (u, v) in samples.items():
            d_raw = float(depth[v, u])
            if not np.isfinite(d_raw) or d_raw <= 0:
                print(f'{label}: no valid depth at this pixel, try a nearby one')
                continue

            # Ray direction in base_link frame (unnormalized, forward component = 1)
            dy = -(u - cx) / fx
            dz = -(v - cy) / fy
            direction = np.array([1.0, dy, dz])

            # Ground-truth Z-depth (t) such that ray hits the floor
            t_zdepth = (FLOOR_Z_BASE - CAMERA_ORIGIN_BASE[2]) / dz if dz != 0 else float('inf')
            # Ground-truth range for the same floor intersection
            t_range = t_zdepth * np.linalg.norm(direction)

            print(f'\n{label} (pixel u={u}, v={v}):')
            print(f'  raw depth value:              {d_raw:.4f}')
            print(f'  predicted Z-depth to floor:   {t_zdepth:.4f}  (diff: {abs(d_raw - t_zdepth):.4f})')
            print(f'  predicted range to floor:     {t_range:.4f}  (diff: {abs(d_raw - t_range):.4f})')
            if abs(d_raw - t_zdepth) < abs(d_raw - t_range):
                print('  --> raw value matches Z-DEPTH convention (depth_image_proc assumption is correct)')
            else:
                print('  --> raw value matches RANGE convention (this is the bug)')

        print('---------------------------------------------------\n')
        self.done = True


def main():
    rclpy.init()
    node = ConventionChecker()
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