#!/usr/bin/env python3
"""
Numerically verifies the fused point cloud against ground truth: given the
cube's and robot's exact world poses (read from Gazebo's Component Inspector),
predicts where the cube's bounding box should land in base_link, then
compares against what the fused cloud actually measures.

Usage example:
    python3 check_fused_cloud_accuracy.py \
        --cube-pos 2.0 0.3 0.25 --cube-size 0.5 0.5 0.5 --cube-yaw 0 \
        --robot-pos 0.0 0.0 0.0925 --robot-yaw 0.0

All positions/sizes in meters, yaw in radians. Get these from Gazebo's
Component Inspector: click the entity, expand Pose, read Position/Rotation.
robot-pos should be the my_robot (base_link) pose; robot Z should read
close to 0.0925 if it's settled correctly (matches our earlier check).
"""
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

FLOOR_Z_BASE = -0.0925
FLOOR_MARGIN = 0.03  # points within this of the floor are treated as floor, not object


def yaw_rot_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def predicted_bbox_in_base(cube_pos, cube_size, cube_yaw, robot_pos, robot_yaw):
    hx, hy, hz = np.array(cube_size) / 2.0
    local_corners = np.array([[sx * hx, sy * hy, sz * hz]
                               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    R_cube = yaw_rot_matrix(cube_yaw)
    world_corners = (R_cube @ local_corners.T).T + np.array(cube_pos)

    R_robot = yaw_rot_matrix(robot_yaw)
    R_robot_inv = R_robot.T
    base_corners = (R_robot_inv @ (world_corners - np.array(robot_pos)).T).T

    return base_corners.min(axis=0), base_corners.max(axis=0)


class FusedCloudChecker(Node):
    def __init__(self, predicted_min, predicted_max):
        super().__init__('fused_cloud_accuracy_checker')
        self.predicted_min = predicted_min
        self.predicted_max = predicted_max
        self.done = False
        self.create_subscription(PointCloud2, '/depth_cam/fused/points',
                                  self.callback, qos_profile_sensor_data)

    def callback(self, msg: PointCloud2):
        if self.done:
            return
        points = np.array([[p[0], p[1], p[2]] for p in
                            pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)])
        if points.shape[0] == 0:
            return

        object_mask = points[:, 2] > (FLOOR_Z_BASE + FLOOR_MARGIN)
        obj_points = points[object_mask]
        if obj_points.shape[0] == 0:
            print('No above-floor points found -- is the object actually in view?')
            self.done = True
            return

        measured_min = obj_points.min(axis=0)
        measured_max = obj_points.max(axis=0)

        print(f'\n--- Fused cloud accuracy report ---')
        print(f'Total points: {points.shape[0]}   Above-floor (object) points: {obj_points.shape[0]}')
        print(f'\n{"axis":<6}{"pred_min":>10}{"meas_min":>10}{"diff":>8}   '
              f'{"pred_max":>10}{"meas_max":>10}{"diff":>8}')
        for i, axis in enumerate('xyz'):
            dmin = measured_min[i] - self.predicted_min[i]
            dmax = measured_max[i] - self.predicted_max[i]
            print(f'{axis:<6}{self.predicted_min[i]:>10.4f}{measured_min[i]:>10.4f}{dmin:>8.4f}   '
                  f'{self.predicted_max[i]:>10.4f}{measured_max[i]:>10.4f}{dmax:>8.4f}')
        print('------------------------------------\n')
        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube-pos', type=float, nargs=3, required=True)
    parser.add_argument('--cube-size', type=float, nargs=3, required=True)
    parser.add_argument('--cube-yaw', type=float, default=0.0)
    parser.add_argument('--robot-pos', type=float, nargs=3, required=True)
    parser.add_argument('--robot-yaw', type=float, default=0.0)
    args = parser.parse_args()

    pred_min, pred_max = predicted_bbox_in_base(
        args.cube_pos, args.cube_size, args.cube_yaw, args.robot_pos, args.robot_yaw)

    rclpy.init()
    node = FusedCloudChecker(pred_min, pred_max)
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