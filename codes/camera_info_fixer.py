#!/usr/bin/env python3
"""
Gazebo Fortress's rgbd_camera sensor has a known bug: it correctly applies
custom <lens><intrinsics> to CameraInfo's K matrix, but computes the P
(projection) matrix through a separate, still-broken default path.
depth_image_proc uses P, not K, for its actual 3D reconstruction math --
so this bug silently corrupts every point cloud even after K looks correct.

This node subscribes to the raw (P-broken) camera_info topics and
republishes corrected versions with both K and P set from the true
intrinsics, on new "_fixed" topics that depth_image_proc should be
pointed at instead.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CameraInfo

# True intrinsics, matching the <lens><intrinsics> block in depth_camera.xacro
FX = FY = 432.43
CX = 319.5
CY = 239.5


class CameraInfoFixer(Node):
    def __init__(self, in_topic: str, out_topic: str, node_name: str):
        super().__init__(node_name)
        self.pub = self.create_publisher(CameraInfo, out_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(CameraInfo, in_topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f'Relaying {in_topic} -> {out_topic} with corrected K/P')

    def cb(self, msg: CameraInfo):
        msg.k = [FX, 0.0, CX,
                 0.0, FY, CY,
                 0.0, 0.0, 1.0]
        msg.p = [FX, 0.0, CX, 0.0,
                 0.0, FY, CY, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        self.pub.publish(msg)


def main():
    rclpy.init()
    left = CameraInfoFixer('/depth_cam/left/camera_info',
                            '/depth_cam/left/camera_info_fixed',
                            'camera_info_fixer_left')
    right = CameraInfoFixer('/depth_cam/right/camera_info',
                             '/depth_cam/right/camera_info_fixed',
                             'camera_info_fixer_right')
    executor = MultiThreadedExecutor()
    executor.add_node(left)
    executor.add_node(right)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        left.destroy_node()
        right.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()