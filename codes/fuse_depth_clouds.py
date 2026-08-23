#!/usr/bin/env python3
"""
Fuses /depth_cam/left/points_corrected and /depth_cam/right/points_corrected
into a single point cloud in base_link.

- Synchronizes left/right by timestamp (both run at the same update_rate,
  so this should match closely).
- Transforms both into base_link using the verified-correct TF chain.
- Concatenates, then voxel-downsamples to collapse the overlap region
  (where both cameras see the same surface) back down to sane density.

Publishes: /depth_cam/fused/points (sensor_msgs/PointCloud2, frame_id=base_link)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

TARGET_FRAME = 'base_link'
VOXEL_SIZE = 0.03  # meters; tune based on how coarse/fine you want the fused cloud


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Groups points into voxel_size cells and averages each group, rather
    than just picking one point per cell -- keeps geometry more faithful
    than nearest-point decimation."""
    if points.shape[0] == 0:
        return points
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return (sums / counts[:, None]).astype(np.float32)


class CloudFuser(Node):
    def __init__(self):
        super().__init__('depth_cloud_fuser')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(PointCloud2, '/depth_cam/fused/points', qos_profile_sensor_data)

        left_sub = message_filters.Subscriber(self, PointCloud2, '/depth_cam/left/points_corrected',
                                               qos_profile=qos_profile_sensor_data)
        right_sub = message_filters.Subscriber(self, PointCloud2, '/depth_cam/right/points_corrected',
                                                qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.callback)

        self.get_logger().info('Depth cloud fuser started, waiting for synced left/right clouds...')

    def transform_to_base(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(TARGET_FRAME, msg.header.frame_id, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'TF not ready: {ex}')
            return None
        cloud_base = do_transform_cloud(msg, tf)
        points = np.array([[p[0], p[1], p[2]] for p in
                            pc2.read_points(cloud_base, field_names=('x', 'y', 'z'), skip_nans=True)],
                           dtype=np.float32)
        return points

    def callback(self, left_msg: PointCloud2, right_msg: PointCloud2):
        left_pts = self.transform_to_base(left_msg)
        right_pts = self.transform_to_base(right_msg)
        if left_pts is None or right_pts is None:
            return
        if left_pts.shape[0] == 0 and right_pts.shape[0] == 0:
            return

        fused = np.concatenate([left_pts, right_pts], axis=0)
        fused = voxel_downsample(fused, VOXEL_SIZE)

        header = left_msg.header
        header.frame_id = TARGET_FRAME
        cloud_msg = pc2.create_cloud_xyz32(header, fused.tolist())
        self.pub.publish(cloud_msg)


def main():
    rclpy.init()
    node = CloudFuser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()