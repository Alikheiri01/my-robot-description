#!/usr/bin/env python3
"""
fuse_depth_clouds_fast.py -- the same fusion as fuse_depth_clouds.py, fast enough
to keep up with the cameras (2026-10-24).

WHY
probe_grid_gaps.py showed the fused cloud coming out only ~6 times per simulated
second while both input clouds arrived ~14 times each, with holes up to 1.1 s:
77 of 78 grid gaps started here. The old callback spent its time in three places:
  * tf2_sensor_msgs.do_transform_cloud: unpacks every point of the ~300k-point
    cloud, transforms it and packs a NEW PointCloud2, which is then unpacked again;
  * np.unique(..., axis=0) on ~600k rows of 3 integers (row-wise unique is very
    slow in numpy);
  * create_cloud_xyz32(fused.tolist()): a Python list of every output point.
While one callback ran, newer clouds were dropped.

WHAT IS DIFFERENT (only HOW it is computed -- the output is the same):
  * x, y, z are read straight from the message bytes (numpy view, no copy loop),
    and the camera -> base_link transform is applied as one matrix product;
  * voxel indices are packed into one integer per point (order-preserving), so
    unique runs on a 1-D array; voxel means use np.bincount;
  * the output message is built directly from the numpy bytes.
Same: synchronisation (ApproximateTimeSynchronizer, queue 10, slop 0.05 s),
latest static transform, floor removal (FLOOR_Z_BASE + GROUND_FILTER_MARGIN),
voxel averaging with VOXEL_SIZE, frame base_link, the LEFT cloud's stamp, topic.
Voxel means can differ from the old code in the last float digit (summation
order); the points come out in the same order.

Every 10 s it logs how many fused clouds it made and how long one took, so you
can see it keep up.

Publishes: /depth_cam/fused/points (sensor_msgs/PointCloud2, frame_id=base_link)
"""
import time

import numpy as np

TARGET_FRAME = 'base_link'
VOXEL_SIZE = 0.03            # m (unchanged)
FLOOR_Z_BASE = -0.0925       # true floor height in base_link (unchanged)
GROUND_FILTER_MARGIN = 0.03  # (unchanged)
FLOAT32 = 7                  # sensor_msgs/PointField.FLOAT32


# --- pure numpy parts (tested without ROS) ------------------------------------
def xyz_from_buffer(data, width, height, point_step, row_step, offsets, big_endian=False):
    """(N, 3) float32 x, y, z straight from PointCloud2 bytes (finite points only)."""
    order = '>' if big_endian else '<'
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': [order + 'f4'] * 3,
                   'offsets': list(offsets), 'itemsize': point_step})
    if row_step == width * point_step:
        pts = np.frombuffer(data, dtype=dt, count=width * height)
    else:                                   # padded rows: cut each row to its points
        buf = np.frombuffer(data, dtype=np.uint8)
        rows = buf[:height * row_step].reshape(height, row_step)[:, :width * point_step]
        pts = np.frombuffer(np.ascontiguousarray(rows).tobytes(), dtype=dt, count=width * height)
    xyz = np.empty((pts.shape[0], 3), dtype=np.float32)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = pts['x'], pts['y'], pts['z']
    return xyz[np.isfinite(xyz).all(axis=1)]


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def apply_transform(xyz, R, t):
    return (xyz.astype(np.float64) @ R.T + t).astype(np.float32)


def voxel_downsample_fast(points, voxel_size):
    """Mean of the points in each voxel_size cell; same result and order as
    fuse_depth_clouds.voxel_downsample (np.unique(axis=0) sorts cells by x, then y, then z)."""
    if points.shape[0] == 0:
        return points.astype(np.float32)
    idx = np.floor(points / voxel_size).astype(np.int64)
    idx -= idx.min(axis=0)
    span = idx.max(axis=0) + 1
    key = (idx[:, 0] * span[1] + idx[:, 1]) * span[2] + idx[:, 2]      # lexicographic order kept
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    p = points.astype(np.float64)
    out = np.empty((counts.shape[0], 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.bincount(inverse, weights=p[:, k], minlength=counts.shape[0])
    return (out / counts[:, None]).astype(np.float32)


def fuse(left_xyz_base, right_xyz_base):
    raw = np.concatenate([left_xyz_base, right_xyz_base], axis=0)
    obstacles = raw[raw[:, 2] > (FLOOR_Z_BASE + GROUND_FILTER_MARGIN)]
    return voxel_downsample_fast(obstacles, VOXEL_SIZE)


# --- ROS ------------------------------------------------------------------------
def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    import message_filters
    from sensor_msgs.msg import PointCloud2, PointField
    import tf2_ros
    from tf2_ros import TransformException

    class CloudFuser(Node):
        def __init__(self):
            super().__init__('depth_cloud_fuser')
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.tf_cache = {}
            self.pub = self.create_publisher(PointCloud2, '/depth_cam/fused/points', qos_profile_sensor_data)
            left_sub = message_filters.Subscriber(self, PointCloud2, '/depth_cam/left/points_corrected',
                                                  qos_profile=qos_profile_sensor_data)
            right_sub = message_filters.Subscriber(self, PointCloud2, '/depth_cam/right/points_corrected',
                                                   qos_profile=qos_profile_sensor_data)
            self.sync = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.05)
            self.sync.registerCallback(self.callback)
            self.fields = [PointField(name=n, offset=4 * i, datatype=FLOAT32, count=1) for i, n in enumerate('xyz')]
            self.n, self.ms = 0, []
            self.create_timer(10.0, self._stats)
            self.get_logger().info('Depth cloud fuser (fast) started.')

        def _to_base(self, msg):
            frame = msg.header.frame_id
            if frame not in self.tf_cache:            # camera -> base_link is fixed: look it up once
                try:
                    tf = self.tf_buffer.lookup_transform(TARGET_FRAME, frame, rclpy.time.Time())
                except TransformException as ex:
                    self.get_logger().warn(f'TF not ready: {ex}', throttle_duration_sec=2.0)
                    return None
                q, tr = tf.transform.rotation, tf.transform.translation
                self.tf_cache[frame] = (quat_to_matrix(q.x, q.y, q.z, q.w), np.array([tr.x, tr.y, tr.z]))
            R, t = self.tf_cache[frame]
            off = {f.name: f.offset for f in msg.fields}
            types = {f.name: f.datatype for f in msg.fields}
            if any(types.get(n) != FLOAT32 for n in 'xyz'):
                self.get_logger().error('x/y/z are not float32 in the input cloud', throttle_duration_sec=10.0)
                return None
            xyz = xyz_from_buffer(msg.data, msg.width, msg.height, msg.point_step, msg.row_step,
                                  (off['x'], off['y'], off['z']), msg.is_bigendian)
            return apply_transform(xyz, R, t)

        def callback(self, left_msg, right_msg):
            t0 = time.perf_counter()
            left = self._to_base(left_msg)
            right = self._to_base(right_msg)
            if left is None or right is None or (left.shape[0] == 0 and right.shape[0] == 0):
                return
            fused = fuse(left, right)
            out = PointCloud2()
            out.header = left_msg.header
            out.header.frame_id = TARGET_FRAME
            out.height, out.width = 1, int(fused.shape[0])
            out.fields = self.fields
            out.is_bigendian = False
            out.point_step, out.row_step = 12, 12 * int(fused.shape[0])
            out.is_dense = True
            out.data = np.ascontiguousarray(fused, dtype='<f4').tobytes()
            self.pub.publish(out)
            self.n += 1
            self.ms.append(1000.0 * (time.perf_counter() - t0))

        def _stats(self):
            if self.ms:
                self.get_logger().info(f'{self.n} fused clouds so far; last 10 s: {len(self.ms)} clouds, '
                                       f'{np.mean(self.ms):.0f} ms each (max {np.max(self.ms):.0f} ms)')
            else:
                self.get_logger().warn('no fused cloud in the last 10 s (are both camera clouds arriving?)')
            self.ms = []

    rclpy.init()
    node = CloudFuser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()