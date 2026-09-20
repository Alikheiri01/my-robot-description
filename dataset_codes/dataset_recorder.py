#!/usr/bin/env python3
"""
Phase D dataset recorder -- assembles (trajectory_history, crop, future_target)
training samples from the live pipeline and writes them out as .npz files.

WHY A SEPARATE TIMER, NOT A CALLBACK ON /occupancy_grid/base
------------------------------------------------------------------
pedestrian_crop_view_dynamic.py recomputes on every incoming base-grid
message, because it's a live VIEWER -- redrawing too often costs nothing.
But that topic publishes at camera rate, far faster than the pedestrian
actually moves (BRIDGE_UPDATE_RATE_HZ=2.0 in hunav_config.py). Sampling a
training example on every grid frame would just duplicate near-identical
poses over and over. This node instead samples on a fixed timer at
DATASET_SAMPLE_PERIOD_SEC (matched to the pedestrian's own update rate),
using whatever the latest /odom, /people_smoothed_pose and
/occupancy_grid/base messages happen to be at each tick.

ONE CONSISTENT LOCAL FRAME PER SAMPLE
------------------------------------------------------------------
The crop (from extract_pedestrian_crop.py) is already expressed in the
pedestrian's own local frame at that instant -- rotated to that instant's
smoothed heading, origin at that instant's position. To keep the
trajectory history and future-target consistent with that same idea, both
are re-expressed relative to the ANCHOR timestep's position+heading (the
timestep the crop itself belongs to), using extract_pedestrian_crop.py's
own base_to_local() helper -- reused as-is rather than reimplemented, since
it's already self-tested (see that file's self_test()). Every pose
involved (/odom, /people_smoothed_pose) already lives in the same shared
'odom' frame after heading_smoother.py's own frame fix, so no additional
world/odom conversion is needed here -- only the crop extraction step
needs the transient base_link transform, exactly as
pedestrian_crop_view_dynamic.py already does it.

Sample layout (each is one .npz file):
    trajectory_history : float32 (TRAJECTORY_HISTORY_LENGTH, 3)
                          [local_x, local_y, local_heading] per step,
                          oldest first, LAST row is the anchor step itself
                          (local_x=local_y=local_heading=0 by construction)
    crop                : int8 (80, 80) -- the anchor step's own crop,
                          same -1/0/100 convention as the base grid
    future_target       : float32 (FUTURE_HORIZON_LENGTH, 3) -- same
                          [local_x, local_y, local_heading] convention,
                          the TRUE (unlagged, from /people_smoothed_pose
                          directly) continuation after the anchor step
    agent_id            : str, currently always 'agent1' (single-agent)
    sample_period_sec    : float, DATASET_SAMPLE_PERIOD_SEC at record time
    wall_time            : float, time.time() when the sample was emitted

REQUIRED: the full pipeline must be running (robot + depth pipeline +
occupancy grid + HuNav or the manual teleop tool driving the pedestrian +
heading_smoother.py). This node is purely a consumer of already-published
topics -- it doesn't drive the pedestrian itself.
"""
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# hunav_config.py and the scene-codes modules (build_occupancy_grid,
# extract_pedestrian_crop) live in sibling directories, same reasoning and
# same fix pattern as pedestrian_crop_view_dynamic.py's own sys.path
# insertion -- see that file's docstring for the full "why" (never moved
# during the hunav_codes/ consolidation, and duplicating either file into
# yet another folder would reintroduce the multiple-copies fragility this
# project has repeatedly fixed elsewhere).
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))
sys.path.insert(0, str(_THIS_DIR.parent / 'scene codes'))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseArray

import build_occupancy_grid as bog  # noqa: F401  (imported for extract_pedestrian_crop's own use)
from extract_pedestrian_crop import extract_pedestrian_crop, base_to_local
from hunav_config import (
    yaw_from_quaternion, ensure_single_instance,
    TRAJECTORY_HISTORY_LENGTH, FUTURE_HORIZON_LENGTH,
    DATASET_SAMPLE_PERIOD_SEC, DATASET_STALE_THRESHOLD_SEC,
)

DEFAULT_OUTPUT_DIR = _THIS_DIR / 'samples'


def _wrap_angle(angle: float) -> float:
    """Wraps to (-pi, pi] -- needed since relative headings are a plain
    subtraction and can land outside the usual range."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_pose_to_base_link(x, y, yaw, robot_pos, robot_yaw):
    """Identical to pedestrian_crop_view_dynamic.py's own version -- same
    convention, deliberately kept in sync with it rather than reimplemented
    differently."""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    dx, dy = x - robot_pos[0], y - robot_pos[1]
    base_x = c * dx + s * dy
    base_y = -s * dx + c * dy
    base_yaw = yaw - robot_yaw
    return base_x, base_y, base_yaw


class DatasetRecorder(Node):
    def __init__(self, output_dir: Path):
        super().__init__('dataset_recorder')

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.sample_count = 0

        self.robot_pose = None       # (x, y, yaw) in odom frame
        self.robot_last_seen = None
        self.ped_pose = None         # (x, y, yaw) in odom frame
        self.ped_last_seen = None
        self.base_grid = None        # np.ndarray, from /occupancy_grid/base
        self.grid_last_seen = None

        window_len = TRAJECTORY_HISTORY_LENGTH + FUTURE_HORIZON_LENGTH
        self.buffer = deque(maxlen=window_len)

        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped_cb, 10)
        self.create_subscription(OccupancyGrid, '/occupancy_grid/base', self._grid_cb, 10)
        self.create_timer(DATASET_SAMPLE_PERIOD_SEC, self._tick)

        self.get_logger().info(
            f'dataset_recorder started. Writing samples to {self.output_dir} '
            f'(run id {self.run_id}). history={TRAJECTORY_HISTORY_LENGTH} steps, '
            f'future={FUTURE_HORIZON_LENGTH} steps, '
            f'period={DATASET_SAMPLE_PERIOD_SEC:.2f}s -- needs '
            f'{window_len} good ticks ({window_len * DATASET_SAMPLE_PERIOD_SEC:.1f}s) '
            f'before the first sample.'
        )

    # --- subscriptions: just cache the latest reading ----------------------
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.robot_pose = (p.x, p.y, yaw)
        self.robot_last_seen = time.monotonic()

    def _ped_cb(self, msg: PoseArray):
        if not msg.poses:
            return  # no agent tracked right now -- ped_last_seen NOT refreshed,
                     # so staleness correctly kicks in below
        pose = msg.poses[0]  # single-agent, matches every other assumption in this codebase
        qz, qw = pose.orientation.z, pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        self.ped_pose = (pose.position.x, pose.position.y, yaw)
        self.ped_last_seen = time.monotonic()

    def _grid_cb(self, msg: OccupancyGrid):
        if msg.info.width != bog.GRID_N or msg.info.height != bog.GRID_N:
            self.get_logger().warn(
                f'received base grid is {msg.info.width}x{msg.info.height}, '
                f'expected {bog.GRID_N}x{bog.GRID_N} -- ignoring this message.',
                throttle_duration_sec=5.0)
            return
        self.base_grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.grid_last_seen = time.monotonic()

    # --- sampling timer ------------------------------------------------------
    def _tick(self):
        now = time.monotonic()

        if self.robot_pose is None or self.robot_last_seen is None or \
                (now - self.robot_last_seen) > DATASET_STALE_THRESHOLD_SEC:
            self._reset_buffer('robot pose (/odom) missing or stale')
            return
        if self.ped_pose is None or self.ped_last_seen is None or \
                (now - self.ped_last_seen) > DATASET_STALE_THRESHOLD_SEC:
            self._reset_buffer('pedestrian pose (/people_smoothed_pose) missing or stale')
            return
        if self.base_grid is None or self.grid_last_seen is None or \
                (now - self.grid_last_seen) > DATASET_STALE_THRESHOLD_SEC:
            self._reset_buffer('occupancy grid (/occupancy_grid/base) missing or stale')
            return

        ped_x_bl, ped_y_bl, ped_heading_bl = world_pose_to_base_link(
            self.ped_pose[0], self.ped_pose[1], self.ped_pose[2],
            self.robot_pose, self.robot_pose[2])
        crop = extract_pedestrian_crop(self.base_grid, ped_x_bl, ped_y_bl, ped_heading_bl)

        self.buffer.append({
            'ped_x_odom': self.ped_pose[0],
            'ped_y_odom': self.ped_pose[1],
            'ped_yaw_odom': self.ped_pose[2],
            'crop': crop,
        })

        if len(self.buffer) == self.buffer.maxlen:
            self._emit_sample()

    def _reset_buffer(self, reason: str):
        if len(self.buffer) > 0:
            self.get_logger().warn(
                f'Resetting sample buffer ({len(self.buffer)} ticks discarded): {reason}',
                throttle_duration_sec=5.0)
            self.buffer.clear()
        else:
            self.get_logger().warn(reason, throttle_duration_sec=5.0)

    # --- sample assembly -------------------------------------------------
    def _emit_sample(self):
        entries = list(self.buffer)
        history_entries = entries[:TRAJECTORY_HISTORY_LENGTH]
        future_entries = entries[TRAJECTORY_HISTORY_LENGTH:]
        anchor = history_entries[-1]  # the crop's own timestep
        anchor_x = anchor['ped_x_odom']
        anchor_y = anchor['ped_y_odom']
        anchor_yaw = anchor['ped_yaw_odom']

        def to_anchor_local(e):
            lx, ly = base_to_local(e['ped_x_odom'], e['ped_y_odom'], anchor_x, anchor_y, anchor_yaw)
            lyaw = _wrap_angle(e['ped_yaw_odom'] - anchor_yaw)
            return float(lx), float(ly), float(lyaw)

        history = np.array([to_anchor_local(e) for e in history_entries], dtype=np.float32)
        future = np.array([to_anchor_local(e) for e in future_entries], dtype=np.float32)

        out_path = self.output_dir / f'sample_{self.run_id}_{self.sample_count:06d}.npz'
        np.savez(
            out_path,
            trajectory_history=history,
            crop=anchor['crop'].astype(np.int8),
            future_target=future,
            agent_id='agent1',
            sample_period_sec=DATASET_SAMPLE_PERIOD_SEC,
            wall_time=time.time(),
        )
        self.sample_count += 1
        if self.sample_count == 1 or self.sample_count % 20 == 0:
            self.get_logger().info(f'{self.sample_count} samples saved -> {self.output_dir}')


def main():
    ensure_single_instance('dataset_recorder')

    output_dir = DEFAULT_OUTPUT_DIR
    if len(sys.argv) == 2:
        output_dir = Path(sys.argv[1]).expanduser().resolve()

    rclpy.init()
    node = DatasetRecorder(output_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Stopped. {node.sample_count} total samples saved to {node.output_dir}.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()