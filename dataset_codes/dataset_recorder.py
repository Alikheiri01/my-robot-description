#!/usr/bin/env python3
"""
Phase D dataset recorder -- assembles (trajectory_history, crop, future_target)
training samples from the live pipeline and writes them out as .npz files.

REWRITTEN 2026-09-30: TIME-ALIGNED, SIMULATION-CLOCK, EVENT-DRIVEN
------------------------------------------------------------------
The first version sampled on a WALL-clock timer using whatever the latest
/odom, /people_smoothed_pose and /occupancy_grid/base messages happened to be.
Two things were wrong with that once the simulation ran slower than real time
(real-time factor ~0.32) and the depth pipeline was measured to trail the
pose streams:
  1. A 0.5 s WALL period is only ~0.16 s of SIMULATION time, so consecutive
     history/future rows were ~0.16 s apart instead of 0.5 s (the "8 steps =
     4 s of history" in hunav_config.py was really ~1.3 s), and the same
     pedestrian pose was recorded about three times over.
  2. The crop was cut from the newest grid at the newest pedestrian pose --
     a pose from a different moment than the depth data in that grid
     (validate_time_alignment.py: ~0.26 m error at walking speed).
Now everything is keyed by SIMULATION timestamps:

  * ANCHOR TIME. For a depth-derived grid stamped t_g, the pedestrian the
    cameras see is where the pose stream put them at  T = t_g + POSE_LOOKUP_OFFSET_SEC.
    (2026-10-01: that stream is now the plugin's APPLIED pose, stamped with the
    simulation time of the step the actor was moved in, relayed by
    applied_pose_relay.py as /people_smoothed_pose, so the offset is 0.0. Until
    then it was HuNav's /people, the command, with a measured offset of -0.10 ..
    -0.29 s that changed between sessions. Samples carry lookup_offset_sec.) The
    sample's anchor is that T, and its crop is cut from THAT grid at the
    pedestrian pose at T, with the heading of the segment containing T
    (time_sync.StampedPoseHistory.step_yaw_at).
  * SAMPLE LATTICE. Anchors are spaced DATASET_SAMPLE_PERIOD_SEC of simulation
    time apart (0.5 s = one HuNav step): for each target time the grid whose T
    is nearest is used, so the crop is at most half a grid interval (~0.14 s)
    from the ideal lattice time and is always centred exactly on the body the
    camera saw.
  * HISTORY / FUTURE. Read from the /people_smoothed_pose history at exactly
    T - (H-1-i)*period and T + (k+1)*period (linear position interpolation,
    heading of the containing segment) -- true 0.5 s spacing whatever the
    real-time factor, and no duplicated poses. Future rows need pose data up
    to T + F*period, so a sample is written that much simulation time after
    its anchor.
  * NO GUESSING. The pedestrian pose is only ever INTERPOLATED (never held or
    extrapolated). A sample is dropped, and counted by reason, if the pose
    history does not cover its window, has a dropout longer than
    MAX_POSE_GAP_SEC inside it, if no grid lies near the target time, if the
    robot pose is missing, or if the clocks of the streams do not match.

ONE CONSISTENT LOCAL FRAME PER SAMPLE
------------------------------------------------------------------
Unchanged: history and future rows are expressed relative to the ANCHOR
position + heading (extract_pedestrian_crop.base_to_local), so the last
history row is exactly (0, 0, 0). All poses (/odom, /people_smoothed_pose)
share the 'odom' frame after heading_smoother.py's own frame fix.

Sample layout (each is one .npz file):
    trajectory_history : float32 (TRAJECTORY_HISTORY_LENGTH, 3)
                          [local_x, local_y, local_heading] per step,
                          oldest first, LAST row is the anchor step itself
                          (local_x=local_y=local_heading=0 by construction)
    crop                : int8 (80, 80) -- the anchor's own crop,
                          same -1/0/100 convention as the base grid
    future_target       : float32 (FUTURE_HORIZON_LENGTH, 3) -- same
                          convention, the TRUE continuation after the anchor
                          (from the /people pose stream, not the lagged depth)
    agent_id            : str, currently always 'agent1' (single-agent)
    sample_period_sec   : float, DATASET_SAMPLE_PERIOD_SEC at record time
    wall_time           : float, time.time() when the sample was emitted
  NEW (for checking a recording frame by frame):
    schema              : 'time_aligned_v2'
    anchor_time_sim     : float64, T (simulation seconds)
    grid_stamp_sim      : float64, t_g of the grid the crop was cut from
    lookup_offset_sec   : float, POSE_LOOKUP_OFFSET_SEC used (T = t_g + this)
    history_times       : float64 (H,) simulation time of each history row
    future_times        : float64 (F,) simulation time of each future row
    robot_pose_odom     : float64 (3,) robot x, y, yaw (odom frame) at t_g
    ped_anchor_odom     : float64 (3,) pedestrian x, y, yaw (odom) at T
    max_pose_gap_sec    : float, largest spacing of pose samples in the window

REQUIRED: the full pipeline must be running (robot + depth pipeline +
occupancy grid + HuNav driving the pedestrian + heading_smoother.py), with
hunav_model_bridge.py on the simulation clock (use_sim_time:=true). This node
is purely a consumer of already-published topics. It does not need
use_sim_time itself: it never reads its own clock for sampling, only message
stamps (a wall-clock timer is used only for the status line).
"""
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# hunav_config.py, time_sync.py and the scene-codes modules (build_occupancy_grid,
# extract_pedestrian_crop) live in sibling directories -- see
# pedestrian_crop_view_dynamic.py's docstring for why they are not copied here.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))
sys.path.insert(0, str(_THIS_DIR.parent / 'scene codes'))

import numpy as np

from hunav_config import (
    POSE_LOOKUP_OFFSET_SEC,
    TRAJECTORY_HISTORY_LENGTH, FUTURE_HORIZON_LENGTH, DATASET_SAMPLE_PERIOD_SEC,
)
from time_sync import StampedPoseHistory, clocks_match, stamp_to_sec, wrap_angle

DEFAULT_OUTPUT_DIR = _THIS_DIR / 'samples'

# Largest allowed spacing between consecutive pedestrian pose samples inside a
# sample's window (normal spacing is 0.5 s). One missed HuNav tick (1.0 s) already
# makes the path between the surviving poses a guess, so such windows are dropped.
MAX_POSE_GAP_SEC = 0.8
# Largest distance between the target lattice time and the chosen grid's own T.
# Grids arrive every ~0.27 s of simulation time, so the nearest is normally < 0.14 s away.
MAX_GRID_MISMATCH_SEC = 0.30
# Robot pose lookup at the grid stamp (odom arrives ~30x per sim second).
ODOM_MAX_HOLD_SEC = 0.2
# If the next grid is this far past the target time, the lattice restarts there.
LATTICE_RESTART_SEC = 10.0
POSE_HISTORY_KEEP_SEC = 40.0
MAX_PENDING = 100


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


class AlignedSampler:
    """
    All the sampling logic, with no ROS in it (so it can be tested with
    synthetic data). Feed it events -- on_odom / on_ped / on_grid -- in arrival
    order; finished samples go to `sink(sample_dict)`.

    crop_fn(grid_array, ped_x_bl, ped_y_bl, ped_heading_bl) -> crop
        (extract_pedestrian_crop.extract_pedestrian_crop)
    to_local_fn(x, y, anchor_x, anchor_y, anchor_yaw) -> (local_x, local_y)
        (extract_pedestrian_crop.base_to_local)
    """

    def __init__(self, crop_fn, to_local_fn, sink, lookup_offset=POSE_LOOKUP_OFFSET_SEC,
                 history_len=TRAJECTORY_HISTORY_LENGTH, future_len=FUTURE_HORIZON_LENGTH,
                 period=DATASET_SAMPLE_PERIOD_SEC, max_gap=MAX_POSE_GAP_SEC, log=None):
        self.crop_fn = crop_fn
        self.to_local_fn = to_local_fn
        self.sink = sink
        self.off = lookup_offset
        self.H, self.F, self.P = history_len, future_len, period
        self.max_gap = max_gap
        self.log = log or (lambda level, msg: None)

        self.ped = StampedPoseHistory(max_age_sec=POSE_HISTORY_KEEP_SEC)
        self.odom = StampedPoseHistory(max_age_sec=POSE_HISTORY_KEEP_SEC)
        self.pending = []          # anchors waiting for pose data
        self.prev = None           # (t_g, T, loader) of the previous grid
        self.next_target = None    # next lattice time (in anchor time T)
        self.last_used_tg = None
        self.drops = Counter()
        self.n_saved = 0
        self.n_anchors = 0

    # --- events -------------------------------------------------------------
    def on_odom(self, t, x, y, yaw):
        self.odom.add(t, x, y, yaw)

    def on_ped(self, t, x, y, yaw):
        self.ped.add(t, x, y, yaw)
        self._advance()

    def on_grid(self, t_g, loader):
        """loader() -> the grid as an int8 array (called only if the grid is used)."""
        if self.prev is not None and t_g <= self.prev[0]:
            if t_g < self.prev[0] - 1.0:
                self._reset('simulation time went backwards (restart?)')
            else:
                return  # duplicate / reordered grid
        last_ped = self.ped.latest_time()
        if last_ped is None:
            self.drops['no pedestrian pose yet'] += 1
            return
        if not clocks_match(t_g, last_ped):
            self.drops['CLOCK MISMATCH grid vs pedestrian pose (run the bridge with use_sim_time:=true)'] += 1
            self.log('error', f'grid stamp {t_g:.2f} and pedestrian pose stamp {last_ped:.2f} are on '
                              f'DIFFERENT clocks -- nothing can be recorded')
            return

        T_new = t_g + self.off
        cur = (t_g, T_new, loader)
        if self.next_target is None:
            self.next_target = T_new
        elif T_new - self.next_target > LATTICE_RESTART_SEC:
            self.drops['grid gap: lattice restarted'] += 1
            self.next_target = T_new

        while T_new >= self.next_target - 1e-9:
            best = min([cur] + ([self.prev] if self.prev else []),
                       key=lambda c: abs(c[1] - self.next_target))
            if abs(best[1] - self.next_target) > MAX_GRID_MISMATCH_SEC:
                self.drops['no grid near the sample time'] += 1
            elif best[0] == self.last_used_tg:
                self.drops['grid already used (grids sparser than the sample period)'] += 1
            else:
                self._start_anchor(best)
            self.next_target += self.P
        self.prev = cur
        self._advance()

    # --- internals -----------------------------------------------------------
    def _reset(self, why):
        self.log('warn', f'resetting sampler: {why}')
        self.pending.clear()
        self.prev = None
        self.next_target = None
        self.last_used_tg = None

    def _start_anchor(self, cand):
        t_g, T, loader = cand
        robot = self.odom.at(t_g, max_hold=ODOM_MAX_HOLD_SEC)
        if robot is None:
            self.drops['no robot pose (/odom) at the grid time'] += 1
            return
        if len(self.pending) >= MAX_PENDING:
            self.pending.pop(0)
            self.drops['pending queue overflow'] += 1
        self.pending.append({'T': T, 't_g': t_g, 'grid': loader(), 'robot': robot, 'crop': None})
        self.last_used_tg = t_g
        self.n_anchors += 1

    def _advance(self):
        latest = self.ped.latest_time()
        if latest is None or not self.pending:
            return
        remaining = []
        for a in self.pending:
            if a['crop'] is None:
                if latest < a['T']:
                    remaining.append(a)     # the pose at the anchor time has not arrived yet
                    continue
                if not self._cut_crop(a):
                    continue
            if latest >= a['T'] + self.F * self.P - 1e-9:
                self._emit(a, latest)
            else:
                remaining.append(a)
        self.pending = remaining

    def _cut_crop(self, a):
        """Pedestrian pose AT the anchor time (interpolated, never held) + crop. False if impossible."""
        p = self.ped.at(a['T'])
        yaw = self.ped.step_yaw_at(a['T'])
        if p is None or yaw is None:
            self.drops['no pedestrian pose at the anchor time'] += 1
            return False
        x, y = p[0], p[1]
        rx, ry, ryaw = a['robot']
        bx, by, byaw = world_pose_to_base_link(x, y, yaw, (rx, ry), ryaw)
        a['crop'] = self.crop_fn(a['grid'], bx, by, byaw)
        a['ped'] = (x, y, yaw)
        a['grid'] = None
        return True

    def _emit(self, a, latest):
        T, P, H, F = a['T'], self.P, self.H, self.F
        t0, t1 = T - (H - 1) * P, T + F * P
        first = self.ped.first_time()
        if first is None or t0 < first - 1e-9:
            self.drops['pose history does not reach back far enough yet (start-up)'] += 1
            return
        gap = self.ped.max_gap(t0, min(t1, latest))
        if gap > self.max_gap:
            self.drops['pose dropout inside the sample window'] += 1
            return

        ax, ay, ayaw = a['ped']

        def row(t):
            t = min(max(t, first), latest)
            pos = self.ped.at(t)
            yaw = self.ped.step_yaw_at(t)
            lx, ly = self.to_local_fn(pos[0], pos[1], ax, ay, ayaw)
            return float(lx), float(ly), float(wrap_angle(yaw - ayaw))

        times_h = np.array([T - (H - 1 - i) * P for i in range(H)], dtype=np.float64)
        times_f = np.array([T + (k + 1) * P for k in range(F)], dtype=np.float64)
        history = np.array([row(t) for t in times_h], dtype=np.float32)
        future = np.array([row(t) for t in times_f], dtype=np.float32)
        if not (np.isfinite(history).all() and np.isfinite(future).all()):
            self.drops['non-finite pose values'] += 1
            return

        self.sink({
            'trajectory_history': history,
            'crop': np.asarray(a['crop']),
            'future_target': future,
            'agent_id': 'agent1',
            'sample_period_sec': P,
            'wall_time': time.time(),
            'schema': 'time_aligned_v2',
            'anchor_time_sim': float(T),
            'grid_stamp_sim': float(a['t_g']),
            'lookup_offset_sec': float(self.off),
            'history_times': times_h,
            'future_times': times_f,
            'robot_pose_odom': np.array(a['robot'], dtype=np.float64),
            'ped_anchor_odom': np.array(a['ped'], dtype=np.float64),
            'max_pose_gap_sec': float(gap),
        })
        self.n_saved += 1

    def status(self):
        drops = ', '.join(f'{n}x {r}' for r, n in self.drops.most_common(4)) or 'none'
        return (f'{self.n_saved} samples saved, {len(self.pending)} waiting for future poses, '
                f'{self.n_anchors} anchors started; dropped: {drops}')


# =============================================================================
# ROS wiring
# =============================================================================
def run_node(output_dir: Path):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid, Odometry
    from geometry_msgs.msg import PoseArray

    import build_occupancy_grid as bog
    from extract_pedestrian_crop import extract_pedestrian_crop, base_to_local
    from hunav_config import yaw_from_quaternion, ensure_single_instance  # noqa: F401

    class DatasetRecorder(Node):
        def __init__(self):
            super().__init__('dataset_recorder')
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.sampler = AlignedSampler(
                extract_pedestrian_crop, base_to_local, self._write,
                log=lambda level, msg: getattr(self.get_logger(), level)(msg, throttle_duration_sec=5.0))

            self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped_cb, 10)
            self.create_subscription(OccupancyGrid, '/occupancy_grid/base', self._grid_cb, 10)
            self.create_timer(10.0, self._status)   # wall-clock timer: status line only

            H, F, P = TRAJECTORY_HISTORY_LENGTH, FUTURE_HORIZON_LENGTH, DATASET_SAMPLE_PERIOD_SEC
            self.get_logger().info(
                f'dataset_recorder (time-aligned) started. Writing to {self.output_dir} '
                f'(run id {self.run_id}). history={H}, future={F}, period={P:.2f} s of SIMULATION '
                f'time, anchor = grid stamp {POSE_LOOKUP_OFFSET_SEC:+.2f} s. First sample after '
                f'~{(H - 1) * P + F * P:.1f} s of simulation time.')

        def _odom_cb(self, msg):
            p = msg.pose.pose.position
            self.sampler.on_odom(stamp_to_sec(msg.header.stamp), p.x, p.y,
                                 yaw_from_quaternion(msg.pose.pose.orientation))

        def _ped_cb(self, msg):
            if not msg.poses:
                return
            pose = msg.poses[0]  # single-agent, matches every other assumption in this codebase
            yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
            self.sampler.on_ped(stamp_to_sec(msg.header.stamp), pose.position.x, pose.position.y, yaw)

        def _grid_cb(self, msg):
            if msg.info.width != bog.GRID_N or msg.info.height != bog.GRID_N:
                self.get_logger().warn(
                    f'received base grid is {msg.info.width}x{msg.info.height}, '
                    f'expected {bog.GRID_N}x{bog.GRID_N} -- ignoring this message.',
                    throttle_duration_sec=5.0)
                return
            self.sampler.on_grid(
                stamp_to_sec(msg.header.stamp),
                lambda m=msg: np.array(m.data, dtype=np.int8).reshape((m.info.height, m.info.width)))

        def _write(self, sample):
            out_path = self.output_dir / f'sample_{self.run_id}_{self.sampler.n_saved:06d}.npz'
            sample = dict(sample)
            sample['crop'] = sample['crop'].astype(np.int8)
            np.savez(out_path, **sample)
            n = self.sampler.n_saved + 1
            if n == 1 or n % 20 == 0:
                self.get_logger().info(f'{n} samples saved -> {self.output_dir}')

        def _status(self):
            self.get_logger().info(self.sampler.status())

    ensure_single_instance('dataset_recorder')
    rclpy.init()
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Stopped. {node.sampler.status()}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    if len(sys.argv) == 2:
        output_dir = Path(sys.argv[1]).expanduser().resolve()
    run_node(output_dir)


if __name__ == '__main__':
    main()