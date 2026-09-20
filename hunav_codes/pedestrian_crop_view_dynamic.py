#!/usr/bin/env python3
"""
Phase D crop visualizer -- LIVE counterpart to pedestrian_crop_view.py.

pedestrian_crop_view.py took a SYNTHETIC pedestrian pose from the command
line (a stand-in for a not-yet-existing tracker, the same category of
simplification as check_occupancy_grid_accuracy.py's --cube-pos), computed
one base_link transform at startup, and stayed correct only as long as
nothing moved. Now that HuNav is fully integrated and heading_smoother.py
publishes a real, continuously-updating, correctly-oriented pedestrian pose,
that whole synthetic/one-shot design is obsolete for anything actually
tracking the live scenario -- this is the real live viewer.

Deliberately a SEPARATE file, not an edit to pedestrian_crop_view.py -- same
reasoning as build_occupancy_grid_dynamic.py's own split from
build_occupancy_grid.py: the CLI/synthetic version stays untouched as a
known-good, reproducible manual test tool (useful any time you want to feed
an exact, hand-picked pose rather than whatever the live sim happens to be
doing), while this file is the one actually meant to be run alongside the
full launch.

LIVE INPUTS
------------------------------------------------------------------
- /odom (nav_msgs/Odometry) -- robot's own pose. Used RAW, in the 'odom'
  frame, with NO SPAWN_X/Y/YAW world-frame correction. That correction
  exists only for things that must be physically true in Gazebo's absolute
  world frame (the SFM computation, set_pose) -- see hunav_config.py's
  odom_to_world()/world_to_odom() docstrings for the full story. Everything
  in THIS script -- the crop, the base grid it's cut from, the robot -- all
  already lives in the robot's own local 'odom'/base_link frame, and so
  does /people_smoothed_pose after heading_smoother.py's own frame fix. So
  the transform here is a plain, direct odom-frame-to-odom-frame lookup:
  no world-frame conversion needed or wanted.
- /people_smoothed_pose (geometry_msgs/PoseArray, from heading_smoother.py)
  -- the real tracked pedestrian's live position + SMOOTHED heading, in
  that same odom frame. Single-agent for now (poses[0]), matching every
  other single-agent assumption already in this codebase.
- /occupancy_grid/base (nav_msgs/OccupancyGrid, from
  build_occupancy_grid_dynamic.py) -- same topic pedestrian_crop_view.py
  already consumes; unchanged wire format, so no downstream change needed
  there.

WHY THIS NEEDS A PAUSE, NOT JUST A LIVE FEED
------------------------------------------------------------------
The user's own stated problem with a naive live view: the robot and
pedestrian are both in continuous motion, so a purely live window is hard
to actually LOOK at -- by the time you register what's on screen, three
new frames have already replaced it. Fix: press SPACE in the matplotlib
window to freeze it. While paused, incoming /occupancy_grid/base messages
are ignored entirely (no crop recompute, no redraw, no republish) -- the
crop, the title, and the /occupancy_grid/pedestrian_crop topic all hold
their exact last value until you press SPACE again. This is deliberately
independent of hunav_model_bridge.py's own PAUSE_PHASE_ENABLED move/freeze
diagnostic (which freezes the PEDESTRIAN's motion at the source) -- this
one freezes the VIEW, works regardless of whether the sim itself is
freezing anything, and costs nothing to leave on permanently.

STALENESS
------------------------------------------------------------------
If /people_smoothed_pose hasn't delivered anything recently, the crop is
NOT recomputed with a stale pose -- displays a clear "no live pedestrian
data" placeholder instead of silently drawing a wrong/frozen-elsewhere
crop. The threshold is deliberately set well above
hunav_config.py's own PAUSE_PHASE_FREEZE_SEC, so hunav_model_bridge.py's
own diagnostic freeze (an EXPECTED several-second gap in /people, not a
dead pipeline) never gets misreported as staleness here.
"""
import math
import sys
import time
from pathlib import Path

# build_occupancy_grid.py and extract_pedestrian_crop.py predate the HuNav
# work and live in the sibling 'scene codes' folder (NOT 'codes' -- that's a
# separate, third folder in my_robot_description/), never moved during the
# hunav_codes/ consolidation. Only python running a script directly
# auto-adds THAT script's own directory to sys.path, not siblings, so
# without this, `import build_occupancy_grid` fails with ModuleNotFoundError
# for anyone running this file from hunav_codes/. Add that sibling directory
# explicitly so both live side by side without duplicating either file into
# hunav_codes/ (which would reintroduce exactly the multiple-copies
# fragility this project has spent real effort eliminating everywhere else
# -- see hunav_config.py's own history).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scene codes'))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseArray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import build_occupancy_grid as bog
from extract_pedestrian_crop import (
    extract_pedestrian_crop, CROP_FORWARD, CROP_BEHIND, CROP_SIDE, CROP_ROWS, CROP_COLS, CROP_RESOLUTION)
from hunav_config import yaw_from_quaternion, PAUSE_PHASE_FREEZE_SEC

# Comfortably above PAUSE_PHASE_FREEZE_SEC so the bridge's own intentional
# freeze diagnostic is never mistaken for a dead /people_smoothed_pose feed.
STALE_THRESHOLD_SEC = PAUSE_PHASE_FREEZE_SEC + 3.0

# -1 unknown, 0 free, 100 occupied -> remapped to 0/1/2 purely for the colormap
_CMAP = ListedColormap(['#b0b0b0', '#ffffff', '#1a1a1a'])  # unknown=grey, free=white, occupied=black
_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _CMAP.N)


def world_pose_to_base_link(x, y, yaw, robot_pos, robot_yaw):
    """
    Same rotation convention as pedestrian_crop_view.py's own version and
    check_occupancy_grid_accuracy.py's cube-footprint transform
    (R_robot.T @ (point - robot_pos)) -- "world" here just means "whatever
    common frame both poses share" (odom, in this script's case; see the
    module docstring for why no SPAWN correction belongs here).
    """
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    dx, dy = x - robot_pos[0], y - robot_pos[1]
    base_x = c * dx + s * dy
    base_y = -s * dx + c * dy
    base_yaw = yaw - robot_yaw
    return base_x, base_y, base_yaw


def to_display(crop: np.ndarray) -> np.ndarray:
    disp = np.zeros_like(crop, dtype=np.int8)
    disp[crop == bog.UNKNOWN] = 0
    disp[crop == bog.FREE] = 1
    disp[crop == bog.OCCUPIED] = 2
    return disp


class PedestrianCropViewLive(Node):
    def __init__(self):
        super().__init__('pedestrian_crop_view_dynamic')

        self.robot_pose = None       # (x, y, yaw) in odom frame, from /odom
        self.ped_pose = None         # (x, y, yaw) in odom frame, from /people_smoothed_pose
        self.ped_last_seen = None    # time.monotonic() of the last non-empty PoseArray
        self.paused = False

        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped_cb, 10)
        self.create_subscription(OccupancyGrid, '/occupancy_grid/base', self._grid_cb, 10)
        self.pub = self.create_publisher(OccupancyGrid, '/occupancy_grid/pedestrian_crop', 10)

        plt.ion()
        self.fig, self.ax = plt.subplots(num='Pedestrian-centric crop (LIVE)')
        init = np.zeros((CROP_ROWS, CROP_COLS), dtype=np.int8)
        self.im = self.ax.imshow(
            init, cmap=_CMAP, norm=_NORM, origin='lower',
            extent=[-CROP_BEHIND, CROP_FORWARD, -CROP_SIDE, CROP_SIDE])
        # BoundaryNorm has no inverse mapping, so matplotlib's mouse-hover
        # cursor-data tooltip crashes trying to compute it -- disabling just
        # that tooltip, purely cosmetic, unrelated to the actual rendering.
        self.im.format_cursor_data = lambda data: ''
        self.ax.set_xlabel('meters ahead of pedestrian (heading-aligned)')
        self.ax.set_ylabel('meters left(+) / right(-) of pedestrian')
        self.title = self.ax.set_title('waiting for live data...', fontsize=9)
        self.status_text = self.ax.text(
            0.02, 0.98, 'LIVE', transform=self.ax.transAxes, fontsize=10,
            fontweight='bold', color='green', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.ax.plot(0, 0, 'r+', markersize=14, markeredgewidth=2)
        self.ax.annotate('', xy=(1.0, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='red', lw=2))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.get_logger().info(
            'Live pedestrian crop viewer started. Press SPACE in the plot window '
            'to pause/resume. Waiting for /odom, /people_smoothed_pose and '
            '/occupancy_grid/base...'
        )

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            state = 'PAUSED (press space to resume)' if self.paused else 'LIVE'
            color = 'red' if self.paused else 'green'
            self.status_text.set_text(state)
            self.status_text.set_color(color)
            self.fig.canvas.draw_idle()
            self.get_logger().info(f'View {"paused" if self.paused else "resumed"}.')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.robot_pose = (p.x, p.y, yaw)

    def _ped_cb(self, msg: PoseArray):
        if not msg.poses:
            return  # no agent currently tracked -- leave last known pose in
                     # place, but ped_last_seen is NOT refreshed, so
                     # staleness detection still kicks in correctly below.
        pose = msg.poses[0]  # single-agent, matches every other assumption in this codebase
        qz, qw = pose.orientation.z, pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)  # pure-yaw quaternion -> angle, exact inverse of yaw_to_quaternion()
        self.ped_pose = (pose.position.x, pose.position.y, yaw)
        self.ped_last_seen = time.monotonic()

    def _grid_cb(self, msg: OccupancyGrid):
        if self.paused:
            return  # frozen: skip recompute, redraw, AND republish entirely

        if self.robot_pose is None:
            self.get_logger().warn('No /odom yet, skipping frame.', throttle_duration_sec=5.0)
            return
        if self.ped_pose is None or self.ped_last_seen is None:
            self._show_placeholder('No pedestrian tracked yet (waiting for /people_smoothed_pose)')
            return
        if time.monotonic() - self.ped_last_seen > STALE_THRESHOLD_SEC:
            self._show_placeholder(
                f'/people_smoothed_pose stale (>{STALE_THRESHOLD_SEC:.0f}s) -- '
                f'is the full sim actually running?')
            return

        if msg.info.width != bog.GRID_N or msg.info.height != bog.GRID_N:
            self.get_logger().warn(
                f'received base grid is {msg.info.width}x{msg.info.height}, '
                f'expected {bog.GRID_N}x{bog.GRID_N} -- skipping this frame.')
            return

        base_grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        ped_x, ped_y, ped_heading = world_pose_to_base_link(
            self.ped_pose[0], self.ped_pose[1], self.ped_pose[2],
            self.robot_pose, self.robot_pose[2])

        crop = extract_pedestrian_crop(base_grid, ped_x, ped_y, ped_heading)

        self.im.set_data(to_display(crop))
        self.title.set_text(
            f'ped (odom): x={self.ped_pose[0]:.2f} y={self.ped_pose[1]:.2f} '
            f'yaw={np.degrees(self.ped_pose[2]):.1f}deg\n'
            f'ped (base_link): x={ped_x:.2f}  y={ped_y:.2f}  heading={np.degrees(ped_heading):.1f} deg'
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.publish_crop(crop, msg.header, ped_x, ped_y, ped_heading)

    def _show_placeholder(self, message: str):
        self.title.set_text(message)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.get_logger().warn(message, throttle_duration_sec=5.0)

    def publish_crop(self, crop: np.ndarray, header, ped_x, ped_y, ped_heading):
        out = OccupancyGrid()
        out.header = header
        out.header.frame_id = (
            f'pedestrian_local;ped_base_link_x={ped_x:.6f}'
            f';ped_base_link_y={ped_y:.6f};ped_base_link_heading={ped_heading:.6f}'
        )
        out.info.resolution = float(CROP_RESOLUTION)
        out.info.width = CROP_COLS
        out.info.height = CROP_ROWS
        out.info.origin.position.x = -CROP_BEHIND
        out.info.origin.position.y = -CROP_SIDE
        out.info.origin.position.z = 0.0
        out.info.origin.orientation.w = 1.0
        out.data = crop.flatten(order='C').tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = PedestrianCropViewLive()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()