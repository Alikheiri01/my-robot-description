#!/usr/bin/env python3
"""
Live visualization of the pedestrian-centric crop, in its own matplotlib
window -- deliberately NOT rendered into the same RViz scene as the base
grid. Reason: the crop is heading-aligned, not robot-aligned, so it would
appear rotated relative to the base grid's own axis-aligned RViz display --
easy to mistake that *expected* rotation for a bug when eyeballed next to
the base grid. A standalone window, drawn in the pedestrian's own local
axes (meters ahead / meters left-right, pedestrian marked at the origin
with a forward-pointing arrow), avoids that confusion entirely.

Subscribes to /occupancy_grid/base (build_occupancy_grid.py), computes the
80x80 pedestrian-centric crop via extract_pedestrian_crop.py using a
SYNTHETIC pedestrian pose given on the command line -- no real tracked
pedestrian exists yet (Hunav isn't integrated), so this stands in for that
future tracker's output the same way check_occupancy_grid_accuracy.py
stands in for the cube's ground truth.

Usage (WORLD frame, read straight from Gazebo's Component Inspector --
same convention as check_occupancy_grid_accuracy.py's --cube-pos/--robot-pos,
not base_link-relative -- the world -> base_link transform happens inside
this script):

    python3 pedestrian_crop_view.py \
        --ped-pos X Y Z --ped-yaw YAW_RAD \
        --robot-pos X Y Z --robot-yaw YAW_RAD

Z components are accepted for CLI consistency with the other checker
scripts but unused, since this is a 2D top-down grid. --robot-pos/--robot-yaw
are the exact same values you'd already read for check_occupancy_grid_accuracy.py
-- reuse them directly.

CAVEAT -- this is a one-time snapshot transform, not a live TF lookup: the
pedestrian's position in base_link is computed ONCE at startup from the
--robot-pos/--robot-yaw you provide. If the robot moves after this script
starts, that transform goes stale (same category of simplification as the
synthetic pedestrian pose itself standing in for a not-yet-existing
tracker) -- keep the robot still while testing, or restart the script after
moving it.

Pick a pedestrian pose with a different origin AND a noticeably different
heading than the robot's own -- if pedestrian pose = robot pose, the crop
transform degenerates toward identity and won't exercise the rotation math
at all.
"""
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import build_occupancy_grid as bog
from extract_pedestrian_crop import (
    extract_pedestrian_crop, CROP_FORWARD, CROP_BEHIND, CROP_SIDE, CROP_ROWS, CROP_COLS, CROP_RESOLUTION)

# -1 unknown, 0 free, 100 occupied -> remapped to 0/1/2 purely for the colormap
_CMAP = ListedColormap(['#b0b0b0', '#ffffff', '#1a1a1a'])  # unknown=grey, free=white, occupied=black
_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _CMAP.N)


def world_pose_to_base_link(x, y, yaw, robot_pos, robot_yaw):
    """
    Transforms a world-frame (x, y, yaw) into base_link, same rotation
    convention already used for the cube footprint in
    check_occupancy_grid_accuracy.py (R_robot.T @ (world - robot_pos)).
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


class PedestrianCropView(Node):
    def __init__(self, ped_x, ped_y, ped_heading, world_ped_pose, world_robot_pose):
        super().__init__('pedestrian_crop_view')
        self.ped_x, self.ped_y, self.ped_heading = ped_x, ped_y, ped_heading
        self.world_ped_pose = world_ped_pose
        self.world_robot_pose = world_robot_pose
        self.pub = self.create_publisher(OccupancyGrid, '/occupancy_grid/pedestrian_crop', 10)
        self.create_subscription(OccupancyGrid, '/occupancy_grid/base', self.callback, 10)

        plt.ion()
        self.fig, self.ax = plt.subplots(num='Pedestrian-centric crop (separate from RViz)')
        init = np.zeros((CROP_ROWS, CROP_COLS), dtype=np.int8)
        self.im = self.ax.imshow(
            init, cmap=_CMAP, norm=_NORM, origin='lower',
            extent=[-CROP_BEHIND, CROP_FORWARD, -CROP_SIDE, CROP_SIDE])
        # BoundaryNorm (needed for clean discrete grey/white/black colors) has
        # no inverse mapping, so matplotlib's mouse-hover cursor-data tooltip
        # crashes trying to compute it. Disabling just that tooltip -- purely
        # cosmetic, unrelated to the actual image rendering.
        self.im.format_cursor_data = lambda data: ''
        self.ax.set_xlabel('meters ahead of pedestrian (heading-aligned)')
        self.ax.set_ylabel('meters left(+) / right(-) of pedestrian')
        self.ax.set_title(
            f'ped (world): x={world_ped_pose[0]:.2f} y={world_ped_pose[1]:.2f} yaw={np.degrees(world_ped_pose[2]):.1f}deg\n'
            f'ped (base_link): x={ped_x:.2f}  y={ped_y:.2f}  heading={np.degrees(ped_heading):.1f} deg',
            fontsize=9)
        # Fixed reference marks that never move regardless of what the crop
        # shows: the pedestrian's own position (always local origin by
        # construction) and a 1m arrow showing which way is "forward".
        self.ax.plot(0, 0, 'r+', markersize=14, markeredgewidth=2)
        self.ax.annotate('', xy=(1.0, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='red', lw=2))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.get_logger().info(
            f'Pedestrian crop viewer started. World ped pose: x={world_ped_pose[0]:.3f} '
            f'y={world_ped_pose[1]:.3f} yaw={world_ped_pose[2]:.3f} rad. Robot pose used for '
            f'transform: x={world_robot_pose[0]:.3f} y={world_robot_pose[1]:.3f} '
            f'yaw={world_robot_pose[2]:.3f} rad. Computed ped pose in base_link: '
            f'x={ped_x:.3f} y={ped_y:.3f} heading={ped_heading:.3f} rad '
            f'({np.degrees(ped_heading):.1f} deg). NOTE: this transform is a one-time '
            f'snapshot -- if the robot moves, restart this script.'
        )

    def callback(self, msg: OccupancyGrid):
        if msg.info.width != bog.GRID_N or msg.info.height != bog.GRID_N:
            self.get_logger().warn(
                f'received base grid is {msg.info.width}x{msg.info.height}, '
                f'expected {bog.GRID_N}x{bog.GRID_N} -- skipping this frame.')
            return
        base_grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        crop = extract_pedestrian_crop(base_grid, self.ped_x, self.ped_y, self.ped_heading)

        self.im.set_data(to_display(crop))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.publish_crop(crop, msg.header)

    def publish_crop(self, crop: np.ndarray, header):
        # NOTE: frame_id is repurposed to carry this run's exact pose values,
        # not a registered TF frame -- this lets check_pedestrian_crop_accuracy.py
        # detect a stale/mismatched viewer (e.g. still running with an old or
        # mistyped pose) instead of silently comparing against the wrong
        # scenario, which produces a confusing, misleading report with no
        # error at all. See that script's docstring for why this matters.
        out = OccupancyGrid()
        out.header = header
        out.header.frame_id = (
            'pedestrian_local'
            f';ped_world_x={self.world_ped_pose[0]:.6f}'
            f';ped_world_y={self.world_ped_pose[1]:.6f}'
            f';ped_world_yaw={self.world_ped_pose[2]:.6f}'
            f';robot_world_x={self.world_robot_pose[0]:.6f}'
            f';robot_world_y={self.world_robot_pose[1]:.6f}'
            f';robot_world_yaw={self.world_robot_pose[2]:.6f}'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--ped-pos', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
                         help='Synthetic pedestrian position in WORLD frame (Z unused, kept for '
                              'CLI consistency with the other checker scripts).')
    parser.add_argument('--ped-yaw', type=float, required=True,
                         help='Synthetic pedestrian heading in WORLD frame, radians.')
    parser.add_argument('--robot-pos', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
                         help='Robot position in WORLD frame -- same value you already read for '
                              'check_occupancy_grid_accuracy.py.')
    parser.add_argument('--robot-yaw', type=float, required=True,
                         help='Robot heading in WORLD frame, radians.')
    args = parser.parse_args()

    world_ped_pose = (args.ped_pos[0], args.ped_pos[1], args.ped_yaw)
    world_robot_pose = (args.robot_pos[0], args.robot_pos[1], args.robot_yaw)

    ped_x, ped_y, ped_heading = world_pose_to_base_link(
        args.ped_pos[0], args.ped_pos[1], args.ped_yaw, args.robot_pos, args.robot_yaw)

    rclpy.init()
    node = PedestrianCropView(ped_x, ped_y, ped_heading, world_ped_pose, world_robot_pose)
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