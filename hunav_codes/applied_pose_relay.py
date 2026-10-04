#!/usr/bin/env python3
"""
applied_pose_relay.py -- the pedestrian's TRUE rendered pose, for everything
that needs to line up with the depth camera.

WHY THIS EXISTS (2026-10-01)
The depth camera shows the pedestrian where the actor plugin
(HuNavActorDriver) has put it. Until now every node took the pedestrian's pose
from HuNav's /people, which is the COMMAND the plugin is given. A command
reaches the rendered actor only after a delay, and that delay changed from
session to session (validate_time_alignment.py measured -0.10, -0.24, -0.29
and -0.06 s), so a fixed lookup offset (POSE_LOOKUP_OFFSET_SEC) was never
reliable: a recording could be 0.2 m out of step with its crop without any
check noticing.

The plugin knows exactly where it moved the actor and when. It publishes that
as /model/<actor>/applied_pose (Pose_V, ~30 Hz of SIMULATION time, header
stamp = simulation time of the step the actor was moved in -- the instant the
cameras render). The launch file bridges it to ROS as a geometry_msgs/PoseArray
under the same name. This node turns that stream into the topic every other
node already reads:

    /model/<actor>/applied_pose   (PoseArray, world frame, sim stamp, ~30 Hz)
        --> applied_pose_relay.py -->
    /people_smoothed_pose         (PoseArray, 'odom'-equivalent frame, same stamp)

so the grid builder, crop viewer, recorder and validator look the pedestrian
up at a camera's own stamp and need no timing correction.

THIS REPLACES heading_smoother.py when the actor plugin is used. Do NOT run
both: they publish the same topic. They share a lock name, so the second one
you start refuses to run.

HEADING
The orientation of the applied pose is the MESH yaw, which the plugin
interpolates between HuNav yaws (it sweeps through the turn). The crop needs
the DIRECTION OF WALKING instead, so the heading published here is computed
from the applied positions: the direction of the displacement over the last
APPLIED_HEADING_WINDOW_SEC (0.1 s), held while the speed is below
MIN_SPEED_FOR_HEADING_UPDATE. The applied path is exact, so no averaging is
needed, and a 180 degree reversal shows up within 0.1 s. The first
heading, before the actor has moved, is the mesh yaw.
The yaw published with a position is the heading at that sample, so look it up
with time_sync.StampedPoseHistory.step_yaw_at() as before.

FRAME
The applied pose is in the Gazebo WORLD frame. It is converted with
hunav_config.world_to_odom() exactly like heading_smoother.py did, and
published under frame_id 'map' (a static identity link to 'odom').

SAFETY
If the incoming message carries no stamp (zero), nothing is published and an
error is printed: a pose without the simulation time of its step would bring
the old problem back silently. That means the plugin was not rebuilt with the
stamp, or the bridge dropped it.
"""
import math
from collections import deque

from hunav_config import (
    APPLIED_POSE_TOPIC, APPLIED_HEADING_WINDOW_SEC, MIN_SPEED_FOR_HEADING_UPDATE, world_to_odom)

CLOCK_JUMP_SEC = 1.0       # a stamp this far BEHIND the last one = the simulation restarted
KEEP_SEC = 1.0             # position samples kept per agent


class MotionHeading:
    """Per-agent state: recent positions -> direction of walking at the newest one."""

    def __init__(self, window_sec=APPLIED_HEADING_WINDOW_SEC, min_speed=MIN_SPEED_FOR_HEADING_UPDATE):
        self.window = window_sec
        self.min_speed = min_speed
        self.samples = deque()      # (t, x, y)
        self.heading = None

    def update(self, t, x, y, mesh_yaw):
        """Add the pose at stamp t; returns the walking direction there, or None if the stamp is not newer."""
        if self.samples:
            last_t = self.samples[-1][0]
            if t < last_t - CLOCK_JUMP_SEC:      # simulation restarted
                self.samples.clear()
                self.heading = None
            elif t <= last_t:                     # duplicate / out of order
                return None
        self.samples.append((t, x, y))
        while len(self.samples) > 1 and t - self.samples[0][0] > KEEP_SEC:
            self.samples.popleft()

        # oldest sample that is still inside the window
        ref = None
        for s in self.samples:
            if t - s[0] <= self.window + 1e-9:
                ref = s
                break
        if ref is not None and ref[0] < t - 1e-9:
            dt = t - ref[0]
            vx, vy = (x - ref[1]) / dt, (y - ref[2]) / dt
            if math.hypot(vx, vy) >= self.min_speed:
                self.heading = math.atan2(vy, vx)
        if self.heading is None:
            return mesh_yaw        # not moving yet: nothing better known (not stored, so motion still wins later)
        return self.heading


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def run_node():
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseArray, Pose
    from hunav_config import ensure_single_instance
    from time_sync import stamp_to_sec

    class AppliedPoseRelay(Node):
        def __init__(self):
            super().__init__('applied_pose_relay')
            self.trackers = {}
            self.n_in = 0
            self.n_out = 0
            self.pub = self.create_publisher(PoseArray, '/people_smoothed_pose', 10)
            self.create_subscription(PoseArray, APPLIED_POSE_TOPIC, self._cb, 10)
            self.create_timer(10.0, self._status)
            self.get_logger().info(
                f'applied_pose_relay started: {APPLIED_POSE_TOPIC} -> /people_smoothed_pose '
                f'(walking direction from the last {APPLIED_HEADING_WINDOW_SEC:.2f} s of positions, '
                f'held below {MIN_SPEED_FOR_HEADING_UPDATE} m/s). Use this INSTEAD of heading_smoother.py.')

        def _cb(self, msg):
            self.n_in += 1
            t = stamp_to_sec(msg.header.stamp)
            if t == 0.0:
                self.get_logger().error(
                    f'{APPLIED_POSE_TOPIC} has no timestamp: rebuild the HuNavActorDriver plugin '
                    f'(colcon build --packages-select hunav_actor_driver), restart Gazebo, and check with '
                    f'`ros2 topic echo {APPLIED_POSE_TOPIC} --once`. Nothing is published.',
                    throttle_duration_sec=5.0)
                return
            out = PoseArray()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = 'map'     # static identity map->odom link, as in heading_smoother.py
            for i, pose in enumerate(msg.poses):
                mesh_yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
                tr = self.trackers.setdefault(i, MotionHeading())
                yaw = tr.update(t, pose.position.x, pose.position.y, mesh_yaw)
                if yaw is None:
                    return              # this stamp was not newer than the last one
                ox, oy, oyaw = world_to_odom(pose.position.x, pose.position.y, yaw)
                p = Pose()
                p.position.x, p.position.y, p.position.z = ox, oy, 0.0
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = yaw_to_quaternion(oyaw)
                out.poses.append(p)
            if out.poses:
                self.pub.publish(out)
                self.n_out += 1

        def _status(self):
            if self.n_in == 0:
                self.get_logger().warn(
                    f'nothing received on {APPLIED_POSE_TOPIC} yet. Is the actor spawned, is the plugin '
                    f'the new build, and is the bridge entry for it in the launch file?')
            else:
                self.get_logger().info(f'{self.n_in} applied poses in, {self.n_out} published so far')

    ensure_single_instance('heading_smoother')    # same lock as heading_smoother.py: never both
    rclpy.init()
    node = AppliedPoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    run_node()