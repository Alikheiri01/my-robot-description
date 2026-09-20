#!/usr/bin/env python3
"""
Phase D, first piece: heading smoothing.

Subscribes to /people (raw, from hunav_model_bridge.py -- real SFM-computed
position and velocity, but per-frame instantaneous heading is noisy/jumpy
frame to frame even when the pedestrian's real path is smooth). Publishes a
SMOOTHED heading per tracked person, for use by anything that needs to
orient something relative to "which way is this pedestrian facing" (chiefly
extract_pedestrian_crop.py's local-frame construction) without every frame's
raw noise showing up as visible jitter/rotation in that downstream consumer.

ALGORITHM -- moving-average with low-speed hold (matches the original MVP
spec's own prescribed approach, not something invented fresh here):
  1. Maintain a short rolling window of each tracked person's recent
     (vx, vy) velocity readings (HEADING_SMOOTHING_WINDOW samples, from
     hunav_config.py).
  2. AVERAGE THE VELOCITY VECTORS THEMSELVES, then take atan2 of the
     result -- not an average of raw angles. Averaging angles directly is
     wrong whenever a heading crosses the +-180 degree wraparound boundary
     (e.g. two readings of +179 and -179 degrees average to 0, which is
     backwards); averaging the underlying vectors and taking atan2 of the
     result sidesteps this entirely and is the standard, correct technique.
  3. LOW-SPEED HOLD: when a person's most recent speed is below
     MIN_SPEED_FOR_HEADING_UPDATE, heading is ill-defined (atan2 of a
     near-zero vector is mostly noise) -- hold the last reliable smoothed
     heading rather than updating it. Falls back to the raw instantaneous
     heading only for a person's very first reading, before any history
     exists to smooth over.

OUTPUT -- publishes /people_smoothed_pose (geometry_msgs/PoseArray), one
Pose per currently-tracked person: position.x/y = that person's current
position, orientation = a quaternion built from the SMOOTHED yaw.
Deliberately a standard PoseArray rather than reusing /people's own
People/Person message (which has no orientation field at all) or the
Z-encodes-yaw convention discovered elsewhere in this project's history
(Phase A's own /people investigation) -- a real orientation field on a
standard message type is cleaner and less surprising for any future
consumer than repurposing a position field to carry something else.

FRAME CONVERSION (RESOLVED -- root cause of the RViz arrow/point-cloud
offset): /people's position/yaw comes from hunav_agent_manager's SFM
computation, which is anchored to the robot pose hunav_model_bridge.py
supplies it -- already converted to true Gazebo WORLD coordinates (via
hunav_config.py's odom_to_world()), because that's what set_pose and the
physics need to be correct. But RViz's own TF tree (the depth-camera point
cloud, the robot model, everything else drawn there) lives in the 'odom'
frame ('map' is just a static identity link to it, no real localization
stack here) -- and 'odom' itself starts at (0,0,0) with zero yaw
regardless of the robot's true SPAWN_YAW=0.7854 rad spawn pose (the
project's own long-documented odom bug). So a world-frame position
published verbatim under frame_id='map' is rotated/translated from
everything else in RViz by exactly that spawn transform -- not latency,
not camera noise, a genuine frame convention mismatch, confirmed by the
fact that build_occupancy_grid_dynamic.py's actual DATASET output was
never affected (it applies this same conversion to the robot's own /odom
consistently, so robot and pedestrian stay self-consistent regardless of
which absolute frame convention is used -- this bug was visualization-only).
Fixed by converting through hunav_config.py's world_to_odom() before
publishing, so the arrow lands in the same frame convention as everything
else RViz draws.

TESTING PLAN (proposed, not yet built as of this file): extend
pedestrian_crop_view.py to live-subscribe to /odom + /people_smoothed_pose
instead of taking a one-shot CLI snapshot, so the crop's orientation can be
watched directly during thesis_static_agent.yaml's natural ~180 degree
reversal at each goal -- the sharpest heading-change case this scenario
already produces, no new scenario needed. Would let HEADING_SMOOTHING_WINDOW
and MIN_SPEED_FOR_HEADING_UPDATE actually be tuned against something
visible, rather than left at their current starting values.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from people_msgs.msg import People

from hunav_config import HEADING_SMOOTHING_WINDOW, MIN_SPEED_FOR_HEADING_UPDATE, world_to_odom


def yaw_to_quaternion(yaw: float):
    """Returns (x, y, z, w) for a pure yaw rotation -- same convention used
    throughout this project (hunav_model_bridge.py, build_occupancy_grid_dynamic.py)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PersonHeadingTracker:
    """Per-agent rolling state -- one instance per tracked person name."""

    def __init__(self):
        self.velocity_window = []  # list of (vx, vy), most recent last
        self.last_smoothed_yaw = None  # None until we have a first real reading

    def update(self, vx: float, vy: float) -> float:
        speed = math.hypot(vx, vy)

        if speed < MIN_SPEED_FOR_HEADING_UPDATE:
            if self.last_smoothed_yaw is None:
                # No history yet and already below the speed threshold --
                # nothing reliable to hold, fall back to raw (likely ~0,
                # a person just spawned and hasn't moved yet).
                return math.atan2(vy, vx)
            return self.last_smoothed_yaw

        self.velocity_window.append((vx, vy))
        if len(self.velocity_window) > HEADING_SMOOTHING_WINDOW:
            self.velocity_window.pop(0)

        avg_vx = sum(v[0] for v in self.velocity_window) / len(self.velocity_window)
        avg_vy = sum(v[1] for v in self.velocity_window) / len(self.velocity_window)
        smoothed_yaw = math.atan2(avg_vy, avg_vx)

        self.last_smoothed_yaw = smoothed_yaw
        return smoothed_yaw


class HeadingSmoother(Node):
    def __init__(self):
        super().__init__('heading_smoother')

        self.trackers = {}  # person name -> PersonHeadingTracker

        self.pub = self.create_publisher(PoseArray, '/people_smoothed_pose', 10)
        self.create_subscription(People, '/people', self._people_cb, 10)

        self.get_logger().info(
            f'heading_smoother started. Window={HEADING_SMOOTHING_WINDOW} samples, '
            f'low-speed hold below {MIN_SPEED_FOR_HEADING_UPDATE} m/s. '
            f'Publishing smoothed headings on /people_smoothed_pose.'
        )

    def _people_cb(self, msg: People):
        out = PoseArray()
        out.header.stamp = msg.header.stamp
        # Deliberately NOT `out.header = msg.header`: /people's frame_id is
        # unreliable depending on which upstream publisher produced it --
        # hunav_agent_manager's own native /people publish (bt_node.cpp)
        # leaves it empty, which is not a resolvable TF frame and made RViz
        # silently refuse to draw every single pose here. World frame is
        # 'map' everywhere else in this project (see SPAWN_X/Y/YAW in
        # hunav_config.py), so enforce it explicitly rather than trust
        # whatever the input happened to carry.
        out.header.frame_id = 'map'  # assumes the standard no-SLAM setup: a static
        # identity map->odom link, so 'map' and 'odom' are numerically the same frame.
        # Poses below are converted to the odom-frame convention via world_to_odom()
        # (see FRAME CONVERSION note above) before being written into this message.

        for person in msg.people:
            if person.name not in self.trackers:
                self.trackers[person.name] = PersonHeadingTracker()
            tracker = self.trackers[person.name]

            smoothed_yaw = tracker.update(person.velocity.x, person.velocity.y)

            # /people's position + smoothed_yaw are in true Gazebo WORLD
            # frame (see FRAME CONVERSION note above) -- convert into the
            # 'odom'-equivalent frame RViz's own TF tree actually uses
            # before publishing, so this lands in the same place as
            # everything else drawn there.
            odom_x, odom_y, odom_yaw = world_to_odom(
                person.position.x, person.position.y, smoothed_yaw)

            pose = Pose()
            pose.position.x = odom_x
            pose.position.y = odom_y
            pose.position.z = 0.0
            qx, qy, qz, qw = yaw_to_quaternion(odom_yaw)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            out.poses.append(pose)

        self.pub.publish(out)


def main():
    from hunav_config import ensure_single_instance
    ensure_single_instance('heading_smoother')

    rclpy.init()
    node = HeadingSmoother()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()