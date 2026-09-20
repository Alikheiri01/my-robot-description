#!/usr/bin/env python3
"""
Bridges hunav_agent_manager's REAL Social Force Model computation to a plain
Gazebo MODEL (not an SDF <actor>), entirely bypassing HuNavSystemPluginIGN /
hunav_gazebo_world_generator's Gazebo-side actor-insertion pipeline -- which
hits a confirmed upstream Ignition Gazebo rendering bug (same class as
gazebosim/gz-sim#1489: Ogre::ItemIdentityException on actor
double-registration, once depth-camera sensors trigger a second render
pass). Plain models have already been shown immune to that whole bug class
in this project's own testing -- only <actor> elements throw; models just
log a harmless duplicate-registration warning and survive.

Reuses hunav_gazebo_world_generator PURELY for its /get_agents service
(parses hunav_loader's YAML into a ready-made hunav_msgs/Agents message,
so this node doesn't need its own YAML parser) -- its own
Gazebo-world-generation output (generatedWorld.sdf) is never loaded; Gazebo
instead loads the stock empty.sdf, same as the original robot-only launch,
with the robot AND a plain "pedestrian_standin" box model spawned into it
directly.

Mirrors HuNavSystemPluginIGN::PreUpdate()'s own logic (read directly from
its C++ source) in plain ROS2/Python:
  1. Read the robot's current pose (from /odom -- see CAVEAT below)
  2. Carry forward each tracked pedestrian's last computed state
  3. Call /compute_agents with both, timestamped with real wall-clock time
     (required -- see the project's own documented /compute_agents gotcha)
  4. Move the pedestrian_standin model to the returned pose via Gazebo's
     native /world/<world>/set_pose service, shelled out to the `ign
     service` CLI (confirmed working manually in this environment for the
     position field; ros_gz_bridge does not expose arbitrary Ignition
     services to ROS2 in this version)
  5. Store the response as next cycle's carried-forward state

  (Step "publish /people" deliberately removed -- see ROOT-CAUSE FIX below.
  Phase C's live occupancy-grid exclusion zone and heading_smoother.py both
  still get /people, just from hunav_agent_manager's own built-in publisher
  instead of a second copy from here.)

CAVEAT -- robot pose source, CONFIRMED BUG NOW FIXED: /odom does NOT report
true world pose -- confirmed empirically from a live run, where the first
/compute_agents call logged the robot's yaw as ~0 despite its known true
spawn yaw of 0.7854 rad. DiffDrive's simulated odometry starts at (0,0,0)
regardless of the robot's actual world spawn pose. Fixed below by composing
odom's own relative motion with the robot's known true spawn transform
(SPAWN_X/SPAWN_Y/SPAWN_YAW, imported from hunav_config.py -- the single
shared source of truth for this value now, no longer a locally duplicated
constant).

CAVEAT -- update rate: default 2 Hz, deliberately conservative. Each cycle
shells out to `ign service` (a real subprocess spawn), which has enough
overhead that a full 30Hz loop is not expected to keep up reliably. Fine
for a stationary or slow-moving agent; revisit with native ign-transport
Python bindings (avoiding subprocess spawn entirely) if a faster, smoother
update rate is needed later for realistic pedestrian motion.

CONFIRMED (previously listed here as untested): the `orientation`
quaternion field on the set_pose request works correctly -- verified
manually with four sequential set_pose calls, each producing a visible
rotation in Gazebo.

DIAGNOSTIC MOVE/FREEZE DUTY CYCLE -- added to debug a reported RViz
position/orientation offset between the live pedestrian pose and the
obstacle cluster the depth-camera pipeline derives from it. When
PAUSE_PHASE_ENABLED (in hunav_config.py) is True, _tick() alternates
between a MOVE phase (calls /compute_agents exactly as before) and a
FREEZE phase (skips the call entirely, so the model's Gazebo pose and
everything downstream hold their exact last value). CONCLUDED: the offset
persisted even during a freeze, ruling out latency -- it's a separate,
still-open issue in the depth-camera pipeline (degrades with range and at
the edges of the FOV, per direct observation), deferred along with the
rest of the camera/detection work. Left enabled as a general debugging
aid; set PAUSE_PHASE_ENABLED = False in hunav_config.py once it's no
longer needed.

ROOT-CAUSE FIX -- duplicate /people publisher (orientation bug, RESOLVED):
this node used to ALSO publish /people itself (people_msgs/People), on top
of hunav_agent_manager's own built-in publisher (BTnode::publish_people in
hunav_sim/hunav_agent_manager/src/bt_node.cpp, gated by pub_people_, which
defaults to true and fires as a side effect of every compute_agents call).
Confirmed via `ros2 topic info /people --verbose` showing exactly one
publisher (hunav_agent_manager) plus a source-level grep of bt_node.cpp --
both were firing per compute_agents cycle, both carrying the exact same
position/velocity (same underlying computed agent), which is why the
duplicate messages had identical positions but ours alone set
frame_id='map' (agent_manager's used whatever frame_id its own internal
Agents message carried, evidently empty). Every duplicated /people message
was re-entering heading_smoother.py's rolling velocity window, silently
corrupting the smoothed-heading average (a real value getting pushed into
the window twice, prematurely evicting an older, different sample) --
this is what caused the pedestrian orientation to be right in some frames
and wrong in others with no visible pattern. Fix: this node no longer
publishes /people at all; hunav_agent_manager's own copy is the single
source of truth for it now.
"""
import math
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from hunav_msgs.srv import GetAgents, ComputeAgents

from hunav_config import (
    WORLD_NAME, MODEL_NAME, BRIDGE_UPDATE_RATE_HZ as UPDATE_RATE_HZ,
    SET_POSE_TIMEOUT_MS, STANDIN_Z, odom_to_world,
    PAUSE_PHASE_ENABLED, PAUSE_PHASE_MOVE_SEC, PAUSE_PHASE_FREEZE_SEC,
)



def yaw_from_quaternion(q) -> float:
    """Standard quaternion -> yaw (Z-axis Euler) extraction."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float):
    """Returns (x, y, z, w) for a pure yaw rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class HunavModelBridge(Node):
    def __init__(self):
        super().__init__('hunav_model_bridge')

        self.robot_pose = None  # (x, y, yaw), populated from /odom
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)

        self.get_agents_client = self.create_client(GetAgents, 'get_agents')
        self.compute_agents_client = self.create_client(ComputeAgents, 'compute_agents')
        # No /people publisher here -- hunav_agent_manager already publishes it
        # natively (see ROOT-CAUSE FIX in the module docstring). Publishing it
        # again from here was corrupting heading_smoother.py's rolling window.

        self.current_agents = None  # hunav_msgs/Agents, carried forward each cycle
        self.timer = None  # started only once initialize_agents() has succeeded

        # Diagnostic move/freeze duty cycle -- see module docstring.
        self.phase = 'move'
        self.phase_started_at = None  # set on first _tick(), once the clock is live

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.robot_pose = odom_to_world(p.x, p.y, odom_yaw)

    def initialize_agents(self) -> bool:
        """
        Call exactly once, synchronously, BEFORE rclpy.spin() starts --
        spin_until_future_complete is not safe to call from inside a
        callback that's already running inside spin() (reentrancy risk),
        so this must happen in main() before the node starts spinning.
        """
        self.get_logger().info('Waiting for /get_agents service...')
        if not self.get_agents_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error('/get_agents service never became available.')
            return False
        req = GetAgents.Request()
        req.empty = 0
        future = self.get_agents_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            self.get_logger().error('Call to /get_agents did not complete.')
            return False
        self.current_agents = future.result().agents
        self.get_logger().info(
            f'Initialized {len(self.current_agents.agents)} agent(s) from /get_agents: '
            f'{[a.name for a in self.current_agents.agents]}'
        )
        return True

    def start(self):
        """Call once, after initialize_agents() succeeds, right before rclpy.spin()."""
        self.timer = self.create_timer(1.0 / UPDATE_RATE_HZ, self._tick)
        self.get_logger().info(
            f'hunav_model_bridge running. Moving model "{MODEL_NAME}" in world '
            f'"{WORLD_NAME}" at {UPDATE_RATE_HZ} Hz via real hunav_agent_manager '
            f'SFM computation, bypassing the Gazebo-actor pipeline entirely.'
        )

    def _set_model_pose(self, x: float, y: float, yaw: float):
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        req = (f'name: "{MODEL_NAME}" '
               f'position: {{x: {x:.4f}, y: {y:.4f}, z: {STANDIN_Z}}} '
               f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}')
        try:
            result = subprocess.run(
                ['ign', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
                 '--reqtype', 'ignition.msgs.Pose',
                 '--reptype', 'ignition.msgs.Boolean',
                 '--timeout', str(SET_POSE_TIMEOUT_MS),
                 '--req', req],
                capture_output=True, text=True, timeout=2.0,
            )
            if 'true' not in result.stdout:
                self.get_logger().warn(
                    f'set_pose call did not report success: stdout={result.stdout!r} '
                    f'stderr={result.stderr!r}', throttle_duration_sec=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn('set_pose call timed out', throttle_duration_sec=5.0)

    def _phase_elapsed_sec(self, now) -> float:
        return (now - self.phase_started_at).nanoseconds / 1e9

    def _update_phase(self, now) -> bool:
        """
        Advances the move/freeze duty cycle and returns True if this tick
        should proceed to call /compute_agents, False if it should be
        skipped (frozen). No-op (always returns True) when
        PAUSE_PHASE_ENABLED is False, so this whole feature is a single
        config flag away from behaving exactly as before.
        """
        if not PAUSE_PHASE_ENABLED:
            return True

        if self.phase_started_at is None:
            self.phase_started_at = now
            return True  # first tick: proceed, and start the MOVE phase clock

        elapsed = self._phase_elapsed_sec(now)

        if self.phase == 'move':
            if elapsed >= PAUSE_PHASE_MOVE_SEC:
                self.phase = 'freeze'
                self.phase_started_at = now
                self.get_logger().info(
                    f'Pause-phase diagnostic: FREEZING for {PAUSE_PHASE_FREEZE_SEC:.1f}s -- '
                    f'watch whether the RViz arrow and the depth obstacle converge.'
                )
                return False
            return True

        # self.phase == 'freeze'
        if elapsed >= PAUSE_PHASE_FREEZE_SEC:
            self.phase = 'move'
            self.phase_started_at = now
            self.get_logger().info(
                f'Pause-phase diagnostic: MOVING for {PAUSE_PHASE_MOVE_SEC:.1f}s.'
            )
            return True
        return False

    def _tick(self):
        if self.robot_pose is None:
            self.get_logger().warn('No /odom received yet, skipping cycle.', throttle_duration_sec=5.0)
            return
        if not self.compute_agents_client.service_is_ready():
            self.get_logger().warn('/compute_agents not ready, skipping cycle.', throttle_duration_sec=5.0)
            return

        now = self.get_clock().now()
        if not self._update_phase(now):
            return  # FREEZE phase: skip the /compute_agents call entirely,
                     # leaving the model's Gazebo pose, /people and
                     # /people_smoothed_pose all exactly where they were.

        rx, ry, ryaw = self.robot_pose
        qx, qy, qz, qw = yaw_to_quaternion(ryaw)

        req = ComputeAgents.Request()
        req.robot.position.position.x = rx
        req.robot.position.position.y = ry
        req.robot.position.orientation.x = qx
        req.robot.position.orientation.y = qy
        req.robot.position.orientation.z = qz
        req.robot.position.orientation.w = qw
        req.robot.yaw = ryaw

        self.current_agents.header.stamp = self.get_clock().now().to_msg()
        req.current_agents = self.current_agents

        future = self.compute_agents_client.call_async(req)
        future.add_done_callback(self._on_compute_agents_response)

    def _on_compute_agents_response(self, future):
        result = future.result()
        if result is None:
            self.get_logger().warn('/compute_agents call failed.', throttle_duration_sec=5.0)
            return
        self.current_agents = result.updated_agents
        if not self.current_agents.agents:
            return
        # Single-agent case for now -- move the one pedestrian stand-in.
        a = self.current_agents.agents[0]
        self._set_model_pose(a.position.position.x, a.position.position.y, a.yaw)
        # /people is published natively by hunav_agent_manager itself as a
        # side effect of the compute_agents call above -- no need (and no
        # longer safe, see ROOT-CAUSE FIX) to also publish it from here.


def main():
    from hunav_config import ensure_single_instance
    ensure_single_instance('hunav_model_bridge')

    rclpy.init()
    node = HunavModelBridge()
    if not node.initialize_agents():
        node.get_logger().error('Failed to initialize agents from /get_agents. Exiting.')
        node.destroy_node()
        rclpy.shutdown()
        return
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()