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

NOTE (2026-09-21): the my_robot_hunav_actor_launch.launch.py variant spawns
a real animated SDF <actor> instead of a plain model, but ONLY after the
robot's depth-camera sensors have already triggered their one-time render
init -- avoiding the crash above by reordering rather than avoiding actors
entirely. This node's logic is unchanged either way; it only points at a
different MODEL_NAME (see hunav_config.py) and moves whichever entity that
name refers to via the same set_pose mechanism.

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
longer needed. (Set to False as of 2026-09-21 -- see hunav_config.py.)

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

ROOT-CAUSE FIX -- overlapping /compute_agents calls (RESOLVED 2026-09-21):
check_actor_motion.py's consistency logging (v2) caught a distinct bug on
the animated-actor launch: periodic samples where position and heading
were frozen at the previous value while velocity still read nonzero (a
self-contradictory state), plus at least one pair of messages arriving at
literally the same wall-clock instant with identical content. Root cause:
_tick() fired unconditionally on a fixed timer (every 1/UPDATE_RATE_HZ
seconds) with no check for whether the PREVIOUS /compute_agents call had
already returned. If any single round-trip took longer than one tick
period, a second request went out before the first resolved, both built
from the same stale self.current_agents snapshot -- producing exactly the
observed frozen/duplicate samples once both responses landed. Fixed by
adding self._compute_agents_pending, checked at the top of _tick() and
set/cleared around the async call -- see the guard below. This does not
change the SFM/pose logic at all, only prevents overlapping requests.

POSE DELIVERY MODES (2026-09-27) -- choose with --pose-mode:
  set_pose (default): one `ign service .../set_pose` CLI subprocess per
      cycle. Works for plain MODELS (the baseline pedestrian_standin launch),
      where Gazebo's Physics applies the command. For an SDF ACTOR, Gazebo
      NEVER applies set_pose on its own -- the HuNavActorDriver plugin
      (package hunav_actor_driver, loaded inside hunav_actor.sdf) does. Each
      subprocess also takes a large part of a 2 Hz cycle to land, which made
      the actor lag /people by 0-1 step (measured |A-C| alternating 0/0.55 m).
  topic: publish each pose as geometry_msgs/Pose on /model/<MODEL_NAME>/cmd_pose;
      ros_gz_bridge forwards it to the same ign-transport topic, where
      HuNavActorDriver picks it up within milliseconds. Actor launch only --
      nothing listens on that topic for a plain model.
--debug-raw logs every pose received from /compute_agents ([bridge-raw]).

CLOCK (2026-09-28): run with `--ros-args -p use_sim_time:=true` (the actor
launch does). Then the 2 Hz timer AND every /compute_agents request stamp use
Gazebo's simulation clock, so:
  - HuNav integrates the pedestrian over SIM seconds. On wall time it moved
    over real seconds while the world ran at ~0.32x real time (measured), so
    in simulation time the pedestrian walked ~3x faster than max_vel.
  - /people (stamped by hunav_agent_manager with our request stamp) and
    /people_smoothed_pose carry SIM stamps -- the same clock as the depth
    images/clouds/grids, so they can be matched by timestamp (latency_probe.py
    showed /people on WALL and everything else on SIM before this).
  - pausing Gazebo pauses HuNav too (no more catch-up jump on resume).
The old wall-clock behaviour is still the default if use_sim_time is unset.
"""
import argparse
import math
import os
import signal
import subprocess
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
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
    def __init__(self, pose_mode='set_pose', debug_raw=False):
        super().__init__('hunav_model_bridge')
        self.pose_mode = pose_mode
        self.debug_raw = debug_raw
        self.cmd_pose_pub = None
        if self.pose_mode == 'topic':
            self.cmd_pose_topic = f'/model/{MODEL_NAME}/cmd_pose'
            self.cmd_pose_pub = self.create_publisher(Pose, self.cmd_pose_topic, 10)

        self.robot_pose = None  # (x, y, yaw), populated from /odom
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)

        self.get_agents_client = self.create_client(GetAgents, 'get_agents')
        self.compute_agents_client = self.create_client(ComputeAgents, 'compute_agents')
        # No /people publisher here -- hunav_agent_manager already publishes it
        # natively (see ROOT-CAUSE FIX in the module docstring). Publishing it
        # again from here was corrupting heading_smoother.py's rolling window.

        self.current_agents = None  # hunav_msgs/Agents, carried forward each cycle
        self.timer = None  # started only once initialize_agents() has succeeded

        # Guards against overlapping /compute_agents calls -- without this,
        # if a response ever takes longer than one tick period (1/UPDATE_RATE_HZ),
        # the timer fires again before the prior request resolves, producing
        # two in-flight requests built from the same stale current_agents
        # snapshot. Confirmed as the cause of periodic frozen-but-nonzero-
        # velocity samples and near-simultaneous duplicate /people messages,
        # via check_actor_motion.py's consistency logging (2026-09-21).
        self._compute_agents_pending = False

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
            f'SFM computation, clock: '
            + ('SIM (/clock)' if self.get_parameter('use_sim_time').value else 'WALL')
            + '. Pose delivery: '
            + (f'topic {self.cmd_pose_topic}' if self.pose_mode == 'topic'
               else 'ign service set_pose (subprocess)')
        )

    def _set_model_pose(self, x: float, y: float, yaw: float):
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        if self.pose_mode == 'topic':
            msg = Pose()
            msg.position.x = x
            msg.position.y = y
            msg.position.z = STANDIN_Z
            msg.orientation.x = qx
            msg.orientation.y = qy
            msg.orientation.z = qz
            msg.orientation.w = qw
            self.cmd_pose_pub.publish(msg)
            return
        req = (f'name: "{MODEL_NAME}" '
               f'position: {{x: {x:.4f}, y: {y:.4f}, z: {STANDIN_Z}}} '
               f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}')
        cmd = ['ign', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
               '--reqtype', 'ignition.msgs.Pose',
               '--reptype', 'ignition.msgs.Boolean',
               '--timeout', str(SET_POSE_TIMEOUT_MS),
               '--req', req]
        # NOTE: plain subprocess.run(..., timeout=2.0) does NOT reliably
        # bound real wall-clock time here. On timeout, Python only kills the
        # immediate child PID -- if `ign service` (ignition-transport CLI)
        # leaves any descendant holding stdout/stderr open, communicate()
        # keeps blocking past the requested timeout waiting for pipe EOF.
        # Confirmed in this project: a single call froze the ENTIRE node
        # (single-threaded executor -- this call blocks the timer too) for
        # ~154s instead of the intended 2s, right around the same time
        # Gazebo itself went away. Fix: run in its own process GROUP and
        # kill the whole group on timeout, not just the one PID.
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                try:
                    stdout, stderr = proc.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    stdout, stderr = '', '(process group killed; pipes still would not drain)'
                self.get_logger().warn(
                    'set_pose call timed out -- process group killed. '
                    f'stderr={stderr!r}', throttle_duration_sec=5.0)
                return
            if 'true' not in stdout:
                self.get_logger().warn(
                    f'set_pose call did not report success: stdout={stdout!r} '
                    f'stderr={stderr!r}', throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().warn(f'set_pose call raised {type(e).__name__}: {e}',
                                    throttle_duration_sec=5.0)
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

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
        if self.get_clock().now().nanoseconds == 0:
            # use_sim_time is on but /clock has not arrived yet -- a zero stamp
            # would give HuNav a meaningless first time step.
            self.get_logger().warn('Waiting for /clock (use_sim_time is on).',
                                   throttle_duration_sec=5.0)
            return
        if self.robot_pose is None:
            self.get_logger().warn('No /odom received yet, skipping cycle.', throttle_duration_sec=5.0)
            return
        if not self.compute_agents_client.service_is_ready():
            self.get_logger().warn('/compute_agents not ready, skipping cycle.', throttle_duration_sec=5.0)
            return
        if self._compute_agents_pending:
            self.get_logger().warn(
                'Previous /compute_agents call has not returned yet -- '
                'skipping this tick instead of overlapping requests.',
                throttle_duration_sec=5.0)
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

        self._compute_agents_pending = True
        future = self.compute_agents_client.call_async(req)
        future.add_done_callback(self._on_compute_agents_response)

    def _on_compute_agents_response(self, future):
        self._compute_agents_pending = False
        result = future.result()
        if result is None:
            self.get_logger().warn('/compute_agents call failed.', throttle_duration_sec=5.0)
            return
        self.current_agents = result.updated_agents
        if not self.current_agents.agents:
            return
        # Single-agent case for now -- move the one pedestrian stand-in.
        a = self.current_agents.agents[0]
        # .warn (not .info) deliberately -- INFO is stdout and gets fully
        # block-buffered once this process isn't attached to a TTY (i.e. as
        # soon as you pipe through `tee`), so it can sit unflushed and never
        # show up. WARN/ERROR go to stderr, which glibc leaves unbuffered.
        #
        # Also logs whatever velocity field(s) hunav_msgs/Agent actually
        # carries (schema not confirmed here, so probe defensively instead
        # of guessing a field name and crashing). This is to tell apart two
        # very different explanations for a repeated position between two
        # consecutive responses: a genuine near-zero SFM force (velocity
        # ~0 too -- a real, if rare, stop) vs. an integration bug where a
        # nonzero velocity was computed but never actually applied to the
        # position that step (velocity NOT ~0 despite position not moving).
        vel_bits = []
        if hasattr(a, 'linear_vel'):
            vel_bits.append(f'linear_vel={a.linear_vel:.4f}')
        if hasattr(a, 'angular_vel'):
            vel_bits.append(f'angular_vel={a.angular_vel:.4f}')
        if hasattr(a, 'velocity'):
            v = a.velocity
            if hasattr(v, 'linear'):
                vel_bits.append(f'velocity=({v.linear.x:.4f},{v.linear.y:.4f})')
            else:
                vel_bits.append(f'velocity={v}')
        vel_str = (' ' + ' '.join(vel_bits)) if vel_bits else ' (no velocity field found on Agent)'
        if self.debug_raw:
            self.get_logger().warn(
                f'[bridge-raw] pos=({a.position.position.x:.4f},{a.position.position.y:.4f}) '
                f'yaw={a.yaw:.4f}{vel_str}')
        self._set_model_pose(a.position.position.x, a.position.position.y, a.yaw)
        # /people is published natively by hunav_agent_manager itself as a
        # side effect of the compute_agents call above -- no need (and no
        # longer safe, see ROOT-CAUSE FIX) to also publish it from here.


def main():
    from hunav_config import ensure_single_instance
    ensure_single_instance('hunav_model_bridge')

    parser = argparse.ArgumentParser(description='HuNav -> Gazebo pedestrian bridge')
    parser.add_argument('--pose-mode', choices=['set_pose', 'topic'], default='set_pose',
                        help='set_pose: ign service subprocess (plain models). '
                             'topic: /model/<MODEL_NAME>/cmd_pose for HuNavActorDriver (actor).')
    parser.add_argument('--debug-raw', action='store_true',
                        help='log every pose received from /compute_agents')
    args, _ = parser.parse_known_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = HunavModelBridge(pose_mode=args.pose_mode, debug_raw=args.debug_raw)
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
        if rclpy.ok():  # launch's SIGINT may already have shut the context down
            rclpy.shutdown()


if __name__ == '__main__':
    main()