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
  5. Publish that same live WORLD-frame position on /people
     (people_msgs/People) -- Phase C's live occupancy-grid exclusion zone
     subscribes to this, see build_occupancy_grid_dynamic.py
  6. Store the response as next cycle's carried-forward state

CAVEAT -- robot pose source, CONFIRMED BUG NOW FIXED: /odom does NOT report
true world pose -- confirmed empirically from a live run, where the first
/compute_agents call logged the robot's yaw as ~0 despite its known true
spawn yaw of 0.7854 rad. DiffDrive's simulated odometry starts at (0,0,0)
regardless of the robot's actual world spawn pose. Fixed below by composing
odom's own relative motion with the robot's known true spawn transform
(SPAWN_X/SPAWN_Y/SPAWN_YAW, matching spawn_entity's arguments in the launch
file exactly -- if that spawn pose ever changes, these three constants must
be updated to match, or this bug returns silently).

CAVEAT -- update rate: default 2 Hz, deliberately conservative. Each cycle
shells out to `ign service` (a real subprocess spawn), which has enough
overhead that a full 30Hz loop is not expected to keep up reliably. Fine
for a stationary or slow-moving agent; revisit with native ign-transport
Python bindings (avoiding subprocess spawn entirely) if a faster, smoother
update rate is needed later for realistic pedestrian motion.

CAVEAT -- untested: the `orientation` quaternion field on the set_pose
request is standard ignition.msgs.Pose structure but has only been
confirmed manually for the `position` field, not `orientation`, in this
environment. If the model's facing doesn't visibly update, check this
first.
"""
import math
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from people_msgs.msg import People, Person
from hunav_msgs.srv import GetAgents, ComputeAgents

WORLD_NAME = 'empty'
MODEL_NAME = 'pedestrian_standin'
UPDATE_RATE_HZ = 2.0
SET_POSE_TIMEOUT_MS = 300
STANDIN_Z = 0.8  # matches HuNavSystemPlugin_fortress.cpp's own updateGazeboPedestrians(),
                 # which sets actorPose.Pos().Z(0.8) for this exact mesh family --
                 # confirmed from source, not guessed. (Old value, 0.15, matched the
                 # earlier box's half-height and was never revisited when the visual
                 # was swapped to the real mesh -- that's what caused the pedestrian
                 # to render half-underground.)

# Must match spawn_entity's -x/-y/-Y arguments in the launch file exactly.
SPAWN_X = 0.0
SPAWN_Y = 0.0
SPAWN_YAW = 0.7854


def yaw_from_quaternion(q) -> float:
    """Standard quaternion -> yaw (Z-axis Euler) extraction."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float):
    """Returns (x, y, z, w) for a pure yaw rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def odom_to_world(odom_x: float, odom_y: float, odom_yaw: float):
    """
    DiffDrive's odometry starts at (0,0,0) regardless of the robot's true
    world spawn pose (confirmed empirically -- see the CONFIRMED BUG note
    above). Composes odom's own relative motion with the robot's known
    true spawn transform to recover genuine world-frame pose.
    """
    c, s = math.cos(SPAWN_YAW), math.sin(SPAWN_YAW)
    world_x = SPAWN_X + odom_x * c - odom_y * s
    world_y = SPAWN_Y + odom_x * s + odom_y * c
    world_yaw = SPAWN_YAW + odom_yaw
    return world_x, world_y, world_yaw


class HunavModelBridge(Node):
    def __init__(self):
        super().__init__('hunav_model_bridge')

        self.robot_pose = None  # (x, y, yaw), populated from /odom
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)

        self.get_agents_client = self.create_client(GetAgents, 'get_agents')
        self.compute_agents_client = self.create_client(ComputeAgents, 'compute_agents')
        self.people_pub = self.create_publisher(People, '/people', 10)

        self.current_agents = None  # hunav_msgs/Agents, carried forward each cycle
        self.timer = None  # started only once initialize_agents() has succeeded

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

    def _tick(self):
        if self.robot_pose is None:
            self.get_logger().warn('No /odom received yet, skipping cycle.', throttle_duration_sec=5.0)
            return
        if not self.compute_agents_client.service_is_ready():
            self.get_logger().warn('/compute_agents not ready, skipping cycle.', throttle_duration_sec=5.0)
            return

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
        self._publish_people(a)

    def _publish_people(self, agent):
        """
        Publishes the agent's live WORLD-frame position on /people
        (people_msgs/People) -- reclaims the same topic/message type
        HuNavSystemPluginIGN would have used, for consistency with the rest
        of this project's established conventions. Published in world
        frame deliberately (not base_link), so any consumer does its own
        world->base_link transform as needed -- keeps this message reusable
        rather than coupled to one specific consumer's frame.
        """
        msg = People()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        person = Person()
        person.name = agent.name
        person.position.x = agent.position.position.x
        person.position.y = agent.position.position.y
        person.position.z = 0.0
        person.velocity.x = agent.velocity.linear.x
        person.velocity.y = agent.velocity.linear.y
        person.velocity.z = 0.0
        person.reliability = 1.0
        msg.people = [person]
        self.people_pub.publish(msg)


def main():
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