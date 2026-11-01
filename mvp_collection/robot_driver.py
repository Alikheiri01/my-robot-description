#!/usr/bin/env python3
"""
robot_driver.py -- P4 of the MVP collection: drives the robot during an episode
(2026-11-01) so the pedestrian stays in the camera's view.

MODES (the same three the generator planned with plan_model.py)
  still   stand; turn once to the planned heading, then hold it
  watch   stand; turn toward the pedestrian whenever it is more than WATCH_DEADBAND/2
          off the camera axis (turn rate <= ROBOT_WMAX)
  follow  drive along the pedestrian's planned route (extended straight back behind the
          start), DIST metres behind the pedestrian, at most ROBOT_VMAX; steer toward a point
          FOLLOW_LOOKAHEAD_M ahead on the route (turn in place first when it is far off);
          stop when closer than ROBOT_MIN_PED_DIST_M to the pedestrian
  stop    zero command

COMMANDS (the episode runner sends them)
  /mvp/robot_cmd   std_msgs/String, JSON, ODOM frame:
      {"mode": "still", "heading": h}
      {"mode": "watch"}
      {"mode": "follow", "route": [[x, y], ...], "dist": d}
      {"mode": "stop"}
OUTPUT
  /cmd_vel          geometry_msgs/Twist, 20 Hz of simulation time
  /mvp/robot_state  std_msgs/String, JSON, 2 Hz: mode, v, w, distance to the pedestrian
INPUT
  /odom (the robot's true pose, robot_pose_relay.py), /people_smoothed_pose (odom frame)

TRY IT BY HAND (launch running):
    python3 robot_driver.py
    ros2 topic pub --once /mvp/robot_cmd std_msgs/String "data: '{\\"mode\\": \\"watch\\"}'"
    (then walk the pedestrian around with the goals function: the robot keeps turning to it)
Ctrl+C stops the robot (zero command) and exits.

The control law is in RobotController (no ROS), so check_scripts/sim_robot_driver.py can
run it against the planned episodes offline.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'hunav_codes'))

import numpy as np  # noqa: E402

import mvp_config as C  # noqa: E402
import plan_model as M  # noqa: E402

BACK_EXTENSION_M = 20.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class RobotController:
    """Pure control law. step() -> (v, w) for one time step."""

    def __init__(self):
        self.mode = 'stop'
        self.heading = None
        self.route = None        # Polyline including the back extension
        self.dist = None

    def set_command(self, cmd):
        mode = cmd.get('mode', 'stop')
        if mode not in ('still', 'watch', 'follow', 'stop'):
            raise ValueError(f'unknown mode {mode!r}')
        self.mode = mode
        self.heading = cmd.get('heading')
        if mode == 'follow':
            pts = [tuple(p) for p in cmd['route']]
            d0 = np.subtract(pts[1], pts[0])
            d0 = d0 / max(np.hypot(*d0), 1e-9)
            back = tuple(np.array(pts[0]) - BACK_EXTENSION_M * d0)
            self.route = M.Polyline([back] + pts)
            self.dist = float(cmd['dist'])

    def step(self, robot, ped):
        """robot: (x, y, yaw) odom; ped: (x, y) odom or None (not known yet)."""
        x, y, h = robot
        if self.mode == 'stop':
            return 0.0, 0.0
        if self.mode == 'still':
            if self.heading is None:
                return 0.0, 0.0
            err = wrap(self.heading - h)
            return 0.0, (clamp(2.0 * err, -C.ROBOT_WMAX, C.ROBOT_WMAX) if abs(err) > 0.03 else 0.0)
        if ped is None:
            return 0.0, 0.0
        if self.mode == 'watch':
            err = wrap(math.atan2(ped[1] - y, ped[0] - x) - h)
            if abs(err) <= C.WATCH_DEADBAND * 0.5:
                return 0.0, 0.0
            return 0.0, clamp(C.WATCH_GAIN * err, -C.ROBOT_WMAX, C.ROBOT_WMAX)
        # follow
        s_r, _ = self.route.project((x, y))
        s_p, _ = self.route.project(ped)
        v = clamp(C.FOLLOW_GAIN * ((s_p - self.dist) - s_r), 0.0, C.ROBOT_VMAX)
        if math.hypot(ped[0] - x, ped[1] - y) < C.ROBOT_MIN_PED_DIST_M:
            v = 0.0
        la = self.route.at([s_r + C.FOLLOW_LOOKAHEAD_M])[0]
        alpha = wrap(math.atan2(la[1] - y, la[0] - x) - h)
        if abs(alpha) > C.FOLLOW_TURN_FIRST_RAD:
            return 0.0, clamp(2.0 * alpha, -C.ROBOT_WMAX, C.ROBOT_WMAX)
        v *= math.cos(alpha)
        w = 2.0 * v * math.sin(alpha) / C.FOLLOW_LOOKAHEAD_M + 1.0 * alpha
        return v, clamp(w, -C.ROBOT_WMAX, C.ROBOT_WMAX)


def run_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseArray, Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from hunav_config import ensure_single_instance, yaw_from_quaternion

    ensure_single_instance('robot_driver')

    class RobotDriver(Node):
        def __init__(self):
            super().__init__('robot_driver', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.ctrl = RobotController()
            self.robot = None
            self.ped = None
            self.last = (0.0, 0.0)
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.state_pub = self.create_publisher(String, '/mvp/robot_state', 10)
            self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped_cb, 10)
            self.create_subscription(String, '/mvp/robot_cmd', self._cmd_cb, 10)
            self.create_timer(0.05, self._tick)
            self.create_timer(0.5, self._state)
            self.get_logger().info('robot_driver ready (mode stop). Commands on /mvp/robot_cmd.')

        def _odom_cb(self, msg):
            p = msg.pose.pose
            self.robot = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

        def _ped_cb(self, msg):
            if msg.poses:
                self.ped = (msg.poses[0].position.x, msg.poses[0].position.y)

        def _cmd_cb(self, msg):
            try:
                self.ctrl.set_command(json.loads(msg.data))
                self.get_logger().info(f'mode -> {self.ctrl.mode}')
            except (ValueError, KeyError, TypeError, IndexError) as e:
                self.get_logger().error(f'bad /mvp/robot_cmd ({e}); stopping the robot')
                self.ctrl.set_command({'mode': 'stop'})

        def _tick(self):
            if self.get_clock().now().nanoseconds == 0 or self.robot is None:
                return
            v, w = self.ctrl.step(self.robot, self.ped)
            self.last = (v, w)
            m = Twist()
            m.linear.x, m.angular.z = float(v), float(w)
            self.pub.publish(m)

        def _state(self):
            d = None
            if self.robot and self.ped:
                d = round(math.hypot(self.ped[0] - self.robot[0], self.ped[1] - self.robot[1]), 2)
            self.state_pub.publish(String(data=json.dumps(
                {'mode': self.ctrl.mode, 'v': round(self.last[0], 3), 'w': round(self.last[1], 3),
                 'dist_to_ped': d})))

        def stop(self):
            for _ in range(3):
                self.pub.publish(Twist())

    rclpy.init()
    node = RobotDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    run_node()