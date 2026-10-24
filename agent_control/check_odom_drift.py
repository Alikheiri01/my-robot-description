#!/usr/bin/env python3
"""
check_odom_drift.py -- does the robot's /odom still agree with where Gazebo
really has the robot? (2026-10-13)

WHY
T17 (robot turning) puts the camera's view of the pedestrian ~0.1-0.25 m away
from its true position, while a standing or straight-driving robot is fine
(3-5 cm). The validator's robot-timing scan says it is NOT a timing shift (best
at 0.00 s). The next suspect: the odometry itself. The pipeline places every
depth cloud in the world with /odom. This robot has four wheels; turning in
place makes them skid, and wheel odometry that assumes no skid then believes it
turned a different angle than it did. Every turn adds a heading error, and a
heading error of 2 deg already moves a point 4 m away by 14 cm.

WHAT IT DOES
Once a second it reads the robot's TRUE pose from Gazebo (/world/<world>/pose/info,
model "my_robot") and compares it with /odom at that same simulation time
(converted to the world frame like the pipeline does). It prints the heading
and position difference, and a summary at Ctrl+C. Read-only.

USAGE (simulation running; run robot_motion.py in another terminal)
    python3 check_odom_drift.py
"""
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'hunav_codes'))

from gazebo_scene import parse_text_proto, first, pose_of, yaw_tilt  # noqa: E402

ROBOT_MODEL = 'my_robot'


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from hunav_config import WORLD_NAME, odom_to_world, yaw_from_quaternion
    from time_sync import StampedPoseHistory, stamp_to_sec

    class Drift(Node):
        def __init__(self):
            super().__init__('check_odom_drift', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.odom = StampedPoseHistory(max_age_sec=30.0)
            self.lock = threading.Lock()
            self.rows = []
            self.create_subscription(Odometry, '/odom', self._odom, qos_profile_sensor_data)
            threading.Thread(target=self._poll, daemon=True).start()
            self.get_logger().info('Comparing /odom with Gazebo\'s true robot pose once a second. Ctrl+C for the summary.')

        def _odom(self, msg):
            p = msg.pose.pose.position
            x, y, yaw = odom_to_world(p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation))
            with self.lock:
                self.odom.add(stamp_to_sec(msg.header.stamp), x, y, yaw)

        def _poll(self):
            while True:
                try:
                    out = subprocess.run(['timeout', '3', 'ign', 'topic', '-e', '-t',
                                          f'/world/{WORLD_NAME}/pose/info', '-n', '1'],
                                         capture_output=True, text=True).stdout
                    msg = parse_text_proto(out)
                    st = first(first(msg, 'header', {}) or {}, 'stamp', {}) or {}
                    t = first(st, 'sec', 0.0) + first(st, 'nsec', 0.0) * 1e-9
                    gz = next((p for p in msg.get('pose', []) if first(p, 'name') == ROBOT_MODEL), None)
                    if gz is None:
                        self.get_logger().warn(f'model "{ROBOT_MODEL}" not in pose/info', throttle_duration_sec=10)
                    else:
                        (gx, gy, _), q = pose_of(gz)
                        gyaw, _ = yaw_tilt(q)
                        with self.lock:
                            o = self.odom.at(t, max_hold=0.2)
                        if o is not None:
                            dyaw = math.degrees(wrap(o[2] - gyaw))
                            dpos = math.hypot(o[0] - gx, o[1] - gy)
                            self.rows.append((t, dyaw, dpos))
                            self.get_logger().info(
                                f't={t:7.2f}  true yaw {math.degrees(gyaw):7.1f} deg  odom yaw {math.degrees(o[2]):7.1f} deg'
                                f'  ->  heading error {dyaw:+6.2f} deg, position error {dpos:.3f} m '
                                f'(= {4.0 * math.radians(abs(dyaw)):.2f} m at 4 m from the robot)')
                except Exception as e:
                    self.get_logger().warn(f'poll failed: {e}', throttle_duration_sec=10)
                time.sleep(1.0)

        def summary(self):
            if not self.rows:
                print('no comparisons made (is the simulation running?)')
                return
            d = [r[1] for r in self.rows]
            p = [r[2] for r in self.rows]
            print(f'\n{len(self.rows)} comparisons over {self.rows[-1][0] - self.rows[0][0]:.0f} s (sim)')
            print(f'  heading error: first {d[0]:+.2f} deg, last {d[-1]:+.2f} deg, largest {max(d, key=abs):+.2f} deg')
            print(f'  position error: last {p[-1]:.3f} m, largest {max(p):.3f} m')
            big = max(abs(x) for x in d)
            print('VERDICT: ' + ('the odometry drifts -- /odom disagrees with the true pose by more than 1 deg. '
                                 'That alone misplaces the depth points while the robot turns.' if big > 1.0 else
                                 'the odometry agrees with the true pose (< 1 deg): not the cause.'))

    rclpy.init()
    node = Drift()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()