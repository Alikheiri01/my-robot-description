#!/usr/bin/env python3
"""
robot_pose_relay.py -- publishes /odom for the whole pipeline from the robot's
TRUE pose in Gazebo instead of its wheel odometry (2026-10-14).

WHY
check_odom_drift.py showed the wheel odometry (DiffDrive) disagreeing with
Gazebo's true robot pose by up to 1.7 deg in heading and 0.28 m in position
while the four-wheeled robot turns (the wheels skid). The pipeline places
every depth cloud, every exclusion circle and every crop with /odom, so on a
turning robot the obstacles and the pedestrian landed 0.1-0.25 m from where
they really were (T17). A dataset must not learn that smear: localisation
error is a real-robot problem to study separately, not something to bake
into the training crops.

HOW
An OdometryPublisher plugin in the robot's model publishes the model's true
WORLD pose, stamped with simulation time, on /robot_truth/odometry (bridged
to ROS). This node converts it into the 'odom' frame the pipeline expects
(origin = robot spawn pose, hunav_config.world_to_odom) and publishes it on
/odom, same message type and stamp. Every consumer (grid builder, recorder,
validator, HuNav bridge, crop viewer) is unchanged -- they just get correct
poses. The wheel odometry is still available on /odom_wheel.

MODES
  --source truth  (default) Gazebo's true pose -> /odom
  --source wheel  old behaviour: /odom_wheel passed through to /odom unchanged
                  (to compare, or if the plugin is missing)

Note: RViz still draws the robot model from the wheel odometry's /tf; that is
display only, nothing in the pipeline uses it.
"""
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TRUTH_TOPIC = '/robot_truth/odometry'
WHEEL_TOPIC = '/odom_wheel'
OUT_TOPIC = '/odom'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['truth', 'wheel'], default='truth')
    args, _ = ap.parse_known_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from hunav_config import world_to_odom, yaw_from_quaternion, yaw_to_quaternion, ensure_single_instance

    ensure_single_instance('robot_pose_relay')

    class Relay(Node):
        def __init__(self):
            super().__init__('robot_pose_relay')
            # RELIABLE publisher: works for both reliable and best-effort subscribers
            self.pub = self.create_publisher(Odometry, OUT_TOPIC, 10)
            self.n = 0
            src = TRUTH_TOPIC if args.source == 'truth' else WHEEL_TOPIC
            self.create_subscription(Odometry, src, self._truth if args.source == 'truth' else self._wheel,
                                     qos_profile_sensor_data)
            self.create_timer(5.0, self._status)
            self.get_logger().info(f'robot pose source: {args.source.upper()} ({src}) -> {OUT_TOPIC}')

        def _truth(self, msg):
            p = msg.pose.pose.position
            ox, oy, oyaw = world_to_odom(p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation))
            out = Odometry()
            out.header.stamp = msg.header.stamp          # simulation time of the physics step
            out.header.frame_id = 'odom'
            out.child_frame_id = 'base_link'
            out.pose.pose.position.x, out.pose.pose.position.y = ox, oy
            qx, qy, qz, qw = yaw_to_quaternion(math.atan2(math.sin(oyaw), math.cos(oyaw)))
            o = out.pose.pose.orientation
            o.x, o.y, o.z, o.w = qx, qy, qz, qw
            out.twist = msg.twist                        # body frame, same convention as DiffDrive
            self.pub.publish(out)
            self.n += 1

        def _wheel(self, msg):
            self.pub.publish(msg)
            self.n += 1

        def _status(self):
            if self.n == 0:
                self.get_logger().warn(
                    f'nothing received on {TRUTH_TOPIC if args.source == "truth" else WHEEL_TOPIC} yet -- '
                    + ('is the OdometryPublisher plugin in the robot model and the bridge entry in the launch file?'
                       if args.source == 'truth' else 'is the bridge remapping /odom -> /odom_wheel?'),
                    throttle_duration_sec=15.0)
            else:
                self.get_logger().info(f'{self.n} robot poses relayed ({args.source})', throttle_duration_sec=60.0)

    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()