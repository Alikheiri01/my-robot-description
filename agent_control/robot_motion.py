#!/usr/bin/env python3
"""
robot_motion.py -- repeatable robot motion for the moving-robot tests (2026-10-13).

Driving by keyboard made T17-T20 different every time (turning with the
pedestrian, against it, out of view). This publishes a fixed, smooth pattern
on /cmd_vel, timed in SIMULATION seconds, so a test can be repeated exactly.

PATTERNS (combine freely)
  --turn A      swing the heading left/right: angular speed A*cos(2*pi*t/P) rad/s.
                The heading swings +-A*P/(2*pi) rad around its start heading
                (A=0.5, P=8 -> about +-37 deg), so the pedestrian stays mostly in view.
  --forward V   drive forward/back: linear speed V*sin(2*pi*t/P) m/s
                (V=0.2, P=8 -> about 0.5 m forward, then back to the start).
  --period P    period of both swings, seconds (default 8)

EXAMPLES
  python3 robot_motion.py --turn 0.5                 # T17: turn left/right +-37 deg
  python3 robot_motion.py --forward 0.2              # T18: forward/back 0.5 m
  python3 robot_motion.py --turn 0.4 --forward 0.15  # T20: both
Ctrl+C stops the robot (sends a zero command) and exits.
"""
import argparse
import math
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--turn', type=float, default=0.0, help='peak angular speed, rad/s')
    ap.add_argument('--forward', type=float, default=0.0, help='peak linear speed, m/s')
    ap.add_argument('--period', type=float, default=8.0, help='period of the swing, s')
    args, _ = ap.parse_known_args()
    if args.turn == 0.0 and args.forward == 0.0:
        sys.exit('Give --turn and/or --forward (see --help).')

    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from geometry_msgs.msg import Twist

    class Motion(Node):
        def __init__(self):
            super().__init__('robot_motion', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.t0 = None
            self.create_timer(0.05, self.tick)
            self.get_logger().info(
                f'turn {args.turn} rad/s, forward {args.forward} m/s, period {args.period} s (sim time). '
                f'Heading swing about +-{math.degrees(args.turn * args.period / (2 * math.pi)):.0f} deg, '
                f'travel about {args.forward * args.period / math.pi:.2f} m. Ctrl+C to stop.')

        def tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now == 0.0:
                return                      # /clock not there yet
            if self.t0 is None:
                self.t0 = now
            ph = 2 * math.pi * (now - self.t0) / args.period
            m = Twist()
            m.angular.z = args.turn * math.cos(ph)     # heading swings around the start heading
            m.linear.x = args.forward * math.sin(ph)   # forward from the start, then back to it
            self.pub.publish(m)

        def stop(self):
            self.pub.publish(Twist())

    rclpy.init()
    node = Motion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()