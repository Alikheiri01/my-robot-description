#!/usr/bin/env python3
"""
agent_control.py -- tell the pedestrian where to walk, by clicking in RViz.

HOW TO USE (simulation running with my_robot_hunav_full.launch.py)
    cd ~/ros2_ws/src/my_robot_description/agent_control
    python3 agent_control.py
  1. In RViz pick the "Publish Point" tool (toolbar, or key U) and click on
     the floor: each click adds a waypoint (yellow, numbered).
  2. Press  g  in THIS terminal: the pedestrian walks to waypoint 1, 2, ...
     and loops through them (cyclic_goals in the yaml). One waypoint = walk
     there and stay. The goals HuNav is really using are drawn green.
  Keys:  g go   u undo last waypoint   c clear waypoints   s stop where it is
         r back to the yaml goals   + / - speed +-0.1 m/s   0 yaml speed
         h help   q quit
  Without a terminal, or for repeatable scenarios, give the goals directly
  (in the RViz / odom frame, metres), e.g. two points and a speed:
    python3 agent_control.py --goals "2,0; 2,3" --speed 0.8
  This sends once, waits until HuNav confirms, and exits.

HOW IT WORKS
  clicks (/clicked_point, RViz 'map' frame = 'odom' here) -> converted to the
  Gazebo WORLD frame (hunav_config.odom_to_world) -> /hunav/agent_goals and
  /hunav/agent_speed -> hunav_model_bridge_ctrl.py -> HuNav (patched
  hunav_agent_manager). The bridge reports back /hunav/current_goals (world)
  and /hunav/current_speed; they are shown here and in RViz
  (/hunav/goal_markers, MarkerArray).
  HuNav still does the walking (social force model), so the agent avoids the
  robot as before. It does NOT know about static obstacles in this world
  (none in the empty world yet) -- that comes with scenes later.
"""
import argparse
import math
import os
import select
import sys
import termios
import threading
import time
import tty

import os as _os
# This file lives in my_robot_description/agent_control/; the shared code
# (hunav_config.py, hunav_model_bridge.py, time_sync.py) is in ../hunav_codes.
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'hunav_codes'))
from hunav_config import odom_to_world, world_to_odom  # noqa: E402

SPEED_STEP = 0.1
SPEED_MIN, SPEED_MAX = 0.1, 2.0
CONFIRM_TIMEOUT_SEC = 10.0

HELP = ('keys: g go | u undo | c clear | s stop here | r yaml goals | + / - speed | 0 yaml speed | h help | q quit\n'
        'RViz: "Publish Point" tool (key U), then click on the floor to add a waypoint')


def parse_goals(text):
    pts = []
    for part in text.split(';'):
        part = part.strip()
        if not part:
            continue
        x, y = (float(v) for v in part.split(','))
        pts.append((x, y))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--goals', help='"x1,y1; x2,y2; ..." in the RViz/odom frame: send once and exit')
    ap.add_argument('--speed', type=float, help='walking speed in m/s (with --goals, or as a start value)')
    ap.add_argument('--yaml', action='store_true', help='send "back to the yaml goals and speed" and exit')
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped, Pose, PoseArray, Point
    from std_msgs.msg import Float32
    from visualization_msgs.msg import Marker, MarkerArray

    class AgentControl(Node):
        def __init__(self):
            super().__init__('agent_control')
            self.lock = threading.Lock()
            self.draft = []            # [(x, y)] odom frame, not sent yet
            self.active = []           # [(x, y)] odom frame, what HuNav uses (from the bridge)
            self.speed = None          # current speed reported by the bridge
            self.ped = None            # (x, y) odom frame
            self.heard_bridge = False
            self.goals_pub = self.create_publisher(PoseArray, '/hunav/agent_goals', 10)
            self.speed_pub = self.create_publisher(Float32, '/hunav/agent_speed', 10)
            self.marker_pub = self.create_publisher(MarkerArray, '/hunav/goal_markers', 10)
            self.create_subscription(PointStamped, '/clicked_point', self._click_cb, 10)
            self.create_subscription(PoseArray, '/hunav/current_goals', self._current_cb, 10)
            self.create_subscription(Float32, '/hunav/current_speed', self._speed_cb, 10)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped_cb, 10)
            self.create_timer(0.5, self._draw)

        # --- inputs ---------------------------------------------------------------
        def _click_cb(self, msg):
            if msg.header.frame_id not in ('map', 'odom', ''):
                self.say(f'click ignored: frame "{msg.header.frame_id}" (set RViz Fixed Frame to map)')
                return
            with self.lock:
                self.draft.append((msg.point.x, msg.point.y))
                n = len(self.draft)
            self.say(f'waypoint {n}: ({msg.point.x:.2f}, {msg.point.y:.2f})   -> g to go, u undo, c clear')

        def _current_cb(self, msg):
            pts = []
            for p in msg.poses:
                ox, oy, _ = world_to_odom(p.position.x, p.position.y, 0.0)
                pts.append((ox, oy))
            with self.lock:
                self.active = pts
            if not self.heard_bridge:
                self.heard_bridge = True
                self.say('connected to hunav_model_bridge_ctrl.py; current goals: ' + self.fmt(pts))

        def _speed_cb(self, msg):
            self.speed = msg.data

        def _ped_cb(self, msg):
            if msg.poses:
                self.ped = (msg.poses[0].position.x, msg.poses[0].position.y)

        # --- commands -------------------------------------------------------------
        def send_goals(self, pts_odom):
            out = PoseArray()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = 'world'
            for x, y in pts_odom:
                wx, wy, _ = odom_to_world(x, y, 0.0)
                p = Pose()
                p.position.x, p.position.y = wx, wy
                p.orientation.w = 1.0
                out.poses.append(p)
            self.goals_pub.publish(out)

        def send_speed(self, v):
            self.speed_pub.publish(Float32(data=float(v)))

        def key(self, k):
            if k == 'g':
                with self.lock:
                    pts = list(self.draft)
                if not pts:
                    self.say('no waypoints yet: click in RViz with the "Publish Point" tool first')
                    return
                self.send_goals(pts)
                with self.lock:
                    self.draft = []
                self.say(f'sent {len(pts)} goal(s): {self.fmt(pts)}')
            elif k == 'u':
                with self.lock:
                    if self.draft:
                        self.draft.pop()
                    n = len(self.draft)
                self.say(f'{n} waypoint(s) left')
            elif k == 'c':
                with self.lock:
                    self.draft = []
                self.say('waypoints cleared (nothing sent)')
            elif k == 's':
                if self.ped is None:
                    self.say('pedestrian position not known yet (/people_smoothed_pose)')
                    return
                self.send_goals([self.ped])
                self.say(f'stop: single goal at the current position ({self.ped[0]:.2f}, {self.ped[1]:.2f})')
            elif k == 'r':
                self.goals_pub.publish(PoseArray())
                self.say('back to the yaml goals')
            elif k in '+=-_':
                base = self.speed if self.speed else 1.0
                v = base + (SPEED_STEP if k in '+=' else -SPEED_STEP)
                v = min(max(round(v, 2), SPEED_MIN), SPEED_MAX)
                self.send_speed(v)
                self.speed = v
                self.say(f'speed {v:.2f} m/s')
            elif k == '0':
                self.send_speed(0.0)
                self.say('back to the yaml speed')
            elif k == 'h':
                self.say(HELP)

        # --- output ---------------------------------------------------------------
        @staticmethod
        def fmt(pts):
            return ', '.join(f'({x:.2f}, {y:.2f})' for x, y in pts) or 'none'

        def say(self, text):
            sys.stdout.write('\r' + text + '\n')
            sys.stdout.flush()

        def _draw(self):
            with self.lock:
                draft, active = list(self.draft), list(self.active)
            arr = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            now = self.get_clock().now().to_msg()

            def mk(ns, mid, mtype, rgba, scale):
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = now
                m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = rgba
                m.scale.x = m.scale.y = m.scale.z = scale
                return m

            for name, pts, rgba, loop in (('draft', draft, (1.0, 0.85, 0.0, 0.9), False),
                                          ('active', active, (0.1, 0.9, 0.2, 0.9), True)):
                for i, (x, y) in enumerate(pts):
                    s = mk(name, 2 * i, Marker.SPHERE, rgba, 0.25)
                    s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, 0.1
                    t = mk(name, 2 * i + 1, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0), 0.3)
                    t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, 0.5
                    t.text = str(i + 1)
                    arr.markers += [s, t]
                if len(pts) >= 2:
                    line = mk(name + '_path', 0, Marker.LINE_STRIP, rgba, 0.04)
                    seq = pts + ([pts[0]] if loop else [])
                    line.points = [Point(x=x, y=y, z=0.05) for x, y in seq]
                    arr.markers.append(line)
            self.marker_pub.publish(arr)

    rclpy.init()
    node = AgentControl()

    # ---- one-shot mode -----------------------------------------------------------
    if args.goals or args.yaml:
        t0 = time.monotonic()
        while rclpy.ok() and not node.heard_bridge and time.monotonic() - t0 < CONFIRM_TIMEOUT_SEC:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.heard_bridge:
            print('No /hunav/current_goals: is the launch running with hunav_model_bridge_ctrl.py?')
            sys.exit(1)
        if args.yaml:
            node.goals_pub.publish(PoseArray())
            node.send_speed(0.0)
            want = None
        else:
            want = parse_goals(args.goals)
            if args.speed:
                node.send_speed(args.speed)
            node.send_goals(want)
        t0 = time.monotonic()
        ok = want is None
        while rclpy.ok() and not ok and time.monotonic() - t0 < CONFIRM_TIMEOUT_SEC:
            rclpy.spin_once(node, timeout_sec=0.2)
            with node.lock:
                act = list(node.active)
            ok = len(act) == len(want) and all(
                any(math.hypot(a[0] - w[0], a[1] - w[1]) < 0.01 for a in act) for w in want)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)
        print('HuNav confirmed the goals.' if ok and want else ('Sent.' if ok else
              'HuNav did NOT take the goals within 10 s: is hunav_agent_manager patched and rebuilt?'))
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if ok else 2)

    # ---- interactive mode ----------------------------------------------------------
    if args.speed:
        node.send_speed(args.speed)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    print('agent_control: ' + HELP)
    if not sys.stdin.isatty():
        print('(no terminal: keys disabled; clicks are collected but cannot be sent -- use --goals)')
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not r:
                    continue
                k = os.read(fd, 1).decode(errors='ignore')
                if k in ('q', '\x03'):
                    break
                node.key(k)
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()