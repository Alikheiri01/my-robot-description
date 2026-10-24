#!/usr/bin/env python3
"""
collection_gate.py -- the START GATE: is the data being produced right now
trustworthy enough to become training samples? (2026-10-24)

WHY
Without it, the recorder also saves samples from moments that are not real
behaviour: the pedestrian standing at its spawn point before it has any goals,
a pedestrian or robot that teleports, obstacles being added or moved. Until now
this was avoided by hand ("wait one full leg, don't touch the obstacles").

WHAT IT DOES
It watches the pedestrian pose, the robot pose, the grid and the obstacle file,
and publishes on /dataset/gate (std_msgs/String, JSON, ~4 Hz) the time
intervals (simulation time) that are CLOSED and why. dataset_recorder.py saves
a sample only if its WHOLE window (history start .. future end, ~9.5 s) touches
no closed interval. The gate only marks things; it never stops anything.

CLOSED (these really corrupt every window that contains them):
  * not started  from launch until the pedestrian first walks for START_WALK_SEC
                 (> WALK_SPEED); the window may start at that first step.
                 Standing still LATER is real behaviour and never closes the gate.
  * teleport     the pedestrian or the robot jumps faster than MAX_*_SPEED
                 (the interval is just the jump itself).
  * scene change current_obstacles.yaml changed (obstacles added/moved/removed):
                 SCENE_BEFORE..SCENE_AFTER around the moment it was noticed (the
                 bridge reads Gazebo once a second, so the change is a bit earlier).
  * clock reset  simulation time went backwards (simulation restarted).

NOT CLOSED, ONLY REPORTED (robustness, 2026-10-24): a short hiccup of a stream
(grid, poses) does not close the gate. The recorder already handles it per
sample: an anchor with no grid near it, or a window with a pose dropout, is
dropped by itself, so a 1 s glitch costs only the 1-2 samples that needed the
missing data, not the ~19 samples whose 9.5 s windows contain it. A stream that
stays silent for more than DEAD_SEC is shown as DEGRADED here, loudly.

Start it once per launch (my_robot_hunav_full.launch.py does). Watch it with
    ros2 topic echo /dataset/gate
"""
import json
import math
import sys
from collections import deque
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))

GATE_TOPIC = '/dataset/gate'
OBSTACLES_FILE = _THIS_DIR.parent / 'agent_control' / 'current_obstacles.yaml'

WALK_SPEED = 0.3          # m/s: faster than this counts as walking
START_WALK_SEC = 2.0      # s: walking this long (continuously) = the scenario has started
SPEED_WINDOW_SEC = 0.5    # s: speed measured over this span
MAX_PED_SPEED = 3.0       # m/s: faster than any walking person -> teleport
MAX_ROBOT_SPEED = 3.0     # m/s
MIN_JUMP_M = 0.3          # a teleport must also be at least this long (no noise on tiny time steps)
SCENE_BEFORE = 2.0        # s before the scene change was noticed
SCENE_AFTER = 2.0         # s after it
DEAD_SEC = 3.0            # s: a stream silent this long is reported as DEGRADED
KEEP_SEC = 120.0          # s: closed intervals kept in the message
CLOCK_BACK_SEC = 1.0      # s: time going back more than this = simulation restart


class GateLogic:
    """No ROS here, so it can be tested with synthetic data. Times are simulation seconds."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.closed = []                # [start, end or None, reason]
        self.now = None
        self.last = {'pedestrian pose': None, 'robot pose': None, 'grid': None}
        self.ped_track = deque()        # (t, x, y) for the walking speed
        self.prev_ped = None
        self.prev_robot = None
        self.walk_since = None
        self.started = None             # time the scenario started (first sustained walk)
        self.closed.append([None, None, 'not started: waiting for the pedestrian to walk'])
        self.events = []                # human-readable log lines for the node

    # --- inputs ---------------------------------------------------------------
    def _tick(self, t):
        if self.now is not None and t < self.now - CLOCK_BACK_SEC:
            self.reset()
            self.events.append('simulation time went backwards: gate reset (closed until walking again)')
        if self.now is None or t > self.now:
            self.now = t
        if self.closed and self.closed[0][0] is None and self.started is None:
            self.closed[0][0] = t        # the "not started" interval begins at the first message

    def on_ped(self, t, x, y):
        self._tick(t)
        self.last['pedestrian pose'] = t
        if self.prev_ped is not None:
            pt, px, py = self.prev_ped
            dt, d = t - pt, math.hypot(x - px, y - py)
            if dt > 0 and d >= MIN_JUMP_M and d / dt > MAX_PED_SPEED:
                self._event(pt, t, f'pedestrian jumped {d:.2f} m in {dt:.2f} s (teleport)')
                self.ped_track.clear()
                self.walk_since = None
        self.prev_ped = (t, x, y)
        self.ped_track.append((t, x, y))
        while len(self.ped_track) > 2 and self.ped_track[1][0] <= t - SPEED_WINDOW_SEC:
            self.ped_track.popleft()
        self._walking(t)

    def on_robot(self, t, x, y):
        self._tick(t)
        self.last['robot pose'] = t
        if self.prev_robot is not None:
            pt, px, py = self.prev_robot
            dt, d = t - pt, math.hypot(x - px, y - py)
            if dt > 0 and d >= MIN_JUMP_M and d / dt > MAX_ROBOT_SPEED:
                self._event(pt, t, f'robot jumped {d:.2f} m in {dt:.2f} s (teleport)')
        self.prev_robot = (t, x, y)

    def on_grid(self, t):
        self._tick(t)
        self.last['grid'] = t

    def on_scene_change(self):
        if self.now is None:
            return
        self._event(self.now - SCENE_BEFORE, self.now + SCENE_AFTER, 'obstacles changed')

    # --- internals -------------------------------------------------------------
    def _event(self, a, b, reason):
        self.closed.append([a, b, reason])
        self.events.append(f'CLOSED {a:.2f}..{b:.2f} s: {reason}')

    def _walking(self, t):
        if self.started is not None:
            return
        t0, x0, y0 = self.ped_track[0]
        _, x1, y1 = self.ped_track[-1]
        span = t - t0
        speed = math.hypot(x1 - x0, y1 - y0) / span if span >= SPEED_WINDOW_SEC * 0.8 else 0.0
        if speed > WALK_SPEED:
            if self.walk_since is None:
                self.walk_since = t0             # the walk began at the start of the measured span
            if t - self.walk_since >= START_WALK_SEC and all(v is not None for v in self.last.values()):
                self.started = self.walk_since
                self.closed[0][1] = self.started
                self.events.append(f'scenario started at {self.started:.2f} s (pedestrian walking, all streams up): '
                                   f'samples from here on')
        else:
            self.walk_since = None

    # --- outputs ---------------------------------------------------------------
    def degraded(self):
        if self.now is None:
            return []
        return [f'no {k} for {self.now - v:.1f} s' if v is not None else f'no {k} yet'
                for k, v in self.last.items() if v is None or self.now - v > DEAD_SEC]

    def state(self):
        """(open, reason) at the current time."""
        if self.now is None:
            return False, 'no data yet'
        for a, b, r in reversed(self.closed):
            if a is not None and a <= self.now and (b is None or b >= self.now):
                return False, r
        return True, 'open'

    def message(self):
        keep = [[a, b, r] for a, b, r in self.closed
                if a is not None and (b is None or self.now is None or b >= self.now - KEEP_SEC)]
        self.closed = [c for c in self.closed if c[0] is None or c in keep]
        is_open, reason = self.state()
        return {'now': self.now, 'open': is_open, 'reason': reason, 'started': self.started,
                'closed': keep, 'degraded': self.degraded()}


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry, OccupancyGrid
    from geometry_msgs.msg import PoseArray
    from hunav_config import ensure_single_instance

    ensure_single_instance('collection_gate')

    class Gate(Node):
        def __init__(self):
            super().__init__('collection_gate')     # wall-clock timers; all times come from message stamps
            self.g = GateLogic()
            self.pub = self.create_publisher(String, GATE_TOPIC, 10)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped, 10)
            self.create_subscription(Odometry, '/odom', self._odom, qos_profile_sensor_data)
            self.create_subscription(OccupancyGrid, '/occupancy_grid/base', self._grid, 10)
            self.scene = self._read_scene()
            self.last_state = None
            self.create_timer(0.25, self._publish)
            self.create_timer(1.0, self._watch_scene)
            self.get_logger().info(f'start gate running: publishing {GATE_TOPIC}. Closed until the pedestrian walks.')

        @staticmethod
        def _stamp(h):
            return h.stamp.sec + h.stamp.nanosec * 1e-9

        def _ped(self, msg):
            if msg.poses:
                p = msg.poses[0].position
                self.g.on_ped(self._stamp(msg.header), p.x, p.y)

        def _odom(self, msg):
            p = msg.pose.pose.position
            self.g.on_robot(self._stamp(msg.header), p.x, p.y)

        def _grid(self, msg):
            self.g.on_grid(self._stamp(msg.header))

        @staticmethod
        def _read_scene():
            try:
                return '\n'.join(l for l in OBSTACLES_FILE.read_text().splitlines() if not l.startswith('#'))
            except OSError:
                return None

        def _watch_scene(self):
            cur = self._read_scene()
            if cur is not None and self.scene is not None and cur != self.scene:
                self.g.on_scene_change()
            if cur is not None:
                self.scene = cur

        def _publish(self):
            # one call line per severity (rclpy refuses a severity change on the same line)
            for e in self.g.events:
                if e.startswith('CLOSED') or 'reset' in e:
                    self.get_logger().warn(e)
                else:
                    self.get_logger().info(e)
            self.g.events.clear()
            m = self.g.message()
            self.pub.publish(String(data=json.dumps(m)))
            st = (m['open'], m['reason'])
            if st != self.last_state:
                if m['open']:
                    self.get_logger().info('GATE OPEN')
                else:
                    self.get_logger().info(f'GATE CLOSED: {m["reason"]}')
                self.last_state = st
            if m['degraded'] and m['started'] is not None:
                self.get_logger().warn('DEGRADED: ' + ', '.join(m['degraded']) +
                                       ' (samples needing this data are dropped by the recorder)',
                                       throttle_duration_sec=10.0)

    rclpy.init()
    node = Gate()
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