#!/usr/bin/env python3
"""
Continuous keyboard teleop for the pedestrian_standin model -- velocity-based
control (same feel as ROS2's own teleop_twist_keyboard), so you can drive the
pedestrian around smoothly in real time while watching detection/exclusion/
crop behavior update.

DESIGN (v2 -- fixes choppy motion / drift-after-release from v1):
  - Each direction key (w/a/s/d/q/e) just records "this key was last pressed
    at time T". Every tick (~30Hz), current velocity is computed from which
    keys were pressed within the last HOLD_TIMEOUT seconds, and position is
    INTEGRATED continuously (x += vx*dt each tick) -- not stepped once per
    keypress. This is what makes motion smooth between keypresses rather
    than jumping in fixed increments.
  - Terminal key-repeat (holding a key down) sends a steady stream of the
    same character with small gaps between repeats -- HOLD_TIMEOUT (150ms)
    is comfortably longer than those gaps but short enough to stop quickly
    once you actually release the key.
  - v1's real bug: it read at most ONE buffered character per ~33ms tick.
    If you held a key for a couple of seconds, the OS queued far more
    characters than that, and they kept draining out (and being treated as
    fresh keypresses) for a while AFTER you let go -- that's the "continues
    after I stop pressing" symptom. Fixed by draining the ENTIRE pending
    input buffer every tick instead of one character at a time.

WHY THIS EXISTS (vs. manual_pedestrian_pose.py, the older type-a-pose REPL):
that script published directly to /people_smoothed_pose, bypassing
heading_smoother.py -- which is still running (started by the full launch
file) and would ALSO be publishing to the same topic, reintroducing the
duplicate-publisher bug class already root-caused and fixed for /people
earlier in this project. This script publishes ONLY /people (world frame,
exactly what hunav_agent_manager normally publishes); heading_smoother.py
remains the sole producer of /people_smoothed_pose, unchanged.

REQUIRED: stop hunav_model_bridge.py first (both would fight over set_pose
on the same model). Leave everything else running normally.
    pkill -9 -f hunav_model_bridge.py

CONTROLS (world frame, same convention as Gazebo's Component Inspector):
    w / s     -- move +x / -x
    a / d     -- move -y / +y
    q / e     -- rotate in place +/- (only while NOT moving -- while moving,
                 the pedestrian auto-faces its direction of travel, same as
                 heading_smoother.py's velocity-derived arrow, so the mesh
                 and the arrow always agree). While stationary, a small
                 phantom velocity is published in the current facing
                 direction (position doesn't change) purely so the arrow
                 tracks q/e turns too -- otherwise heading_smoother.py has
                 nothing to update the heading from and holds the old one.
    +  (or =) -- increase speed (both linear and angular) by 25%
    -         -- decrease speed by 25%
    space     -- print current pose and speed
    x or Ctrl-C -- quit

Hold a key down for continuous motion (your terminal's own key-repeat
drives it) -- a single tap just nudges it slightly, by design.

WHEN YOU'RE DONE: press x or Ctrl-C, then restart hunav_model_bridge.py to
resume normal autonomous HuNav-driven pedestrian behavior.
"""
import math
import os
import select
import subprocess
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from people_msgs.msg import People, Person

from hunav_config import (
    WORLD_NAME, MODEL_NAME, SET_POSE_TIMEOUT_MS, STANDIN_Z,
    yaw_to_quaternion, MIN_SPEED_FOR_HEADING_UPDATE,
)

# --- tunables ---------------------------------------------------------------
DEFAULT_LINEAR_SPEED = 0.5     # m/s
DEFAULT_ANGULAR_SPEED = 0.8    # rad/s (~46 deg/s)
MIN_LINEAR_SPEED, MAX_LINEAR_SPEED = 0.05, 2.0
MIN_ANGULAR_SPEED, MAX_ANGULAR_SPEED = 0.1, 3.0

# heading_smoother.py holds the last heading whenever /people's velocity is
# below MIN_SPEED_FOR_HEADING_UPDATE (there's no orientation field on
# /people to read instead). So to test the agent standing still at various
# headings, we publish a "signal" velocity just above that threshold, in
# whatever direction q/e last turned to face -- WITHOUT adding it to the
# actual position -- purely so the arrow tracks the chosen facing while the
# pedestrian doesn't actually move. 1.5x margin so float noise never dips
# it back under the threshold.
PHANTOM_HEADING_SPEED = MIN_SPEED_FOR_HEADING_UPDATE * 1.5
SPEED_STEP_FACTOR = 1.25       # multiplicative change per +/- press

HOLD_TIMEOUT_SEC = 0.15        # a direction counts as "held" if its key was
                                # seen within this long ago -- survives the
                                # gaps between terminal key-repeat events but
                                # decays quickly once you actually release

SET_POSE_RATE_HZ = 15.0        # rate-limits the `ign service` subprocess
                                # call -- same overhead concern noted in
                                # hunav_config.py's BRIDGE_UPDATE_RATE_HZ,
                                # bumped up from the bridge's normal 2Hz
                                # since teleop is a single interactive model
PEOPLE_PUBLISH_RATE_HZ = 15.0
TICK_RATE_HZ = 30.0            # how often the main loop integrates position
                                # and drains the keyboard buffer


def _check_bridge_not_running():
    lock_path = '/tmp/hunav_hunav_model_bridge.lock'
    if not os.path.exists(lock_path):
        return
    try:
        with open(lock_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # raises OSError if that PID isn't alive
        print(
            f'ERROR: hunav_model_bridge.py appears to still be running (pid {pid}). '
            f'Stop it first: pkill -9 -f hunav_model_bridge.py',
            file=sys.stderr,
        )
        sys.exit(1)
    except (ValueError, OSError):
        pass  # stale/unreadable lock -- fine to proceed


def set_model_pose(x: float, y: float, yaw: float) -> bool:
    """Identical to hunav_model_bridge.py's own _set_model_pose."""
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    req = (f'name: "{MODEL_NAME}" '
           f'position: {{x: {x:.4f}, y: {y:.4f}, z: {STANDIN_Z}}} '
           f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}')
    try:
        result = subprocess.run(
            ['ign', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
             '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean',
             '--timeout', str(SET_POSE_TIMEOUT_MS), '--req', req],
            capture_output=True, text=True, timeout=2.0,
        )
        return 'true' in result.stdout
    except subprocess.TimeoutExpired:
        return False


class _RawTerminal:
    """Puts stdin into cbreak mode (single chars, no Enter needed) for the
    life of the `with` block, restoring the original settings on exit even
    if an exception or Ctrl-C interrupts it."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def _drain_keys():
    """Returns every character currently waiting in stdin's input buffer
    (possibly empty), without blocking. Draining the WHOLE buffer each tick
    -- rather than one character per tick -- is what stops a long keyhold
    from leaving a backlog that keeps triggering movement after release."""
    keys = []
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            break
        ch = sys.stdin.read(1)
        if not ch:
            break
        keys.append(ch)
    return keys


class ManualPedestrianTeleop(Node):
    def __init__(self, start_x: float, start_y: float, start_yaw: float):
        super().__init__('manual_pedestrian_teleop')
        self.people_pub = self.create_publisher(People, '/people', 10)

        self.x = start_x
        self.y = start_y
        self.yaw = start_yaw
        self.vx = 0.0
        self.vy = 0.0

        self.linear_speed = DEFAULT_LINEAR_SPEED
        self.angular_speed = DEFAULT_ANGULAR_SPEED

        now = time.monotonic()
        self._last_press = {k: -math.inf for k in 'wsadqe'}
        self._last_tick_time = now
        self._last_set_pose_time = 0.0
        self._last_people_publish_time = 0.0

        self.get_logger().info(
            f'manual_pedestrian_teleop ready at world=({start_x:.2f}, '
            f'{start_y:.2f}, yaw={math.degrees(start_yaw):.1f}deg). '
            f'speed: linear={self.linear_speed:.2f} m/s, '
            f'angular={math.degrees(self.angular_speed):.0f} deg/s'
        )

    def _held(self, key: str, now: float) -> bool:
        return (now - self._last_press[key]) < HOLD_TIMEOUT_SEC

    def handle_key(self, key: str) -> bool:
        """Applies one keypress. Returns False on quit."""
        now = time.monotonic()
        lower = key.lower()

        if lower in self._last_press:
            self._last_press[lower] = now
        elif key in ('+', '='):
            self.linear_speed = min(self.linear_speed * SPEED_STEP_FACTOR, MAX_LINEAR_SPEED)
            self.angular_speed = min(self.angular_speed * SPEED_STEP_FACTOR, MAX_ANGULAR_SPEED)
            print(
                f'\rspeed: linear={self.linear_speed:.2f} m/s, '
                f'angular={math.degrees(self.angular_speed):.0f} deg/s          '
            )
        elif key == '-':
            self.linear_speed = max(self.linear_speed / SPEED_STEP_FACTOR, MIN_LINEAR_SPEED)
            self.angular_speed = max(self.angular_speed / SPEED_STEP_FACTOR, MIN_ANGULAR_SPEED)
            print(
                f'\rspeed: linear={self.linear_speed:.2f} m/s, '
                f'angular={math.degrees(self.angular_speed):.0f} deg/s          '
            )
        elif key == ' ':
            print(
                f'\rpose: x={self.x:.3f} y={self.y:.3f} yaw={math.degrees(self.yaw):.1f}deg '
                f'| speed: linear={self.linear_speed:.2f} m/s, '
                f'angular={math.degrees(self.angular_speed):.0f} deg/s          '
            )
        elif key in ('x', '\x03'):  # 'x' or Ctrl-C
            return False

        return True

    def tick(self):
        """Integrates position from current velocity and handles rate-limited
        set_pose / /people publishing. Called every ~1/TICK_RATE_HZ sec."""
        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        vx = 0.0
        vy = 0.0
        if self._held('w', now):
            vx += self.linear_speed
        if self._held('s', now):
            vx -= self.linear_speed
        if self._held('d', now):
            vy += self.linear_speed
        if self._held('a', now):
            vy -= self.linear_speed

        self.x += vx * dt
        self.y += vy * dt

        # /people carries position + velocity only (no orientation field) --
        # heading_smoother.py derives the heading/arrow from the VELOCITY
        # direction, exactly like a real HuNav-driven pedestrian, who faces
        # the way they're walking. So the mesh's visual orientation tracks
        # direction of travel while actually moving. q/e turn it in place
        # while stationary (self.yaw is what set_pose sends to Gazebo either
        # way, so the visual always matches self.yaw).
        if vx != 0.0 or vy != 0.0:
            self.yaw = math.atan2(vy, vx)
        else:
            wz = 0.0
            if self._held('q', now):
                wz += self.angular_speed
            if self._held('e', now):
                wz -= self.angular_speed
            self.yaw += wz * dt

        # What we PUBLISH on /people can differ from actual motion: while
        # genuinely moving, report the real velocity as always. While
        # stationary, report a small phantom velocity pointing in the
        # current self.yaw (see PHANTOM_HEADING_SPEED above) so
        # heading_smoother.py's arrow tracks whatever heading q/e just set,
        # even though the pedestrian isn't actually translating -- this is
        # what lets you test one fixed position at many different headings.
        if vx != 0.0 or vy != 0.0:
            self.vx = vx
            self.vy = vy
        else:
            self.vx = PHANTOM_HEADING_SPEED * math.cos(self.yaw)
            self.vy = PHANTOM_HEADING_SPEED * math.sin(self.yaw)

        if (now - self._last_set_pose_time) >= (1.0 / SET_POSE_RATE_HZ):
            ok = set_model_pose(self.x, self.y, self.yaw)
            if not ok:
                self.get_logger().warn('set_pose did not report success.')
            self._last_set_pose_time = now

        if (now - self._last_people_publish_time) >= (1.0 / PEOPLE_PUBLISH_RATE_HZ):
            self._publish_people()
            self._last_people_publish_time = now

    def _publish_people(self):
        msg = People()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        person = Person()
        person.name = 'agent1'
        person.position.x, person.position.y, person.position.z = self.x, self.y, 0.0
        person.velocity.x, person.velocity.y, person.velocity.z = self.vx, self.vy, 0.0
        person.reliability = 1.0
        msg.people = [person]
        self.people_pub.publish(msg)


def main():
    _check_bridge_not_running()

    # Starting pose: override with CLI args if given:
    #   python3 manual_pedestrian_teleop.py [x y yaw_deg]
    start_x, start_y, start_yaw = 2.0, 2.0, 0.0
    if len(sys.argv) == 4:
        start_x, start_y = float(sys.argv[1]), float(sys.argv[2])
        start_yaw = math.radians(float(sys.argv[3]))

    rclpy.init()
    node = ManualPedestrianTeleop(start_x, start_y, start_yaw)

    print(
        'Manual pedestrian teleop.\n'
        '  w/a/s/d   move (world +x/-y/-x/+y) -- hold for continuous motion\n'
        '  q/e       rotate in place +/- (only while stopped -- while moving\n'
        '            it auto-faces the direction of travel)\n'
        '  + / -     increase / decrease speed\n'
        '  space     print current pose and speed\n'
        '  x         quit (Ctrl-C also works)\n'
    )
    tick_period = 1.0 / TICK_RATE_HZ
    try:
        with _RawTerminal():
            running = True
            while running:
                for key in _drain_keys():
                    running = node.handle_key(key)
                    if not running:
                        break
                if not running:
                    break
                node.tick()
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(tick_period)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('\nDone. Restart hunav_model_bridge.py to resume normal autonomous behavior.')


if __name__ == '__main__':
    main()