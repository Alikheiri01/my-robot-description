#!/usr/bin/env python3
"""
compare_actor_pose.py -- does Gazebo's actor actually sit where HuNav says
the pedestrian is?

WHY THIS EXISTS
check_actor_motion.py only reads /people, i.e. HuNav's own numbers. Its own
docstring says it "CANNOT prove the stream matches what Gazebo actually
renders" -- that needs a second source read from Gazebo itself. This is that
second source.

There are three different "poses" in play, and they are easy to mix up:
  A. HuNav's pose        -> /people (what the dataset labels use)
  B. The COMMANDED pose  -> what hunav_model_bridge.py sends via set_pose.
                            Gazebo's inspector shows this as "World Pose Cmd".
                            It is just a copy of A, so it SHOULD keep changing.
  C. The APPLIED pose    -> the pose the HuNavActorDriver plugin actually wrote
                            into the actor (its TrajectoryPose), published by
                            the plugin on /model/<actor>/applied_pose. This is
                            what the renderer -- and so the depth cameras --
                            draws. (Gazebo's own /world/<world>/pose/info does
                            NOT carry actors at all, which is why the plugin
                            publishes this.)

This script logs A and C side by side on every /people message and flags
every moment where C does not follow A.

Timing note: hunav_agent_manager publishes /people BEFORE it ticks, from the
state the bridge just sent in -- i.e. from the pose the bridge already
commanded one cycle (~0.5 s) earlier. So at the moment a /people message
arrives, Gazebo has had ~0.5 s to apply that exact pose, and A and C should
agree closely (a few cm). A large gap means Gazebo is not applying the
commanded pose.

Independent checks that do not rely on the plugin's own report: the
Component Inspector's "World Pose" (written by the renderer from what it
drew) should follow "World Pose Cmd", and with the bridge killed the actor
must stand still.

USAGE (with the simulation already running)
    python3 compare_actor_pose.py                 # /model/<actor>/applied_pose
    python3 compare_actor_pose.py <gz_pose_topic> # any Pose_V topic
Stop with Ctrl+C -- NOT Ctrl+Z (Ctrl+Z only suspends it and you get no summary).
"""
import math
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from people_msgs.msg import People

from hunav_config import WORLD_NAME, MODEL_NAME


# A and C should agree to within a few cm (see "Timing note" above).
DIVERGE_M = 0.20
# Below this, a position counts as "not moved".
MOVE_EPS_M = 0.02
# Gazebo-side displacement faster than this between two Gazebo samples is a
# teleport, not walking (max_vel is 1.0 m/s). In the plugin's snap mode the
# actor legitimately jumps one bridge step (~0.5 m) at a time, so only jumps
# above 1 m are flagged (e.g. the old "reappears at the start" behaviour).
GZ_TELEPORT_SPEED = 3.0
GZ_TELEPORT_MIN_M = 1.0
# If the newest Gazebo sample is older than this, Gazebo data stopped arriving
# (a "freeze" seen then is a data gap, not a real freeze).
GZ_STALE_SEC = 1.0
# Warn if the actor has not appeared on the Gazebo topic after this long.
FIND_ACTOR_TIMEOUT_SEC = 5.0

PRINT_LOCK = threading.Lock()


def say(text):
    with PRINT_LOCK:
        print(text, flush=True)


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PoseVTextParser:
    """
    Streaming parser for `ign topic -e` text output of ignition.msgs.Pose_V.
    Feed it one line at a time; it returns a finished pose dict whenever a
    top-level `pose { ... }` block closes, else None.

    Protobuf text format OMITS fields whose value is 0, so every missing
    number defaults to 0.0. A missing orientation block means identity.
    """

    def __init__(self):
        self.depth = 0
        self.cur = None      # pose block being collected
        self.section = None  # 'position' / 'orientation' / other, inside it

    def feed(self, raw_line):
        line = raw_line.strip()
        if not line:
            return None

        if line.endswith('{'):
            key = line[:-1].strip()
            if self.depth == 0 and key == 'pose':
                self.cur = {'name': None, 'pos': {}, 'ori': None}
            elif self.cur is not None and self.depth == 1:
                self.section = key
                if key == 'orientation':
                    self.cur['ori'] = {}
            self.depth += 1
            return None

        if line == '}':
            self.depth -= 1
            if self.cur is not None and self.depth == 1:
                self.section = None
            if self.cur is not None and self.depth == 0:
                done, self.cur, self.section = self.cur, None, None
                return self._finish(done)
            return None

        if self.cur is not None and ':' in line:
            key, val = (s.strip() for s in line.split(':', 1))
            if self.depth == 1 and key == 'name':
                self.cur['name'] = val.strip('"')
            elif self.depth == 2 and self.section == 'position' and key in ('x', 'y', 'z'):
                self.cur['pos'][key] = float(val)
            elif self.depth == 2 and self.section == 'orientation' and key in ('x', 'y', 'z', 'w'):
                self.cur['ori'][key] = float(val)
        return None

    @staticmethod
    def _finish(block):
        p = block['pos']
        o = block['ori']
        if o is None:
            yaw = 0.0
        else:
            yaw = yaw_from_quat(o.get('x', 0.0), o.get('y', 0.0),
                                o.get('z', 0.0), o.get('w', 0.0))
        return {'name': block['name'], 'x': p.get('x', 0.0), 'y': p.get('y', 0.0), 'yaw': yaw}


class GazeboPoseReader(threading.Thread):
    """Runs `ign topic -e` in the background and keeps the actor's latest pose."""

    def __init__(self, topic, model_name, t0):
        super().__init__(daemon=True)
        self.topic = topic
        self.model_name = model_name
        self.t0 = t0
        self.lock = threading.Lock()
        self.latest = None          # (x, y, yaw, wall_time)
        self.blocks_seen = 0
        self.actor_samples = 0
        self.other_names = set()
        self.teleports = []         # (t, (x0, y0), (x1, y1), dist, speed)
        self.proc = None

    def run(self):
        # stdbuf -oL: force line buffering so samples arrive as they happen,
        # not in delayed bursts (the same pipe-buffering trap as before).
        self.proc = subprocess.Popen(
            ['stdbuf', '-oL', 'ign', 'topic', '-e', '-t', self.topic],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        parser = PoseVTextParser()
        prev = None
        for raw in self.proc.stdout:
            pose = parser.feed(raw)
            if pose is None:
                continue
            now = time.time()
            with self.lock:
                self.blocks_seen += 1
                if pose['name'] != self.model_name:
                    if len(self.other_names) < 40:
                        self.other_names.add(pose['name'])
                    continue
                self.actor_samples += 1
                self.latest = (pose['x'], pose['y'], pose['yaw'], now)
            if prev is not None:
                d = math.hypot(pose['x'] - prev[0], pose['y'] - prev[1])
                gap = now - prev[2]
                speed = d / gap if gap > 1e-6 else float('inf')
                if d > GZ_TELEPORT_MIN_M and speed > GZ_TELEPORT_SPEED:
                    t = now - self.t0
                    with self.lock:
                        self.teleports.append((t, (prev[0], prev[1]), (pose['x'], pose['y']), d, speed))
                    say(f'  !!! GAZEBO TELEPORT at t={t:6.2f}s: ({prev[0]:5.2f},{prev[1]:5.2f}) -> '
                        f'({pose["x"]:5.2f},{pose["y"]:5.2f})  {d:.2f} m in {gap:.3f} s')
            prev = (pose['x'], pose['y'], now)

    def snapshot(self):
        with self.lock:
            return self.latest, self.blocks_seen, self.actor_samples, set(self.other_names)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class ActorPoseComparer(Node):
    def __init__(self, gz_topic):
        super().__init__('compare_actor_pose')
        self.t0 = time.time()
        self.gz_topic = gz_topic
        self.gz = GazeboPoseReader(gz_topic, MODEL_NAME, self.t0)
        self.gz.start()

        self.n = 0
        self.last_people = None
        self.last_gz = None
        self.last_t = None
        self.dists = []
        self.diverged = 0
        self.not_following = 0
        self.nf_run_start = None
        self.nf_longest = 0.0
        self.stale = 0
        self.warned_missing = False

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(People, '/people', self._cb, qos)

        say('=' * 118)
        say(f'A = /people (HuNav)     C = pose applied to actor "{MODEL_NAME}" (from {gz_topic})')
        say(f'Expect |A-C| < {DIVERGE_M} m on every row. Stop with Ctrl+C (NOT Ctrl+Z).')
        say('=' * 118)
        say('   idx      t   A: /people pos     C: gazebo pos      |A-C|   A_step  C_step  C_yaw   flags')

    def _cb(self, msg: People):
        now = time.time()
        t = now - self.t0
        if not msg.people:
            return
        p = msg.people[0]
        ax, ay = p.position.x, p.position.y

        latest, blocks, actor_n, others = self.gz.snapshot()
        if latest is None:
            if not self.warned_missing and t > FIND_ACTOR_TIMEOUT_SEC:
                self.warned_missing = True
                if blocks == 0:
                    say(f'  WARNING: no poses at all on {self.gz_topic} after {t:.0f}s. '
                        f'Is the HuNavActorDriver plugin loaded? Look for its "[HuNavActorDriver] '
                        f'Driving actor" line in the launch terminal; `ign topic -l | grep applied` '
                        f'lists its topic.')
                else:
                    say(f'  WARNING: "{MODEL_NAME}" never appears on {self.gz_topic}. '
                        f'Names seen there: {sorted(others)[:20]}')
                    say('           Is the HuNavActorDriver plugin loaded? Look for its '
                        '"[HuNavActorDriver] Driving actor" line in the launch terminal.')
            return

        gx, gy, gyaw, gtime = latest
        age = now - gtime
        dist = math.hypot(ax - gx, ay - gy)
        a_step = math.hypot(ax - self.last_people[0], ay - self.last_people[1]) if self.last_people else float('nan')
        c_step = math.hypot(gx - self.last_gz[0], gy - self.last_gz[1]) if self.last_gz else float('nan')

        flags = []
        if age > GZ_STALE_SEC:
            self.stale += 1
            flags.append(f'GZ_STALE({age:.1f}s)')
        if dist > DIVERGE_M:
            self.diverged += 1
            flags.append('DIVERGED')
        following = True
        if self.last_people is not None and a_step > MOVE_EPS_M and c_step < MOVE_EPS_M:
            self.not_following += 1
            flags.append('GZ_NOT_FOLLOWING')
            following = False
            if self.nf_run_start is None:
                self.nf_run_start = self.last_t
        if following and self.nf_run_start is not None:
            self.nf_longest = max(self.nf_longest, now - self.nf_run_start)
            self.nf_run_start = None

        self.dists.append(dist)
        say(f'  {self.n:5d}  {t:6.2f}   ({ax:5.2f},{ay:5.2f})     ({gx:5.2f},{gy:5.2f})     '
            f'{dist:5.2f}   {a_step:5.2f}   {c_step:5.2f}  {math.degrees(gyaw):6.1f}   {" ".join(flags)}')

        self.n += 1
        self.last_people = (ax, ay)
        self.last_gz = (gx, gy)
        self.last_t = now

    def summary(self):
        latest, blocks, actor_n, others = self.gz.snapshot()
        if self.nf_run_start is not None and self.last_t is not None:
            self.nf_longest = max(self.nf_longest, self.last_t - self.nf_run_start)
        say('')
        say('=' * 118)
        say('SUMMARY')
        dur = time.time() - self.t0
        say(f'  Gazebo actor samples     : {actor_n} over {dur:.1f}s '
            f'({actor_n / dur if dur > 0 else 0:.1f} Hz) on {self.gz_topic}')
        if not self.dists:
            say('  No comparisons made (no /people messages, or the actor never appeared on the Gazebo topic).')
            say('=' * 118)
            return
        ds = sorted(self.dists)
        say(f'  Compared samples         : {len(ds)}')
        say(f'  |A-C| median / mean / max: {ds[len(ds) // 2]:.3f} / {sum(ds) / len(ds):.3f} / {ds[-1]:.3f} m')
        say(f'  DIVERGED (>{DIVERGE_M} m)       : {self.diverged}')
        say(f'  GZ_NOT_FOLLOWING         : {self.not_following}  (longest stretch {self.nf_longest:.1f}s)')
        say(f'  Gazebo teleports         : {len(self.gz.teleports)}')
        for t, a, b, d, s in self.gz.teleports:
            say(f'      t={t:6.2f}s  ({a[0]:.2f},{a[1]:.2f}) -> ({b[0]:.2f},{b[1]:.2f})  {d:.2f} m')
        say(f'  GZ_STALE samples         : {self.stale}')
        say('')
        if self.diverged == 0 and self.not_following == 0 and not self.gz.teleports:
            say('  VERDICT: Gazebo actor follows /people. The pose numbers and the rendered actor agree.')
        else:
            say('  VERDICT: Gazebo actor does NOT follow /people on the flagged rows.')
            say('           /people (HuNav) and the actor the cameras see are in different places there.')
        say('=' * 118)


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else f'/model/{MODEL_NAME}/applied_pose'
    rclpy.init()
    node = ActorPoseComparer(topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.gz.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()