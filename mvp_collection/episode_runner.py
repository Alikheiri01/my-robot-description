#!/usr/bin/env python3
"""
episode_runner.py -- P5 of the MVP collection: plays the planned episodes of one or
more environments in the running simulation and records them (2026-11-01).

NEEDS: my_robot_hunav_full.launch.py running with the NEW bridge (episode mode,
hunav_model_bridge_nav.py of 2026-11-01). The runner starts robot_driver.py itself.

PER ENVIRONMENT
  1. robot and pedestrian teleported to the first episode's starts (the environment
     is 4 m away from the launch poses, so nothing is inside a wall when it appears)
  2. spawn_obstacles.py scene.yaml; wait until the bridge's current_obstacles.yaml lists
     exactly these obstacles
  3. record_session.py --name <env_id> started (unless --no-record)
  4. every episode:
       'reset_<id>' on /dataset/episode; robot stop; teleport robot (ign set_pose) and
       pedestrian (/hunav/reset_pose); check both arrived (robot: /odom AND Gazebo's own
       model pose, height included); robot mode sent to the driver;
       wait until the start gate is OPEN again; ONE goal + speed to the bridge
       (/hunav/episode_goal, path mode for E arcs, slowdown for C)
       walking (> 0.3 m/s)               -> '<id>' on /dataset/episode   (the episode starts)
       1.5 m of route left (A/B/D/E)     -> 'end_<id>'                    (before HuNav brakes)
       C: arrived, then stood 7.5 s      -> 'end_<id>'
       arrival = < 0.1 m/s within 1 m of the goal; abort on timeout or stuck
       event time measured: the pedestrian passes the event's point of the planned route
       (C: the arrival); written to episodes/<id>.json with all planned fields
  5. record_session.py stopped like Ctrl+C (it writes report.txt), the run folder moved to
     <env dir>/runs/, the scene removed

TRACE (2026-11-02): the runner starts check_scripts/trace_episode.py itself for every
environment (true robot track, stuck, usable share, one PNG per episode; --gif adds
animations) and stops it before the recording is stopped.

WHERE THINGS GO (one folder per run)
  recorded      datasets/<dataset>/<scenario>/<env>/runs/<run>/    samples/, report.txt (record_session),
                                                     episodes/<id>.json, trace/, runner.log
  --no-record   datasets/<dataset>/<scenario>/<env>/dry_runs/<time>/   episodes/, trace/, runner.log
  datasets/<dataset>/logs/   only the whole-session logs: runner_<time>.log, robot_driver_<time>.log

USAGE (from mvp_collection/)
    python3 episode_runner.py datasets/dev/E_walk/env_E_001 --episodes 1 --no-record   # watch one episode
    python3 episode_runner.py datasets/dev/A_detour/env_A_000                          # one environment, recorded
    python3 episode_runner.py datasets/dev/A_detour/env_A_000 --gif                    # + animations
    python3 episode_runner.py datasets/mvp_v1/A_detour datasets/mvp_v1/B_corner --skip-done          # the collection
Ctrl+C: stops after cleaning up (robot stopped, recording stopped and reported, scene removed).
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
AC = PKG / 'agent_control'
DC = PKG / 'dataset_codes'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AC))
sys.path.insert(0, str(PKG / 'hunav_codes'))

import mvp_config as C  # noqa: E402
import plan_model as M  # noqa: E402
from hunav_config import WORLD_NAME, odom_to_world, world_to_odom, yaw_to_quaternion  # noqa: E402

CURRENT_OBSTACLES = AC / 'current_obstacles.yaml'


class Abort(Exception):
    pass


def ign_set_pose(name, x, y, z, yaw):
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    req = (f'name: "{name}" position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}} '
           f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}')
    cmd = ['ign', 'service', '-s', f'/world/{WORLD_NAME}/set_pose', '--reqtype', 'ignition.msgs.Pose',
           '--reptype', 'ignition.msgs.Boolean', '--timeout', '3000', '--req', req]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=6.0)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        return False
    return 'true' in out


def ign_remove_model(name):
    """Removes one model from Gazebo by name. True if Gazebo answered true."""
    cmd = ['ign', 'service', '-s', f'/world/{WORLD_NAME}/remove', '--reqtype', 'ignition.msgs.Entity',
           '--reptype', 'ignition.msgs.Boolean', '--timeout', '3000', '--req', f'name: "{name}" type: MODEL']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=6.0)
    except subprocess.TimeoutExpired:
        return False
    return 'true' in r.stdout


def ign_model_pose(name):
    """Gazebo's own pose of a model (ign model -m NAME -p): (x, y, z, yaw) world, or None.
    FIXED (2026-11-01): the robot's height comes from here. /robot_truth/odometry is 2D
    odometry, its z is always 0, and teleporting with that height put the robot 2 cm high,
    i.e. with its wheels inside the floor (stuck while the wheels spin)."""
    try:
        r = subprocess.run(['ign', 'model', '-m', name, '-p'], capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = r.stdout[r.stdout.find('Pose'):] if 'Pose' in r.stdout else ''
    triples = re.findall(r'\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]', text)
    if len(triples) < 2:
        return None
    (x, y, z), (_, _, yaw) = [tuple(map(float, t)) for t in triples[:2]]
    return x, y, z, yaw


def read_current_obstacles():
    try:
        from world_obstacles import load_obstacles
        return load_obstacles(CURRENT_OBSTACLES)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', help='environment folders, or scenario folders (all environments inside)')
    ap.add_argument('--episodes', default='all', help='"all" or e.g. "0,2"')
    ap.add_argument('--no-record', action='store_true', help='play only, do not run record_session.py')
    ap.add_argument('--skip-done', action='store_true', help='skip environments that already have a recorded run')
    ap.add_argument('--gif', action='store_true', help='the trace also writes one animation per episode (slow)')
    args = ap.parse_args()

    env_dirs = []
    for d in map(Path, args.dirs):
        d = d.resolve()
        if (d / 'environment.json').exists():
            env_dirs.append(d)
        else:
            env_dirs += sorted(p for p in d.iterdir() if (p / 'environment.json').exists())
    if not env_dirs:
        sys.exit('No environment folders given (folders with environment.json).')
    if args.skip_done:
        env_dirs = [d for d in env_dirs if not any((r / 'report.txt').exists() for r in (d / 'runs').glob('*'))]
    # session logs go to the dataset of the first environment: datasets/<dataset>/logs
    LOG_DIR = env_dirs[0].parents[1] / 'logs'
    LOG_DIR.mkdir(exist_ok=True)

    import rclpy
    from rclpy.clock import Clock
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Pose, PoseArray
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from hunav_config import ensure_single_instance, yaw_from_quaternion

    ensure_single_instance('episode_runner')

    class Runner(Node):
        def __init__(self):
            super().__init__('episode_runner', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.ped = deque(maxlen=200)      # (t, x, y) odom
            self.robot = None                 # (x, y, yaw) odom
            self.robot_z = None               # resting height of the robot in Gazebo (ign model -p)
            self.bstate = {}                  # last /hunav/episode_state
            self.gate = {}
            self.episode_id = 'idle'
            self.ep_pub = self.create_publisher(String, '/dataset/episode', 10)
            self.goal_pub = self.create_publisher(String, '/hunav/episode_goal', 10)
            self.reset_pub = self.create_publisher(Pose, '/hunav/reset_pose', 10)
            self.robot_pub = self.create_publisher(String, '/mvp/robot_cmd', 10)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped_cb, 10)
            self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
            self.create_subscription(String, '/hunav/episode_state', self._bstate_cb, 10)
            self.create_subscription(String, '/dataset/gate', self._gate_cb, 10)
            self.create_timer(1.0, self._repeat_id, clock=Clock())   # wall clock: survives a paused sim

        # ---- inputs
        def _ped_cb(self, msg):
            if msg.poses:
                t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
                self.ped.append((t, msg.poses[0].position.x, msg.poses[0].position.y))

        def _odom_cb(self, msg):
            p = msg.pose.pose
            self.robot = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

        def _bstate_cb(self, msg):
            try:
                self.bstate = json.loads(msg.data)
            except ValueError:
                pass

        def _gate_cb(self, msg):
            try:
                self.gate = json.loads(msg.data)
            except ValueError:
                pass

        # ---- helpers
        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def set_id(self, eid):
            self.episode_id = eid
            self.ep_pub.publish(String(data=eid))

        def _repeat_id(self):
            self.ep_pub.publish(String(data=self.episode_id))

        def ped_pos(self):
            return (self.ped[-1][1], self.ped[-1][2]) if self.ped else None

        def ped_speed(self, window=0.5):
            if len(self.ped) < 2:
                return None
            t1, x1, y1 = self.ped[-1]
            for t0, x0, y0 in reversed(self.ped):
                if t1 - t0 >= window:
                    return math.hypot(x1 - x0, y1 - y0) / (t1 - t0)
            return None

        def robot_cmd(self, d):
            self.robot_pub.publish(String(data=json.dumps(d)))

    rclpy.init()
    node = Runner()
    # FIXED (2026-11-02): an executor that is shut down before rclpy (the plain rclpy.spin
    # thread was still running at exit: 'terminate called without an active exception').
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_f = open(LOG_DIR / f'runner_{stamp}.log', 'a')

    run_log = {'f': None}           # per-run copy of the log (in the run folder)

    def say(msg):
        line = f'[{datetime.now().strftime("%H:%M:%S")} sim {node.now():8.1f}] {msg}'
        print(line, flush=True)
        for f in (log_f, run_log['f']):
            if f is not None:
                f.write(line + '\n')
                f.flush()

    stop = {'flag': False}

    def on_sigint(sig, frm):
        if stop['flag']:
            raise KeyboardInterrupt
        stop['flag'] = True
        say('Ctrl+C: finishing this episode cleanly, then stopping (Ctrl+C again = stop now)')
    signal.signal(signal.SIGINT, on_sigint)

    def wait(cond, sim_sec=None, wall_sec=None, what=''):
        """Wait until cond() is true. Raises Abort on a timeout (sim or wall seconds)."""
        t0w, t0s = time.monotonic(), node.now()
        while True:
            if cond():
                return
            if wall_sec is not None and time.monotonic() - t0w > wall_sec:
                raise Abort(f'timeout waiting for {what} ({wall_sec:.0f} s wall)')
            if sim_sec is not None and node.now() - t0s > sim_sec:
                raise Abort(f'timeout waiting for {what} ({sim_sec:.0f} s sim)')
            time.sleep(0.05)

    def sim_sleep(sec):
        t0 = node.now()
        wait(lambda: node.now() - t0 >= sec, wall_sec=sec / 0.05 + 30, what='simulation time')

    # ---- start the robot driver ---------------------------------------------------
    drv_log = open(LOG_DIR / f'robot_driver_{stamp}.log', 'w')
    driver = subprocess.Popen([sys.executable, str(HERE / 'robot_driver.py')], cwd=str(HERE), stdout=drv_log,
                              stderr=subprocess.STDOUT, start_new_session=True,
                              env=dict(os.environ, PYTHONUNBUFFERED='1'))
    say(f'robot_driver.py started (log {drv_log.name})')
    try:
        wait(lambda: node.ped_pos() is not None and node.robot is not None,
             wall_sec=30, what='pedestrian pose and /odom (is the launch running?)')
        node.robot_cmd({'mode': 'stop'})
        sim_sleep(1.0)
        rest = ign_model_pose(C.ROBOT_NAME)
        if rest is None:
            raise Abort(f'ign model -m {C.ROBOT_NAME} -p gave no pose (robot name? world running?)')
        node.robot_z = rest[2]
        say(f'robot resting height in Gazebo: z = {node.robot_z:.3f} m (used for every teleport)')
    except Abort as e:
        say(f'NOT STARTED: {e}')
        driver.send_signal(signal.SIGINT)
        sys.exit(1)

    # ---- one episode ----------------------------------------------------------------
    def teleport(ep):
        rx, ry, rh = ep['robot']['start_pose']
        sx, sy = ep['start']
        r0 = ep['route']
        ph = math.atan2(r0[1][1] - r0[0][1], r0[1][0] - r0[0][0])
        node.robot_cmd({'mode': 'stop'})
        sim_sleep(0.5)
        wx, wy, wyaw = odom_to_world(rx, ry, rh)
        for attempt in range(3):
            if not ign_set_pose(C.ROBOT_NAME, wx, wy, node.robot_z + 0.02, wyaw):
                say(f'  teleport attempt {attempt + 1}: ign set_pose did not answer true')
            pose = Pose()
            px, py, pyaw = odom_to_world(sx, sy, ph)
            pose.position.x, pose.position.y = px, py
            pose.orientation.z, pose.orientation.w = math.sin(pyaw / 2), math.cos(pyaw / 2)
            node.reset_pub.publish(pose)
            sim_sleep(C.RESET_SETTLE_SIM_SEC)
            pe = math.dist(node.ped_pos(), (sx, sy))
            re_ = math.dist(node.robot[:2], (rx, ry))
            g = ign_model_pose(C.ROBOT_NAME)            # Gazebo's own view, independent of /odom
            ge = math.dist(g[:2], (wx, wy)) if g else math.inf
            dz = (g[2] - node.robot_z) if g else math.inf
            if pe < 0.3 and re_ < 0.25 and ge < 0.25 and abs(dz) < 0.05:
                return pe, re_
            say(f'  teleport attempt {attempt + 1}: pedestrian {pe:.2f} m, robot {re_:.2f} m (/odom) / '
                f'{ge:.2f} m (Gazebo) from the target, height {dz:+.3f} m vs resting; retrying')
        raise Abort(f'teleport failed (pedestrian {pe:.2f} m, robot {re_:.2f} m /odom, {ge:.2f} m Gazebo, '
                    f'height {dz:+.3f} m)')

    def run_episode(env, ep, out_dir):
        eid = ep['episode_id']
        meas = {'outcome': None}
        node.set_id(f'reset_{eid}')
        meas['t_reset_sim'] = node.now()
        meas['ped_reset_error_m'], meas['robot_reset_error_m'] = [round(v, 3) for v in teleport(ep)]
        rb = ep['robot']
        if rb['mode'] == 'still':
            node.robot_cmd({'mode': 'still', 'heading': rb['start_pose'][2]})
        elif rb['mode'] == 'watch':
            node.robot_cmd({'mode': 'watch'})
        else:
            node.robot_cmd({'mode': 'follow', 'route': ep['route'], 'dist': rb['dist']})
        t_tp = node.now()
        wait(lambda: node.gate.get('open') and node.gate.get('now', 0) > t_tp + 0.5,
             wall_sec=C.GATE_WAIT_WALL_SEC, what='the start gate to open after the teleport')
        gx, gy, _ = odom_to_world(*ep['goal'], 0.0)
        goal = {'id': eid, 'goal': [gx, gy], 'speed': ep['speed'], 'path': None, 'stop_slowdown_m': None}
        if ep['path_mode']:
            goal['path'] = [list(odom_to_world(x, y, 0.0)[:2]) for x, y in ep['route']]
        if ep['stop']:
            goal['stop_slowdown_m'] = ep['stop']['slowdown_m']
        node.goal_pub.publish(String(data=json.dumps(goal)))
        meas['t_goal_sim'] = node.now()
        node.ped.clear()
        wait(lambda: (node.ped_speed() or 0.0) > C.WALK_START_SPEED, sim_sec=15, wall_sec=120,
             what='the pedestrian to start walking')
        node.set_id(eid)
        t0 = meas['t_walk_start_sim'] = node.now()
        say(f'  {eid}: walking ({ep["speed"]:.2f} m/s, robot {rb["mode"]}'
            f'{" d=%.1f" % rb["dist"] if rb["mode"] == "follow" else ""})')
        pl = M.Polyline(ep['route'])
        ev = ep['event']
        timeout = C.TIMEOUT_FACTOR * ep['planned']['t_end_sec'] + 20.0
        ended = arrived = False
        slow_since = None
        while True:
            time.sleep(0.05)
            t = node.now()
            p, v = node.ped_pos(), node.ped_speed() or 0.0
            d_goal = math.dist(p, ep['goal'])
            if ev and ev['type'] != 'stop' and 't_event_sim' not in meas and pl.project(p)[0] >= ev['s']:
                meas['t_event_sim'] = t
            rem = node.bstate.get('remaining_m') if node.bstate.get('id') == eid else None
            if not ended and not ep['stop'] and rem is not None and rem <= C.END_TRIGGER_M:
                node.set_id(f'end_{eid}')
                meas['t_end_sim'], ended = t, True
            if not arrived and d_goal <= C.ARRIVE_DIST_M and v < C.ARRIVE_SPEED:
                meas['t_arrival_sim'], arrived = t, True
                if ep['stop']:
                    meas['t_event_sim'] = t
                    sim_sleep(ep['stop']['stand_sec'])
                    node.set_id(f'end_{eid}')
                    meas['t_end_sim'], ended = node.now(), True
            if arrived and ended:
                meas['outcome'] = 'ok'
                break
            if arrived and not ended:             # the bridge never reported remaining_m (old bridge?)
                node.set_id(f'end_{eid}')
                meas['t_end_sim'], ended = t, True
                meas['outcome'] = 'ok_late_end'
                say(f'  {eid}: WARNING: no remaining_m from the bridge; ended only at the arrival '
                    f'(the samples before it are dropped by the recorder anyway, but check the bridge)')
                break
            slow_since = (slow_since or t) if (v < C.ARRIVE_SPEED and d_goal > C.ARRIVE_DIST_M) else None
            if slow_since is not None and t - slow_since > C.STUCK_SEC:
                meas['outcome'] = 'stuck'
                break
            if t - t0 > timeout:
                meas['outcome'] = 'timeout'
                break
        if not ended:
            node.set_id(f'end_{eid}')
            meas['t_end_sim'] = node.now()
        node.robot_cmd({'mode': 'stop'})
        for k in ('t_event_sim', 't_end_sim', 't_arrival_sim'):
            if k in meas:
                meas[k.replace('_sim', '_after_walk_start_sec')] = round(meas[k] - t0, 2)
        rec = dict(ep)
        rec.update({'environment_id': env['environment_id'], 'scenario': env['scenario'],
                    'template': env['template'], 'environment_params': env['params'], 'measured': meas})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f'{eid}.json').write_text(json.dumps(rec, indent=1))
        ev_txt = f', event at {meas["t_event_after_walk_start_sec"]:.1f} s (planned {ev["t_planned_sec"]:.1f})' \
            if ev and 't_event_after_walk_start_sec' in meas else ''
        say(f'  {eid}: {meas["outcome"].upper()}, end at {meas.get("t_end_after_walk_start_sec", -1):.1f} s '
            f'(planned {ep["planned"]["t_end_sec"]:.1f}){ev_txt}')
        return meas['outcome']

    # ---- scene removal ----------------------------------------------------------------
    def clear_scene(when):
        """NEW (2026-11-02): removes every obstacle the bridge still sees and waits until the
        bridge reports none. spawn_obstacles.py --remove forgets names whose removal failed,
        so leftovers are then removed by the names the bridge reports. Raises Abort if
        obstacles are still there after that (a leftover wall would break the next scene)."""
        subprocess.run([sys.executable, 'spawn_obstacles.py', '--remove'], cwd=str(AC), capture_output=True)
        empty = lambda: (lambda c: c is not None and len(c) == 0)(read_current_obstacles())  # noqa: E731
        try:
            wait(empty, wall_sec=C.SCENE_WAIT_WALL_SEC, what=f'the scene to be removed ({when})')
            return
        except Abort:
            pass
        left = read_current_obstacles() or []
        say(f'  {len(left)} obstacle(s) still in Gazebo ({when}), removing them by name: '
            + ', '.join(o.name for o in left[:8]) + (' ...' if len(left) > 8 else ''))
        for o in left:
            ign_remove_model(re.sub(r'_\d+$', '', o.name) if not o.name.startswith('obs_') else o.name)
        wait(empty, wall_sec=C.SCENE_WAIT_WALL_SEC, what=f'the leftover obstacles to be removed ({when})')

    # ---- one environment ------------------------------------------------------------
    def run_env(env_dir):
        env = json.loads((env_dir / 'environment.json').read_text())
        eps = env['episodes'] if args.episodes == 'all' else \
            [e for e in env['episodes'] if e['index'] in {int(i) for i in args.episodes.split(',')}]
        say(f'== {env["environment_id"]} ({env["scenario_name"]}, {env["template"]}): {len(eps)} episode(s)')
        clear_scene('before spawning')             # nothing from an earlier environment may remain
        teleport(eps[0])
        r = subprocess.run([sys.executable, 'spawn_obstacles.py', str(env_dir / 'scene.yaml')], cwd=str(AC),
                           capture_output=True, text=True)
        n_want = len(env['obstacles'])
        if f'{n_want}/{n_want} obstacle(s)' not in r.stdout:
            raise Abort(f'spawn failed:\n{r.stdout[-800:]}{r.stderr[-400:]}')
        names = {'obs_' + o['name'] for o in env['obstacles']}
        wait(lambda: (lambda c: c is not None and {o.name for o in c} == names)(read_current_obstacles()),
             wall_sec=C.SCENE_WAIT_WALL_SEC, what='the bridge to report the new scene')
        say(f'  scene spawned and confirmed by the bridge ({n_want} obstacles)')
        rec, run_dir = None, None
        if not args.no_record:
            rlog = open(LOG_DIR / f'record_{env["environment_id"]}_{stamp}.log', 'w')
            rec = subprocess.Popen([sys.executable, 'record_session.py', '--name', env['environment_id']],
                                   cwd=str(DC), stdout=rlog, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, PYTHONUNBUFFERED='1'))
            wait(lambda: re.search(r'Run folder: (\S+)', Path(rlog.name).read_text()) is not None,
                 wall_sec=30, what='record_session.py to start')
            run_dir = Path(re.search(r'Run folder: (\S+)', Path(rlog.name).read_text()).group(1))
            time.sleep(C.RECORD_START_WALL_SEC)
            if rec.poll() is not None:
                raise Abort(f'record_session.py exited at start-up, see {rlog.name}')
            say(f'  recording into {run_dir.name}')
        out_root = run_dir if run_dir else (env_dir / 'dry_runs' / datetime.now().strftime('%Y%m%d_%H%M%S'))
        out_root.mkdir(parents=True, exist_ok=True)
        ep_dir = out_root / 'episodes'
        run_log['f'] = open(out_root / 'runner.log', 'a')
        run_log['f'].write(f'{env["environment_id"]} ({env["scenario_name"]}, {env["template"]}), '
                           f'{len(eps)} episode(s), session log {log_f.name}\n')
        tcmd = [sys.executable, str(HERE / 'check_scripts' / 'trace_episode.py'), str(env_dir),
                '--out', str(out_root / 'trace')] + (['--gif'] if args.gif else [])
        tlog = open(out_root / 'trace_stdout.txt', 'w')
        tracer = subprocess.Popen(tcmd, cwd=str(HERE), stdout=tlog, stderr=subprocess.STDOUT, start_new_session=True,
                                  env=dict(os.environ, PYTHONUNBUFFERED='1'))
        say(f'  trace started -> {out_root / "trace"}')
        time.sleep(3.0)
        outcomes = []
        try:
            for ep in eps:
                if stop['flag']:
                    break
                try:
                    outcomes.append(run_episode(env, ep, ep_dir))
                except Abort as e:
                    say(f'  {ep["episode_id"]}: ABORTED: {e}')
                    node.set_id(f'end_{ep["episode_id"]}')
                    node.robot_cmd({'mode': 'stop'})
                    outcomes.append('aborted')
        finally:
            node.robot_cmd({'mode': 'stop'})
            node.set_id('idle')
            # the trace first: it writes into the run folder, which is moved below
            if tracer.poll() is None:
                os.killpg(tracer.pid, signal.SIGINT)
            try:
                tracer.wait(timeout=600 if args.gif else 180)
            except subprocess.TimeoutExpired:
                os.killpg(tracer.pid, signal.SIGKILL)
            trep = out_root / 'trace' / 'trace_report.txt'
            tres = re.search(r'TRACE: (.*)', trep.read_text()) if trep.exists() else None
            say(f'  trace: {tres.group(1) if tres else "NO TRACE REPORT (see trace_stdout.txt)"}')
            if trep.exists():
                for ln in trep.read_text().splitlines():
                    if ln.strip().startswith(('PASS', 'FAIL')):
                        say('    ' + ln.strip())
            if rec is not None:
                say('  stopping record_session.py (it writes its report, ~1 min)...')
                os.killpg(rec.pid, signal.SIGINT)
                try:
                    rec.wait(timeout=C.RECORD_STOP_WALL_SEC)
                except subprocess.TimeoutExpired:
                    os.killpg(rec.pid, signal.SIGKILL)
                rep = run_dir / 'report.txt'
                overall = re.search(r'OVERALL: (.*)', rep.read_text()) if rep.exists() else None
                dest = env_dir / 'runs' / run_dir.name
                dest.parent.mkdir(exist_ok=True)
                shutil.move(str(run_dir), str(dest))
                rlog.close()
                shutil.copy(rlog.name, dest / 'record_session.log')
                say(f'  report: {overall.group(1) if overall else "NO REPORT"}  -> {dest}')
            try:
                clear_scene('after the environment')
            except Abort as e:
                say(f'  WARNING: {e} (tried again before the next environment)')
        say(f'  {env["environment_id"]} done: ' + ', '.join(f'{o}' for o in outcomes))
        if run_log['f'] is not None:
            run_log['f'].close()
            run_log['f'] = None
        return outcomes

    summary = []
    try:
        for d in env_dirs:
            if stop['flag']:
                break
            try:
                summary.append((d.name, run_env(d)))
            except Abort as e:
                say(f'{d.name}: ENVIRONMENT FAILED: {e}')
                summary.append((d.name, ['failed']))
    finally:
        node.robot_cmd({'mode': 'stop'})
        driver.send_signal(signal.SIGINT)
        try:
            driver.wait(timeout=10)
        except subprocess.TimeoutExpired:
            driver.kill()
        say('SUMMARY: ' + '; '.join(f'{n}: {"/".join(o)}' for n, o in summary))
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()