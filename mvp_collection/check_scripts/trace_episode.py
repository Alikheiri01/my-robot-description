#!/usr/bin/env python3
"""
trace_episode.py -- records what REALLY happens during episode_runner.py episodes and
checks it (2026-11-01). Run it in a second terminal next to the runner.

WHY
  The pipeline uses /odom (the TRUE pose); RViz draws the robot model from the WHEEL
  odometry's /tf, which does not follow teleports. This tool records the truth and shows
  where the robot really drove, whether it got stuck and whether the camera saw the
  pedestrian.

WHAT IT DOES
  live    10 Hz trace (sim time) of: /odom (true pose), /odom_wheel (what RViz shows),
          /people_smoothed_pose, /cmd_vel, /dataset/episode.
  Ctrl+C  saves logs/trace_<time>.csv and checks every episode it saw:
            seen       share of the walk the camera model sees the pedestrian, and the
                       usable share with the training filter (>= 50 % to pass)
            stuck      seconds where /cmd_vel asks to drive (> 0.1 m/s) but the robot
                       does not move (< 0.05 m/s) -- wall contact / wheels slipping
            walls      closest true distance robot centre -> wall (>= robot radius)
            pedestrian closest robot-pedestrian distance (>= 1.0 m)
          and writes one PNG per episode (map + speed/seen over time); --gif also writes
          an animation like sim_robot_driver.py, but with the REAL tracks.

OUTPUT (2026-11-02: one folder per run, nothing in logs/)
    <out>/trace.csv  <out>/trace_report.txt  <out>/<episode_id>.png  (and .gif with --gif)
  episode_runner.py starts this script itself, with --out <run folder>/trace
  (recorded: datasets/<dataset>/<scenario>/<env>/runs/<run>/trace, not recorded: .../dry_runs/<time>/trace).
  Started by hand without --out: datasets/<dataset>/<scenario>/<env>/traces/<time>/

USAGE
    python3 check_scripts/trace_episode.py datasets/dev/A_detour/env_A_000           # live, Ctrl+C at the end
    python3 check_scripts/trace_episode.py datasets/dev/A_detour/env_A_000 --gif --out some/folder
    python3 check_scripts/trace_episode.py datasets/dev/A_detour/env_A_000 --replay .../trace/trace.csv [--gif]
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MVP = HERE.parent
sys.path.insert(0, str(MVP))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MVP.parent / 'agent_control'))
sys.path.insert(0, str(MVP.parent / 'hunav_codes'))

import numpy as np  # noqa: E402

import mvp_config as C  # noqa: E402
import plan_model as M  # noqa: E402
from sim_robot_driver import env_obstacles  # noqa: E402
from world_obstacles import min_distance  # noqa: E402

FIELDS = ['t', 'episode', 'rx', 'ry', 'rh', 'wx', 'wy', 'wh', 'px', 'py', 'cmd_v', 'cmd_w']
STUCK_CMD_V = 0.1
STUCK_SPEED = 0.05


# ---------------------------------------------------------------------------
# analysis (no ROS)
# ---------------------------------------------------------------------------
def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    out = {k: np.array([float(r[k]) for r in rows]) for k in FIELDS if k != 'episode'}
    out['episode'] = [r['episode'] for r in rows]
    return out


def episode_slices(tr):
    """{episode_id: (i0, i1, t_end or None)}: rows while '<id>' was the current label (the walk)."""
    out = {}
    labels = tr['episode']
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab and lab != 'idle' and not lab.startswith(('reset_', 'end_')):
            j = i
            while j < len(labels) and labels[j] == lab:
                j += 1
            t_end = None
            if j < len(labels) and labels[j] == f'end_{lab}':
                t_end = tr['t'][j] - tr['t'][i]
            out[lab] = (i, j, t_end)
            i = j
        else:
            i += 1
    return out


def analyse_episode(tr, i0, i1, t_end, obstacles):
    t = tr['t'][i0:i1] - tr['t'][i0]
    rxy = np.stack([tr['rx'][i0:i1], tr['ry'][i0:i1]], 1)
    rh = tr['rh'][i0:i1]
    wxy = np.stack([tr['wx'][i0:i1], tr['wy'][i0:i1]], 1)
    pxy = np.stack([tr['px'][i0:i1], tr['py'][i0:i1]], 1)
    cmd_v = tr['cmd_v'][i0:i1]
    known = np.isfinite(pxy[:, 0])
    seen = np.zeros(len(t), dtype=bool)
    if known.any():
        seen[known] = M.seen_series(pxy[known], rxy[known], rh[known], obstacles)
    # true speed
    dt = np.diff(t, prepend=t[0] - 0.1)
    dt[dt <= 0] = 0.1
    step = np.hypot(*np.diff(rxy, axis=0, prepend=rxy[:1]).T)
    speed = step / dt
    stuck_mask = (cmd_v > STUCK_CMD_V) & (speed < STUCK_SPEED)
    stuck_sec = float(np.sum(dt[stuck_mask]))
    walls = [o for o in obstacles]
    min_wall = min(min_distance(x, y, walls)[0] for x, y in rxy) if len(rxy) else math.inf
    dped = np.hypot(*(pxy[known] - rxy[known]).T) if known.any() else np.array([math.inf])
    gap = np.hypot(*(wxy - rxy).T)
    # usable share with the training filter on a 0.1 s grid
    usable = None
    if t_end is not None and t_end > 0:
        grid = np.arange(0.0, t_end + 1e-9, C.PLAN_DT)
        idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        tl = {'t_end': t_end, 'dt': C.PLAN_DT}
        n, u, share = M.usable_stats(tl, seen[idx])
        usable = (n, u, share)
    return {'t': t, 'rxy': rxy, 'rh': rh, 'wxy': wxy, 'pxy': pxy, 'cmd_v': cmd_v, 'speed': speed,
            'seen': seen, 'known': known, 'stuck_sec': stuck_sec, 'stuck_mask': stuck_mask,
            'min_wall': min_wall, 'min_ped': float(dped.min()), 'gap_med': float(np.median(gap)),
            'usable': usable, 't_end': t_end}


def draw_obstacles(ax, obstacles):
    from matplotlib.patches import Circle, Polygon
    for o in obstacles:
        c = {'wall': '#444', 'event': '#d62728', 'furniture': '#8c564b', 'column': '#7f7f7f'}.get(o.role, 'k')
        if o.kind == 'box':
            ax.add_patch(Polygon(o.corners(0.0), closed=True, color=c))
        else:
            ax.add_patch(Circle((o.x, o.y), o.r, color=c))


def plot_episode(ep, a, obstacles, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [3, 1]})
    draw_obstacles(ax, obstacles)
    r = np.array(ep['route'])
    ax.plot(r[:, 0], r[:, 1], '--', color='#1f77b4', lw=1, label='planned pedestrian route')
    k = a['known']
    ax.plot(a['pxy'][k, 0], a['pxy'][k, 1], '-', color='#1f77b4', lw=2, label='pedestrian (real)')
    ax.plot(a['rxy'][:, 0], a['rxy'][:, 1], 'k-', lw=2, label='robot TRUE (/odom)')
    if a['stuck_mask'].any():
        s = a['stuck_mask']
        ax.plot(a['rxy'][s, 0], a['rxy'][s, 1], 'rx', ms=8, label='stuck (cmd > 0.1, not moving)')
    sp = ep['robot']['start_pose']
    ax.plot(sp[0], sp[1], 'ks', mfc='none', ms=10, label='planned robot start')
    if ep.get('event'):
        ax.plot(*ep['event']['position'], '*', color='#ff7f0e', ms=14, mec='k')
    allx = np.concatenate([r[:, 0], a['rxy'][:, 0]])
    ally = np.concatenate([r[:, 1], a['rxy'][:, 1]])
    ax.set_xlim(allx.min() - 3, allx.max() + 3)
    ax.set_ylim(ally.min() - 3, ally.max() + 3)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='best')
    ax.set_title(f"{ep['episode_id']}  robot {ep['robot']['mode']}  (real trace, odom frame)", fontsize=10)
    bx.plot(a['t'], a['cmd_v'], label='commanded v (m/s)')
    bx.plot(a['t'], a['speed'], label='true speed (m/s)')
    bx.fill_between(a['t'], 0, a['seen'] * 0.2, color='g', alpha=0.3, step='post', label='seen')
    if a['t_end'] is not None:
        keep = M.kept_anchor_times(a['t_end'])
        if len(keep):
            bx.axvspan(keep[0], keep[-1], color='0.85', zorder=0, label='samples kept')
    bx.set_ylim(0, 1.6)
    bx.set_xlabel('time since walk start (s)')
    bx.legend(fontsize=8, loc='upper right')
    fig.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def animate_episode(ep, a, obstacles, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Wedge
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_obstacles(ax, obstacles)
    r = np.array(ep['route'])
    ax.plot(r[:, 0], r[:, 1], '--', color='#1f77b4', lw=1)
    ax.plot(a['rxy'][:, 0], a['rxy'][:, 1], ':', color='0.5', lw=1)
    ped, = ax.plot([], [], 'o', color='#1f77b4', ms=9)
    rob = Circle((0, 0), C.ROBOT_RADIUS_M, fc='none', ec='k', lw=2)
    ax.add_patch(rob)
    head, = ax.plot([], [], 'k-', lw=2)
    wedge = Wedge((0, 0), C.CAM_RANGE_M, 0, 1, alpha=0.18)
    ax.add_patch(wedge)
    txt = ax.text(0.01, 0.98, '', transform=ax.transAxes, va='top', fontsize=9, family='monospace')
    allx = np.concatenate([r[:, 0], a['rxy'][:, 0]])
    ally = np.concatenate([r[:, 1], a['rxy'][:, 1]])
    ax.set_xlim(allx.min() - 3, allx.max() + 3)
    ax.set_ylim(ally.min() - 3, ally.max() + 3)
    ax.set_aspect('equal')
    ax.set_title(f"{ep['episode_id']}: REAL trace (Gazebo), robot {ep['robot']['mode']}", fontsize=10)
    keep = M.kept_anchor_times(a['t_end']) if a['t_end'] else np.zeros(0)
    frames = np.arange(0, len(a['t']), 2)     # 10 Hz trace -> 5 frames per second of sim

    def upd(i):
        x, y = a['rxy'][i]
        h = a['rh'][i]
        rob.center = (x, y)
        head.set_data([x, x + 0.6 * math.cos(h)], [y, y + 0.6 * math.sin(h)])
        cx, cy = x + C.CAM_OFFSET_M * math.cos(h), y + C.CAM_OFFSET_M * math.sin(h)
        wedge.set_center((cx, cy))
        wedge.set_theta1(math.degrees(h) - math.degrees(C.CAM_HALF_FOV))
        wedge.set_theta2(math.degrees(h) + math.degrees(C.CAM_HALF_FOV))
        wedge.set_color('#2ca02c' if a['seen'][i] else '#d62728')
        if a['known'][i]:
            ped.set_data([a['pxy'][i, 0]], [a['pxy'][i, 1]])
        t = a['t'][i]
        kept = len(keep) and keep[0] - 1e-9 <= t <= keep[-1] + 1e-9
        txt.set_text(f't = {t:5.1f} s   pedestrian {"SEEN" if a["seen"][i] else "not seen"}   '
                     f'{"samples kept" if kept else "outside the kept window"}'
                     f'{"   STUCK" if a["stuck_mask"][i] else ""}')
        return ped, rob, head, wedge, txt

    FuncAnimation(fig, upd, frames=frames, blit=False).save(path, writer=PillowWriter(fps=5))
    plt.close(fig)


def report(env_dir, csv_path, gif):
    """Checks every episode in the trace; prints and writes <csv folder>/trace_report.txt;
    plots go next to the csv as <episode_id>.png (.gif)."""
    out_dir = csv_path.parent
    rep_f = open(out_dir / 'trace_report.txt', 'w')

    def print(*a):                      # noqa: A001 -- to the terminal AND the report file
        line = ' '.join(str(x) for x in a)
        sys.stdout.write(line + '\n')
        sys.stdout.flush()
        rep_f.write(line + '\n')
    env = json.loads((env_dir / 'environment.json').read_text())
    obstacles = env_obstacles(env)
    eps = {e['episode_id']: e for e in env['episodes']}
    tr = load_csv(csv_path)
    sl = episode_slices(tr)
    print(f'\n== trace {csv_path.name}: {len(tr["t"])} rows, {len(sl)} episode(s) walked')
    all_ok = True
    for eid, (i0, i1, t_end) in sl.items():
        if eid not in eps:
            print(f'  ?     {eid}: not in {env_dir.name}/environment.json (another environment?), skipped')
            continue
        ep = eps[eid]
        a = analyse_episode(tr, i0, i1, t_end, obstacles)
        probs = []
        if a['usable'] is None:
            probs.append('no end_ label seen (stopped early?)')
        elif a['usable'][2] < 0.5:
            probs.append(f'usable {a["usable"][2]:.0%} < 50%')
        if a['stuck_sec'] > 1.0:
            probs.append(f'stuck {a["stuck_sec"]:.1f} s')
        if a['min_wall'] < C.ROBOT_RADIUS_M:
            probs.append(f'robot {a["min_wall"]:.2f} m from a wall')
        if a['min_ped'] < 1.0:
            probs.append(f'robot {a["min_ped"]:.2f} m from the pedestrian')
        all_ok &= not probs
        us = f'{a["usable"][2]:4.0%} (planned {ep["planned"]["usable_share"]:.0%})' if a['usable'] else '   ?'
        print(f'  {"PASS" if not probs else "FAIL"}  {eid:18s} {ep["robot"]["mode"]:6s} usable {us}  '
              f'seen {a["seen"].mean():4.0%}  stuck {a["stuck_sec"]:.1f} s  closest wall {a["min_wall"]:.2f} m  '
              f'closest pedestrian {a["min_ped"]:.2f} m'
              + ('  <- ' + '; '.join(probs) if probs else ''))
        png = out_dir / f'{eid}.png'
        plot_episode(ep, a, obstacles, png)
        print(f'        plot: {png}')
        if gif:
            g = png.with_suffix('.gif')
            animate_episode(ep, a, obstacles, g)
            print(f'        animation: {g}')
    print('\nTRACE:', 'PASS' if all_ok and sl else 'NOT YET')
    rep_f.close()
    return all_ok


# ---------------------------------------------------------------------------
# live recording (ROS)
# ---------------------------------------------------------------------------
def run_live(env_dir, gif, out_dir):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseArray, Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from hunav_config import yaw_from_quaternion

    class Tracer(Node):
        def __init__(self):
            super().__init__('mvp_trace', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.rows = []
            self.robot = self.wheel = None
            self.ped = (math.nan, math.nan)
            self.cmd = (0.0, 0.0)
            self.episode = ''
            self.create_subscription(Odometry, '/odom', self._odom, qos_profile_sensor_data)
            self.create_subscription(Odometry, '/odom_wheel', self._wheel, qos_profile_sensor_data)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped, 10)
            self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
            self.create_subscription(String, '/dataset/episode', self._ep, 10)
            self.create_timer(0.1, self._tick)
            self.get_logger().info('tracing; Ctrl+C when the runner is done')

        @staticmethod
        def _pose(msg):
            p = msg.pose.pose
            return p.position.x, p.position.y, yaw_from_quaternion(p.orientation)

        def _odom(self, msg):
            self.robot = self._pose(msg)

        def _wheel(self, msg):
            self.wheel = self._pose(msg)

        def _ped(self, msg):
            if msg.poses:
                self.ped = (msg.poses[0].position.x, msg.poses[0].position.y)

        def _cmd(self, msg):
            self.cmd = (msg.linear.x, msg.angular.z)

        def _ep(self, msg):
            self.episode = msg.data

        def _tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now == 0 or self.robot is None:
                return
            w = self.wheel or (math.nan, math.nan, math.nan)
            self.rows.append([now, self.episode, *self.robot, *w, *self.ped, *self.cmd])
            if len(self.rows) % 100 == 1:
                print(f'  sim {now:7.1f} s  {len(self.rows)} rows  episode label: {self.episode or "(none yet)"}',
                      flush=True)

    print(f'trace_episode.py: tracing {env_dir} -- waiting for /odom (is the launch running?)', flush=True)
    rclpy.init()
    node = Tracer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    rows = node.rows
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'trace.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f'\nsaved {len(rows)} rows: {path}')
    if rows:
        report(env_dir, path, gif)


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith('-'):
        sys.exit(__doc__)
    env_dir = Path(args[0])
    gif = '--gif' in args
    if '--replay' in args:
        report(env_dir, Path(args[args.index('--replay') + 1]), gif)
    else:
        out = Path(args[args.index('--out') + 1]) if '--out' in args else \
            env_dir / 'traces' / time.strftime('%Y%m%d_%H%M%S')
        run_live(env_dir, gif, out)


if __name__ == '__main__':
    main()