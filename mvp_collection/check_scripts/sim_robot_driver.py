#!/usr/bin/env python3
"""
sim_robot_driver.py -- offline test of P4 (2026-11-01): runs the REAL control law of
robot_driver.py (RobotController) on a simple robot model (unicycle, cmd_vel applied
exactly) against every planned episode, and checks:

  usable     the usable share the driver ACTUALLY gets (camera model of plan_model.py on the
             simulated robot track) is >= 0.5 and close to the planned share
  collision  the robot never comes closer than ROBOT_RADIUS_M to a wall or obstacle
  personal   the robot never comes closer than 1.0 m to the pedestrian

It does not test Gazebo (wheel slip, the DiffDrive limits, HuNav reacting to the robot):
that is P5. It tests that the driver does what the plan assumed.

USAGE
    python3 check_scripts/sim_robot_driver.py datasets/dev/A_detour            # all episodes of a scenario
    python3 check_scripts/sim_robot_driver.py datasets/dev/*
    python3 check_scripts/sim_robot_driver.py datasets/dev/B_corner --animate env_B_001 3
        -> datasets/dev/B_corner/env_B_001/anim_ep3.gif : pedestrian, robot, camera wedge (green = seen)
    python3 check_scripts/sim_robot_driver.py datasets/dev/B_corner --animate env_B_001 all
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MVP = HERE.parent
sys.path.insert(0, str(MVP))
sys.path.insert(0, str(MVP.parent / 'agent_control'))
sys.path.insert(0, str(MVP.parent / 'hunav_codes'))

import numpy as np  # noqa: E402

import mvp_config as C  # noqa: E402
import plan_model as M  # noqa: E402
from robot_driver import RobotController  # noqa: E402
from world_obstacles import Obstacle, min_distance  # noqa: E402

SIM_DT = 0.05


def env_obstacles(env):
    out = []
    for o in env['obstacles']:
        if o['type'] == 'box':
            out.append(Obstacle(o['name'], 'box', o['x'], o['y'], o['yaw'], o['size_x'], o['size_y']))
        else:
            out.append(Obstacle(o['name'], 'cylinder', o['x'], o['y'], radius=o['radius']))
        out[-1].role = o['role']
    return out


def robot_command(ep):
    rb = ep['robot']
    if rb['mode'] == 'still':
        return {'mode': 'still', 'heading': rb['start_pose'][2]}
    if rb['mode'] == 'watch':
        return {'mode': 'watch'}
    return {'mode': 'follow', 'route': ep['route'], 'dist': rb['dist']}


def simulate(ep, obstacles):
    """Returns dict: t, ped_xy, robot_xy, robot_h (on the PLAN_DT grid), seen, stats."""
    pl = M.Polyline(ep['route'])
    tl = M.ped_timeline(pl, ep['speed'], ep['stop'])
    ctrl = RobotController()
    ctrl.set_command(robot_command(ep))
    x, y, h = ep['robot']['start_pose']
    t_grid = tl['t']
    rxy, rh = np.zeros((len(t_grid), 2)), np.zeros(len(t_grid))
    min_wall, min_ped = math.inf, math.inf
    t, k = 0.0, 0
    while k < len(t_grid):
        if t >= t_grid[k] - 1e-9:
            rxy[k], rh[k] = (x, y), h
            k += 1
        px = np.interp(t, t_grid, tl['xy'][:, 0])
        py = np.interp(t, t_grid, tl['xy'][:, 1])
        v, w = ctrl.step((x, y, h), (px, py))
        x += v * math.cos(h) * SIM_DT
        y += v * math.sin(h) * SIM_DT
        h = (h + w * SIM_DT + math.pi) % (2 * math.pi) - math.pi
        min_wall = min(min_wall, min_distance(x, y, obstacles)[0])
        min_ped = min(min_ped, math.hypot(px - x, py - y))
        t += SIM_DT
    seen = M.seen_series(tl['xy'], rxy, rh, obstacles)
    return {'tl': tl, 'robot_xy': rxy, 'robot_h': rh, 'seen': seen, 'stats': M.usable_stats(tl, seen),
            'min_wall': min_wall, 'min_ped': min_ped}


def animate(env, ep, obstacles, sim, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Polygon, Wedge

    tl, rxy, rh, seen = sim['tl'], sim['robot_xy'], sim['robot_h'], sim['seen']
    keep = M.kept_anchor_times(tl['t_end'])
    fig, ax = plt.subplots(figsize=(10, 6))
    for o in obstacles:
        c = {'wall': '#444', 'event': '#d62728', 'furniture': '#8c564b', 'column': '#7f7f7f'}.get(o.role, 'k')
        if o.kind == 'box':
            ax.add_patch(Polygon(o.corners(0.0), closed=True, color=c))
        else:
            ax.add_patch(Circle((o.x, o.y), o.r, color=c))
    r = np.array(ep['route'])
    ax.plot(r[:, 0], r[:, 1], '--', color='#1f77b4', lw=1)
    if ep['event']:
        ax.plot(*ep['event']['position'], '*', color='#ff7f0e', ms=14, mec='k')
    ax.plot(rxy[:, 0], rxy[:, 1], ':', color='0.5', lw=1)
    ped, = ax.plot([], [], 'o', color='#1f77b4', ms=9)
    rob = Circle((0, 0), C.ROBOT_RADIUS_M, fc='none', ec='k', lw=2)
    ax.add_patch(rob)
    head, = ax.plot([], [], 'k-', lw=2)
    wedge = Wedge((0, 0), C.CAM_RANGE_M, 0, 1, alpha=0.18)
    ax.add_patch(wedge)
    txt = ax.text(0.01, 0.98, '', transform=ax.transAxes, va='top', fontsize=9, family='monospace')
    x0, x1 = min(r[:, 0].min(), rxy[:, 0].min()) - 3, max(r[:, 0].max(), rxy[:, 0].max()) + 3
    y0, y1 = min(r[:, 1].min(), rxy[:, 1].min()) - 3, max(r[:, 1].max(), rxy[:, 1].max()) + 3
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_title(f"{ep['episode_id']}: {ep['speed']:.2f} m/s, robot {ep['robot']['mode']}  "
                 f"(simulated driver, odom frame)", fontsize=10)
    step = max(1, int(round(0.25 / tl['dt'])))
    frames = range(0, len(tl['t']), step)

    def upd(i):
        t = tl['t'][i]
        ped.set_data([tl['xy'][i, 0]], [tl['xy'][i, 1]])
        rob.center = tuple(rxy[i])
        hx, hy = rxy[i] + 0.6 * np.array([math.cos(rh[i]), math.sin(rh[i])])
        head.set_data([rxy[i, 0], hx], [rxy[i, 1], hy])
        cam = rxy[i] + C.CAM_OFFSET_M * np.array([math.cos(rh[i]), math.sin(rh[i])])
        wedge.set_center(tuple(cam))
        wedge.set_theta1(math.degrees(rh[i] - C.CAM_HALF_FOV))
        wedge.set_theta2(math.degrees(rh[i] + C.CAM_HALF_FOV))
        wedge.set_color('#2ca02c' if seen[i] else '#d62728')
        kept = len(keep) and keep[0] - 1e-9 <= t <= keep[-1] + 1e-9
        txt.set_text(f't = {t:5.1f} s   pedestrian {"SEEN" if seen[i] else "not seen"}   '
                     f'{"samples kept" if kept else "outside the kept window"}')
        return ped, rob, head, wedge, txt

    anim = FuncAnimation(fig, upd, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    anim_env = anim_ep = None
    if '--animate' in args:
        i = args.index('--animate')
        anim_env, anim_ep = args[i + 1], args[i + 2]
        args = args[:i] + args[i + 3:]
    all_ok = True
    for sd in map(Path, args):
        envs = sorted(p for p in sd.iterdir() if (p / 'environment.json').exists())
        if anim_env:
            envs = [p for p in envs if p.name == anim_env]
        if not envs:
            continue
        print(f'== {sd}')
        diffs = []
        for d in envs:
            env = json.loads((d / 'environment.json').read_text())
            obs = env_obstacles(env)
            for ep in env['episodes']:
                if anim_env and anim_ep != 'all' and ep['index'] != int(anim_ep):
                    continue
                sim = simulate(ep, obs)
                n, u, share = sim['stats']
                planned = ep['planned']['usable_share']
                diffs.append(share - planned)
                probs = []
                if share < 0.5:
                    probs.append(f'usable {share:.0%} < 50%')
                if sim['min_wall'] < C.ROBOT_RADIUS_M:
                    probs.append(f'robot {sim["min_wall"]:.2f} m from a wall (collision)')
                if sim['min_ped'] < 1.0:
                    probs.append(f'robot {sim["min_ped"]:.2f} m from the pedestrian')
                all_ok &= not probs
                print(f'  {"PASS" if not probs else "FAIL"}  {ep["episode_id"]:18s} {ep["robot"]["mode"]:6s} '
                      f'planned {planned:4.0%}  driver {share:4.0%}  closest wall {sim["min_wall"]:.2f} m  '
                      f'closest pedestrian {sim["min_ped"]:.2f} m' + ('  <- ' + '; '.join(probs) if probs else ''))
                if anim_env:
                    out = d / f'anim_ep{ep["index"]}.gif'
                    animate(env, ep, obs, sim, out)
                    print(f'        animation: {out}')
        if diffs:
            print(f'  driver minus planned usable share: median {np.median(diffs):+.0%}, worst {min(diffs):+.0%}')
    print('\nOVERALL:', 'PASS' if all_ok else 'NOT YET')
    sys.exit(0 if all_ok else 2)


if __name__ == '__main__':
    main()