#!/usr/bin/env python3
"""
check_environments.py -- offline check of the planned environments (P2, 2026-11-01).

Run it on the output of environment_generator.py BEFORE anything goes to Gazebo.
It does not trust the generator: the obstacles are read from each scene.yaml (the
very file spawn_obstacles.py will load), and everything is recomputed from them.

Per environment it checks
  scene      scene.yaml loads; only box/cylinder; heights the obstacle reader keeps
             (top >= 0.10 m, bottom <= 1.80 m); same obstacles as environment.json
  routes     every episode re-planned with the bridge's planner on scene.yaml = the stored
             route (within 5 cm); planning time below PLANNER_SLOW_SEC (the bridge plans
             inside its 10 Hz tick); arc paths (E) keep >= 0.6 m from every obstacle
  start/goal free (>= 0.35 m from obstacles) and inside the environment
  timing     walking time before the event >= 12 s and after it >= 8 s (C: standing >= 7.5 s)
  event      A and D: the route leaves the straight line by >= 0.4 m (there IS a manoeuvre)
  robot      start pose free (>= 0.5 m from obstacles), parked robots >= 1.2 m from the route
             and >= 1.5 m from start/goal; follow start free behind the pedestrian
  usable     the planned share of usable samples (camera model) >= USABLE_TARGET
and prints one line per environment, a summary (mode mix, planned samples, wall-clock
budget) and writes overview.png (all environments of the folder, top view).

USAGE
    python3 check_scripts/check_environments.py datasets/dev/A_detour
    python3 check_scripts/check_environments.py datasets/dev/*          # every scenario
Exit code 0 = all PASS.
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MVP = HERE.parent
sys.path.insert(0, str(MVP))
sys.path.insert(0, str(MVP.parent / 'agent_control'))
sys.path.insert(0, str(MVP.parent / 'hunav_codes'))

import numpy as np  # noqa: E402

import mvp_config as C  # noqa: E402
import plan_model as M  # noqa: E402
from hunav_config import odom_to_world, world_to_odom  # noqa: E402
from world_obstacles import CLEARANCE_M, MIN_CLEARANCE_M, Obstacle, load_obstacles, min_distance, plan_human_route  # noqa: E402

READER_MIN_TOP, READER_MAX_BOTTOM = 0.10, 1.80     # gazebo_scene.py MIN_TOP_Z / MAX_BOTTOM_Z


def o2w(p):
    return odom_to_world(p[0], p[1], 0.0)[:2]


def w2o(p):
    return world_to_odom(p[0], p[1], 0.0)[:2]


def odom_obstacles(world_obs):
    out = []
    for o in world_obs:
        x, y = w2o((o.x, o.y))
        if o.kind == 'box':
            yaw = math.atan2(o.s, o.c) - (odom_to_world(0, 0, 0.0)[2])
            out.append(Obstacle(o.name, 'box', x, y, yaw, 2 * o.hx, 2 * o.hy))
        else:
            out.append(Obstacle(o.name, 'cylinder', x, y, radius=o.r))
    return out


def inside(rects, p, margin=0.0):
    return any(x0 + margin <= p[0] <= x1 - margin and y0 + margin <= p[1] <= y1 - margin for x0, y0, x1, y1 in rects)


def check_env(d):
    """Returns (problems, info)."""
    import yaml
    probs, info = [], {'plan_times': [], 'episodes': []}
    env = json.loads((d / 'environment.json').read_text())
    raw = yaml.safe_load((d / 'scene.yaml').read_text())
    wobs = load_obstacles(d / 'scene.yaml')
    info['env'] = env
    # ---- scene -----------------------------------------------------------------
    if len(wobs) != len(env['obstacles']):
        probs.append(f'scene.yaml has {len(wobs)} obstacles, environment.json {len(env["obstacles"])}')
    for e in raw['obstacles']:
        h = float(e.get('height', 1.0))
        if e['type'] not in ('box', 'cylinder'):
            probs.append(f'{e["name"]}: type {e["type"]} is not read by the bridge')
        if h < READER_MIN_TOP:            # spawned standing on the floor: bottom 0 <= READER_MAX_BOTTOM
            probs.append(f'{e["name"]}: height {h} m is ignored by the obstacle reader')
        if h < 1.0 - 1e-6:
            probs.append(f'{e["name"]}: lower than 1 m (visibility of low objects is not verified)')
    oobs = odom_obstacles(wobs)
    by_name = {o['name']: o for o in env['obstacles']}
    for o in oobs:
        j = by_name.get(o.name)
        if j is None or math.hypot(o.x - j['x'], o.y - j['y']) > 0.01:
            probs.append(f'{o.name}: scene.yaml and environment.json disagree')
    rects = env['rects']
    # ---- episodes ----------------------------------------------------------------
    for ep in env['episodes']:
        eid = ep['episode_id'].split('_')[-1]
        start, goal, v = tuple(ep['start']), tuple(ep['goal']), ep['speed']
        for name, p in (('start', start), ('goal', goal)):
            if min_distance(*p, oobs)[0] < MIN_CLEARANCE_M:
                probs.append(f'{eid}: {name} {p} closer than {MIN_CLEARANCE_M} m to an obstacle')
            if not inside(rects, p, 0.2):
                probs.append(f'{eid}: {name} {p} is outside the environment')
        if ep['path_mode']:
            pl = M.Polyline(ep['route'])
            dmin = min(min_distance(*q, oobs)[0] for q in pl.at(np.arange(0, pl.length, 0.1)))
            if dmin < CLEARANCE_M - 1e-6:
                probs.append(f'{eid}: arc path comes {dmin:.2f} m close to an obstacle')
        else:
            t0 = time.time()
            r, _ = plan_human_route(o2w(start), o2w(goal), wobs)
            dt = time.time() - t0
            info['plan_times'].append(dt)
            if dt > C.PLANNER_SLOW_SEC:
                probs.append(f'{eid}: route planning took {dt:.2f} s (bridge tick would stall)')
            if r is None:
                probs.append(f'{eid}: NO ROUTE with the bridge planner')
                continue
            route = [start] + [w2o(p) for p in r]
            if len(route) != len(ep['route']) or max(math.dist(a, b) for a, b in zip(route, ep['route'])) > 0.05:
                probs.append(f'{eid}: the bridge would plan a different route than planned')
            pl = M.Polyline(route)
        # timing and event
        ev = ep['event']
        if ev is not None:
            s_ev = ev['s']
            if ev['type'] == 'stop':
                tl = M.ped_timeline(pl, v, ep['stop'])
                t_before = tl['t_end'] - ep['stop']['stand_sec']
                t_after = ep['stop']['stand_sec']
                if t_after < C.STAND_SEC - 1e-6:
                    probs.append(f'{eid}: standing {t_after:.1f} s < {C.STAND_SEC} s')
            else:
                t_before = s_ev / v
                t_after = (pl.length - C.END_TRIGGER_M - s_ev) / v
                if t_after < C.AFTER_EVENT_SEC - 0.05:
                    probs.append(f'{eid}: only {t_after:.1f} s of walking after the event (< {C.AFTER_EVENT_SEC})')
            if t_before < C.BEFORE_EVENT_SEC - 0.05:
                probs.append(f'{eid}: only {t_before:.1f} s of walking before the event (< {C.BEFORE_EVENT_SEC})')
            if ev['type'] in ('detour', 'door'):
                a, b = np.array(start), np.array(goal)
                n = np.array([-(b - a)[1], (b - a)[0]]) / np.hypot(*(b - a))
                dev = max(abs((np.array(p) - a) @ n) for p in pl.p)
                if dev < C.DETOUR_MIN_M - 0.01:
                    probs.append(f'{eid}: route leaves the straight line by only {dev:.2f} m')
        # robot
        rb = ep['robot']
        rp = rb['start_pose']
        if min_distance(rp[0], rp[1], oobs)[0] < C.ROBOT_WALL_CLEAR_M - 0.01:
            probs.append(f'{eid}: robot start {rp[:2]} too close to an obstacle')
        if not inside(rects, rp, 0.0):
            probs.append(f'{eid}: robot start outside the environment')
        if rb['mode'] in ('still', 'watch'):
            if pl.distance_to(rp[:2]) < C.ROBOT_ROUTE_CLEAR_M - 0.01:
                probs.append(f'{eid}: parked robot only {pl.distance_to(rp[:2]):.2f} m from the route')
            if min(math.dist(rp[:2], start), math.dist(rp[:2], goal)) < C.ROBOT_START_CLEAR_M - 0.01:
                probs.append(f'{eid}: parked robot too close to the start or goal')
            robot = {'mode': rb['mode'], 'pose': tuple(rp)}
        else:
            robot = {'mode': 'follow', 'dist': rb['dist']}
        tl = M.ped_timeline(pl, v, ep['stop'])
        rxy, _, _, st = M.evaluate(tl, pl, robot, oobs)
        if rb['mode'] == 'follow':
            cl = min(min_distance(*q, oobs)[0] for q in rxy[::3])
            if cl < C.ROBOT_PATH_CLEAR_M - 0.01:
                probs.append(f'{eid}: the following robot passes {cl:.2f} m from a wall')
        if st[2] < C.USABLE_TARGET - 1e-6:
            probs.append(f'{eid}: planned usable share {st[2]:.0%} < {C.USABLE_TARGET:.0%}')
        info['episodes'].append({'mode': rb['mode'], 'share': st[2], 'kept': st[0], 'usable': st[1],
                                 't_end': tl['t_end']})
    return probs, info


def overview(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    n = len(results)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
    for ax in axs.flat:
        ax.axis('off')
    for ax, (d, probs, info) in zip(axs.flat, results):
        env = info.get('env')
        if env is None:
            continue
        for o in env['obstacles']:
            c = {'wall': '#444', 'event': '#d62728', 'furniture': '#8c564b', 'column': '#7f7f7f'}.get(o['role'], 'k')
            if o['type'] == 'box':
                ob = Obstacle('x', 'box', o['x'], o['y'], o['yaw'], o['size_x'], o['size_y'])
                ax.add_patch(Polygon(ob.corners(0.0), closed=True, color=c))
            else:
                ax.add_patch(Circle((o['x'], o['y']), o['radius'], color=c))
        for ep in env['episodes']:
            r = np.array(ep['route'])
            ax.plot(r[:, 0], r[:, 1], lw=0.8)
            rp = ep['robot']['start_pose']
            ax.plot(rp[0], rp[1], 'o', mfc='none', ms=4)
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(f'{d.name}  {"PASS" if not probs else "FAIL"}', fontsize=8,
                     color='green' if not probs else 'red')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        sys.exit(__doc__)
    all_ok = True
    for sd in dirs:
        envs = sorted(p for p in sd.iterdir() if (p / 'environment.json').exists())
        if not envs:
            print(f'{sd}: no environments')
            continue
        print(f'== {sd}  ({len(envs)} environments)')
        results = []
        for d in envs:
            probs, info = check_env(d)
            results.append((d, probs, info))
            sh = [e['share'] for e in info['episodes']]
            pt = max(info['plan_times']) if info['plan_times'] else 0.0
            print(f'  {"PASS" if not probs else "FAIL"}  {d.name}: {len(info["episodes"])} episodes, planned usable '
                  f'{min(sh):.0%}-{max(sh):.0%}, slowest route plan {pt * 1000:.0f} ms')
            for p in probs:
                print(f'        - {p}')
            all_ok &= not probs
        eps = [e for _, _, i in results for e in i['episodes']]
        modes = Counter(e['mode'] for e in eps)
        kept = sum(e['kept'] for e in eps)
        usable = sum(e['usable'] for e in eps)
        sim = sum(e['t_end'] + C.RESET_SIM_SEC for e in eps)
        wall_h = (sim / C.RTF + len(envs) * C.ENV_OVERHEAD_WALL_SEC) / 3600
        print(f'  summary: {len(eps)} episodes, robot modes ' + ', '.join(f'{m} {n}' for m, n in sorted(modes.items()))
              + f'; planned samples {kept} kept / {usable} usable ({usable / max(kept, 1):.0%}); '
              f'estimated wall clock {wall_h:.1f} h (RTF {C.RTF})')
        overview(results, sd / 'overview.png')
        print(f'  overview: {sd / "overview.png"}')
    print('\nOVERALL:', 'PASS' if all_ok else 'NOT YET')
    sys.exit(0 if all_ok else 2)


if __name__ == '__main__':
    main()