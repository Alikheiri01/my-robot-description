#!/usr/bin/env python3
"""
environment_generator.py -- P2 of the MVP collection (2026-11-01).

Turns an environment TEMPLATE plus random sizes into a concrete ENVIRONMENT and
plans all its EPISODES offline, before Gazebo is involved:

  * the environment: walls (boxes, 0.2 m thick, 2 m high), the event obstacle (A),
    a door (D), columns (E hall), 0-3 pieces of furniture against the walls, and
    wall niches ("bays") where a parked robot can stand;
  * every episode: pedestrian start, goal and speed, the route the bridge will plan
    (the SAME planner, world_obstacles.plan_human_route, on the same obstacles in the
    world frame), the event (where and when), and the robot behaviour (still / watch /
    follow) with a robot pose chosen so that the planned share of USABLE samples is
    at least USABLE_TARGET (plan_model.py: camera wedge, 8 m range, wall occlusion).

Output (2026-11-02): one DATASET folder per generated set, everything of that set inside it
(environments, their recorded runs, the session logs). Default dataset: 'dev' (tests).
    datasets/<dataset>/<scenario>/<env_id>/scene.yaml        world frame, spawn_obstacles.py format
    datasets/<dataset>/<scenario>/<env_id>/environment.json  everything else (odom frame)
    datasets/<dataset>/<scenario>/<env_id>/preview.png       top view: routes, events, robot poses
    datasets/<dataset>/<scenario>/summary.csv                one line per episode
  later, written by episode_runner.py:
    datasets/<dataset>/<scenario>/<env_id>/runs/<run>/       a recording (samples, reports, trace)
    datasets/<dataset>/<scenario>/<env_id>/dry_runs/<time>/  a run without recording
    datasets/<dataset>/logs/                                 session logs of the runner
  datasets/ is in .gitignore; a dataset is rebuilt exactly from its command (same --seed).

USAGE (no ROS, no simulation needed)
    cd ~/ros2_ws/src/my_robot_description/mvp_collection
    python3 environment_generator.py --all --envs 4 --seed 1                  # -> datasets/dev
    python3 environment_generator.py --all --envs 20 --seed 1 --dataset mvp_v1   # the real set
    python3 environment_generator.py --scenario B --envs 1 --seed 7 --dataset try_b
Then check them:  python3 check_scripts/check_environments.py datasets/dev/*

Templates (local frame: the pedestrian walks roughly along +x; a random mirror flips y):
    T1 straight hallway   (A with the event obstacle, E straight, C ends in a dead end)
    T2 L-corner hallway   (B)
    T3 two rooms + door   (D)
    T4 hall with columns  (E arcs)
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'agent_control'))
sys.path.insert(0, str(HERE.parent / 'hunav_codes'))

import numpy as np  # noqa: E402

import mvp_config as C  # noqa: E402
import plan_model as M  # noqa: E402
from hunav_config import SPAWN_YAW, odom_to_world, world_to_odom  # noqa: E402
from world_obstacles import CLEARANCE_M, Obstacle, min_distance, plan_human_route  # noqa: E402

DATASETS_DIR = HERE / 'datasets'
GENERATOR_VERSION = '2026-11-01'


def U(rng, lo_hi):
    return rng.uniform(*lo_hi)


def w2o(x, y):
    ox, oy, _ = world_to_odom(x, y, 0.0)
    return ox, oy


def o2w(x, y):
    wx, wy, _ = odom_to_world(x, y, 0.0)
    return wx, wy


# ===========================================================================
# Environment
# ===========================================================================
class Env:
    """Everything in the ODOM frame (= RViz). Templates write in local coordinates;
    self.m (+1/-1) mirrors y."""

    def __init__(self, env_id, scenario, template, rng, mirror):
        self.env_id, self.scenario, self.template, self.rng = env_id, scenario, template, rng
        self.m = -1 if mirror else 1
        self.obstacles = []          # Obstacle (odom), with .height and .role
        self.rects = []              # walkable interior rectangles (x0, y0, x1, y1), odom
        self.faces = []              # wall faces for furniture: (x0, y0, x1, y1, nx, ny), odom
        self.params = {'mirror': bool(mirror)}
        self.episodes = []
        self._n = 0
        self._wcache = None

    # ---- primitives (local coordinates in, odom out) -------------------------
    def P(self, x, y):
        return (float(x), float(self.m * y))

    def add_box(self, role, cx, cy, sx, sy, h, yaw=0.0):
        x, y = self.P(cx, cy)
        o = Obstacle(f'{role}_{self._n:02d}', 'box', x, y, self.m * yaw, sx, sy)
        o.height, o.role = h, role
        self._n += 1
        self.obstacles.append(o)
        self._wcache = None
        return o

    def add_cyl(self, role, cx, cy, r, h):
        x, y = self.P(cx, cy)
        o = Obstacle(f'{role}_{self._n:02d}', 'cylinder', x, y, radius=r)
        o.height, o.role = h, role
        self._n += 1
        self.obstacles.append(o)
        self._wcache = None
        return o

    def remove(self, o):
        self.obstacles.remove(o)
        self._wcache = None

    def add_rect(self, x0, y0, x1, y1):
        a, b = self.P(x0, y0), self.P(x1, y1)
        self.rects.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))

    def hwall(self, xa, xb, y_face, out, ext_a=True, ext_b=True, face=True):
        """Wall along x from xa to xb; its face (towards the walking area) at y_face, the wall body
        on the side `out` (+1: +y). Ends extended by the wall thickness to close corners."""
        T = C.WALL_T
        a, b = xa - (T if ext_a else 0.0), xb + (T if ext_b else 0.0)
        self.add_box('wall', (a + b) / 2, y_face + out * T / 2, b - a, T, C.WALL_H)
        if face and xb - xa > 0.3:
            p0, p1 = self.P(xa, y_face), self.P(xb, y_face)
            self.faces.append((*p0, *p1, 0.0, float(-out * self.m)))

    def vwall(self, ya, yb, x_face, out, ext_a=True, ext_b=True, face=True):
        T = C.WALL_T
        lo, hi = min(ya, yb), max(ya, yb)
        ext_lo, ext_hi = (ext_a, ext_b) if ya <= yb else (ext_b, ext_a)
        a, b = lo - (T if ext_lo else 0.0), hi + (T if ext_hi else 0.0)
        self.add_box('wall', x_face + out * T / 2, (a + b) / 2, T, b - a, C.WALL_H)
        if face and hi - lo > 0.3:
            p0, p1 = self.P(x_face, lo), self.P(x_face, hi)
            self.faces.append((*p0, *p1, float(-out), 0.0))

    def hwall_gaps(self, xa, xb, y_face, out, gaps, ext_a=True, ext_b=True):
        """hwall with openings [(g0, g1)] (bays): no extension into an opening."""
        cur, first = xa, True
        for g0, g1 in sorted(gaps):
            if g0 > cur + 1e-6:
                self.hwall(cur, g0, y_face, out, ext_a=(ext_a if first else False), ext_b=False)
            cur, first = g1, False
        if xb > cur + 1e-6:
            self.hwall(cur, xb, y_face, out, ext_a=(ext_a if first else False), ext_b=ext_b)

    def vwall_gaps(self, ya, yb, x_face, out, gaps, ext_a=True, ext_b=True):
        cur, first = ya, True
        for g0, g1 in sorted(gaps):
            if g0 > cur + 1e-6:
                self.vwall(cur, g0, x_face, out, ext_a=(ext_a if first else False), ext_b=False)
            cur, first = g1, False
        if yb > cur + 1e-6:
            self.vwall(cur, yb, x_face, out, ext_a=(ext_a if first else False), ext_b=ext_b)

    def bay_h(self, bx, y_face, out):
        """Niche of BAY_W x BAY_D behind an opening [bx, bx+BAY_W] of a wall along x."""
        W, D = C.BAY_W, C.BAY_D
        self.vwall(y_face, y_face + out * D, bx, -1, ext_a=False, ext_b=True, face=False)
        self.vwall(y_face, y_face + out * D, bx + W, +1, ext_a=False, ext_b=True, face=False)
        self.hwall(bx, bx + W, y_face + out * D, out, face=False)
        self.add_rect(bx, y_face, bx + W, y_face + out * D)

    def bay_v(self, by, x_face, out):
        W, D = C.BAY_W, C.BAY_D
        self.hwall(x_face, x_face + out * D, by, -1, ext_a=False, ext_b=True, face=False)
        self.hwall(x_face, x_face + out * D, by + W, +1, ext_a=False, ext_b=True, face=False)
        self.vwall(by, by + W, x_face + out * D, out, face=False)
        self.add_rect(x_face, by, x_face + out * D, by + W)

    # ---- frames ------------------------------------------------------------------
    def world_obstacles(self):
        """The obstacles exactly as spawn_obstacles.py / the bridge will have them (world frame)."""
        if self._wcache is None:
            out = []
            for o in self.obstacles:
                wx, wy = o2w(o.x, o.y)
                if o.kind == 'box':
                    w = Obstacle(o.name, 'box', wx, wy, o.yaw + SPAWN_YAW, 2 * o.hx, 2 * o.hy)
                else:
                    w = Obstacle(o.name, 'cylinder', wx, wy, radius=o.r)
                w.height, w.role = o.height, o.role
                out.append(w)
            self._wcache = out
        return self._wcache

    def plan(self, start, goal):
        """The bridge's route (planned in the WORLD frame, like the bridge), returned in odom,
        INCLUDING the start point. None if there is no route."""
        r, _ = plan_human_route(o2w(*start), o2w(*goal), self.world_obstacles())
        if r is None:
            return None
        return [tuple(start)] + [w2o(*p) for p in r]

    def clear_at(self, x, y):
        return min_distance(x, y, self.obstacles)[0]

    def translate(self, dx, dy):
        """Shift everything (odom frame) by (dx, dy)."""
        for o in self.obstacles:
            o.x, o.y = o.x + dx, o.y + dy
        self._wcache = None
        self.rects = [(a + dx, b + dy, c + dx, d + dy) for a, b, c, d in self.rects]
        self.faces = [(a + dx, b + dy, c + dx, d + dy, nx, ny) for a, b, c, d, nx, ny in self.faces]

        def mv(p):
            return [round(p[0] + dx, 3), round(p[1] + dy, 3)] + list(p[2:])
        for ep in self.episodes:
            ep['start'], ep['goal'] = mv(ep['start']), mv(ep['goal'])
            ep['route'] = [mv(p) for p in ep['route']]
            if ep['event']:
                ep['event']['position'] = mv(ep['event']['position'])
            ep['robot']['start_pose'] = mv(ep['robot']['start_pose'])
        self.params['offset'] = [round(dx, 3), round(dy, 3)]


# ===========================================================================
# Templates (geometry only)
# ===========================================================================
def place_bays(rng, x_lo, x_hi, n, avoid=(), side=None):
    """n bay openings along [x_lo, x_hi] (each BAY_W long), not overlapping, not in `avoid` intervals.
    side: +1 / -1 to put all on one wall, None = random wall per bay."""
    out = []
    fixed = side
    for _ in range(60):
        if len(out) >= n:
            break
        bx = rng.uniform(x_lo, x_hi - C.BAY_W)
        side = fixed if fixed is not None else rng.choice((-1, 1))
        if any(abs(bx - b) < C.BAY_W + 0.8 and s == side for b, s in out):
            continue
        if any(bx < a1 and bx + C.BAY_W > a0 for a0, a1 in avoid):
            continue
        out.append((bx, side))
    return out


def build_hallway(env, x0, x1, w, bays, close_end=True):
    """Hallway x in [x0, x1], y in [-w/2, w/2], closed at x0 (and at x1 if close_end)."""
    hw = w / 2
    for side in (+1, -1):
        gaps = [(bx, bx + C.BAY_W) for bx, s in bays if s == side]
        env.hwall_gaps(x0, x1, side * hw, side, gaps)
        for g0, _ in gaps:
            env.bay_h(g0, side * hw, side)
    env.vwall(-hw, hw, x0, -1)
    if close_end:
        env.vwall(-hw, hw, x1, +1)
    env.add_rect(x0, -hw, x1, hw)


def template_T1(env, rng, length_needed, event_x=None):
    w = U(rng, C.HALL_WIDTH_RANGE)
    x0, x1 = 0.0, length_needed
    avoid = [(event_x - 1.0, event_x + 1.0)] if event_x is not None else []
    bays = place_bays(rng, x0 + 0.5, x1 - 0.5, rng.randint(*C.BAYS_PER_HALL), avoid)
    build_hallway(env, x0, x1, w, bays)
    env.params.update({'hall_width': round(w, 3), 'length': round(x1 - x0, 3),
                       'bays': [[round(b, 2), s] for b, s in bays]})
    return w


# ===========================================================================
# Episodes per scenario
# ===========================================================================
def speeds_for(rng):
    return [round(U(rng, C.SPEED_RANGE), 2) for _ in range(C.EPISODES_PER_ENV)]


def detour_m(route):
    """Largest distance of the route from the straight start->goal line."""
    a, b = np.array(route[0]), np.array(route[-1])
    ab = b - a
    L = np.hypot(*ab)
    if L < 1e-9:
        return 0.0
    n = np.array([-ab[1], ab[0]]) / L
    return float(max(abs((np.array(p) - a) @ n) for p in route))


def timing(route, speed, s_event, stop=None):
    """(t_before, t_after) in walking seconds; for C t_after is the standing time."""
    pl = M.Polyline(route)
    if stop is not None:
        return pl.length / speed, stop['stand_sec']
    return s_event / speed, (pl.length - C.END_TRIGGER_M - s_event) / speed


def new_episode(env, k, start, goal, speed, route, event, path=None, stop=None):
    pl = M.Polyline(route)
    if event is not None:
        s_ev = pl.project(event['position'])[0] if 's' not in event else event['s']
        event['s'] = round(float(s_ev), 3)
        event['position'] = [round(float(v), 3) for v in pl.at([s_ev])[0]]
    tb, ta = timing(route, speed, event['s'] if event else 0.0, stop)
    return {'episode_id': f'{env.env_id}_ep{k}', 'index': k, 'speed': speed,
            'start': [round(start[0], 3), round(start[1], 3)], 'goal': [round(goal[0], 3), round(goal[1], 3)],
            'route': [[round(x, 3), round(y, 3)] for x, y in route],
            'path_mode': path is not None, 'event': event, 'stop': stop,
            't_before_sec': round(tb, 2), 't_after_sec': round(ta, 2)}


def timing_ok(ep):
    if ep['event'] is None:          # E: total length is what matters
        return True
    return ep['t_before_sec'] >= C.BEFORE_EVENT_SEC - 1e-6 and ep['t_after_sec'] >= C.AFTER_EVENT_SEC - 1e-6


def episodes_A(env, rng):
    speeds = speeds_for(rng)
    vmax = max(speeds)
    x_obs = C.BEHIND_START_M + C.BEFORE_EVENT_SEC * vmax + 1.5
    length = x_obs + C.AFTER_EVENT_SEC * vmax + C.END_TRIGGER_M + 2.5
    w = template_T1(env, rng, length, event_x=x_obs)
    hw = w / 2
    h = U(rng, C.EVENT_OBSTACLE_H_RANGE)
    for _ in range(50):
        if rng.random() < C.A_BOX_SHARE:
            along, across = U(rng, C.A_BOX_ALONG_RANGE), U(rng, C.A_BOX_ACROSS_RANGE)
            kind = 'box'
        else:
            r = U(rng, C.A_CYL_R_RANGE)
            along = across = 2 * r
            kind = 'cylinder'
        y = rng.uniform(-hw + across / 2, hw - across / 2) if across < w else 0.0
        if max(hw - (y + across / 2), (y - across / 2) + hw) >= C.EVENT_OBSTACLE_GAP_M:
            break
    else:
        return None
    obs = env.add_box('event', x_obs, y, along, across, h) if kind == 'box' else env.add_cyl('event', x_obs, y, across / 2, h)
    env.params['event_obstacle'] = {'kind': kind, 'x': x_obs, 'y': round(y, 3), 'along': round(along, 3),
                                    'across': round(across, 3), 'height': round(h, 2)}
    lim = hw - 0.6
    for k, v in enumerate(speeds):
        for _ in range(40):
            start = env.P(x_obs - (C.BEFORE_EVENT_SEC * v + 1.0), rng.uniform(-lim, lim))
            goal = env.P(x_obs + C.AFTER_EVENT_SEC * v + C.END_TRIGGER_M + 1.0, rng.uniform(-lim, lim))
            route = env.plan(start, goal)
            if route is None or detour_m(route) < C.DETOUR_MIN_M:
                continue
            ep = new_episode(env, k, start, goal, v, route,
                             {'type': 'detour', 'position': [obs.x, obs.y], 'obstacle': obs.name})
            if timing_ok(ep):
                env.episodes.append(ep)
                break
        else:
            return None
    return True


def episodes_B(env, rng):
    speeds = speeds_for(rng)
    vmax = max(speeds)
    w = U(rng, C.HALL_WIDTH_RANGE)
    hw = w / 2
    Xc = C.BEHIND_START_M + C.BEFORE_EVENT_SEC * vmax + 1.5 + hw
    ymax = hw + C.AFTER_EVENT_SEC * vmax + C.END_TRIGGER_M + 2.5
    x_out = Xc + hw
    bays1 = place_bays(rng, 0.5, Xc - hw - 0.5, rng.randint(1, 2), side=-1)   # leg 1: outer (bottom) wall
    bays2 = place_bays(rng, hw + 0.5, ymax - 0.5, rng.randint(1, 2), side=+1)  # leg 2: outer wall
    gaps1 = [(b, b + C.BAY_W) for b, _ in bays1]
    gaps2 = [(b, b + C.BAY_W) for b, _ in bays2]
    env.hwall_gaps(0.0, x_out, -hw, -1, gaps1)                            # leg 1 outer wall
    for g0, _ in gaps1:
        env.bay_h(g0, -hw, -1)
    env.hwall(0.0, Xc - hw, hw, +1, ext_b=False)                          # leg 1 inner wall
    env.vwall_gaps(-hw, ymax, x_out, +1, gaps2)                           # outer wall of the turn + leg 2
    for g0, _ in gaps2:
        env.bay_v(g0, x_out, +1)
    env.vwall(hw, ymax, Xc - hw, -1, ext_a=False)                         # leg 2 inner wall
    env.vwall(-hw, hw, 0.0, -1)                                           # back of leg 1
    env.hwall(Xc - hw, x_out, ymax, +1)                                   # end of leg 2
    env.add_rect(0.0, -hw, x_out, hw)
    env.add_rect(Xc - hw, hw, x_out, ymax)
    inner = env.P(Xc - hw, hw)
    env.params.update({'hall_width': round(w, 3), 'leg1': round(x_out, 2), 'leg2': round(ymax + hw, 2),
                       'turn': 'right' if env.m < 0 else 'left',
                       'bays': [[round(b, 2), s] for b, s in bays1 + bays2]})
    lim = hw - 0.6
    for k, v in enumerate(speeds):
        for _ in range(40):
            start = env.P(Xc - hw - (C.BEFORE_EVENT_SEC * v + 0.5), rng.uniform(-lim, lim))
            goal = env.P(Xc + rng.uniform(-lim, lim), hw + C.AFTER_EVENT_SEC * v + C.END_TRIGGER_M + 1.0)
            route = env.plan(start, goal)
            if route is None:
                continue
            ep = new_episode(env, k, start, goal, v, route, {'type': 'corner', 'position': list(inner)})
            if timing_ok(ep):
                env.episodes.append(ep)
                break
        else:
            return None
    return True


def episodes_D(env, rng):
    speeds = speeds_for(rng)
    vmax = max(speeds)
    W, dw = U(rng, C.ROOM_WIDTH_RANGE), U(rng, C.DOOR_WIDTH_RANGE)
    hW, T = W / 2, C.WALL_T
    xL = -(C.BEHIND_START_M + C.BEFORE_EVENT_SEC * vmax + 1.5)
    xR = C.AFTER_EVENT_SEC * vmax + C.END_TRIGGER_M + 2.5
    yd = rng.uniform(-(hW - dw / 2 - 0.6), hW - dw / 2 - 0.6)
    env.hwall(xL, xR, hW, +1)
    env.hwall(xL, xR, -hW, -1)
    env.vwall(-hW, hW, xL, -1)
    env.vwall(-hW, hW, xR, +1)
    # dividing wall centred on x = 0, door opening [yd - dw/2, yd + dw/2]
    lo, hi = yd - dw / 2, yd + dw / 2
    env.add_box('wall', 0.0, (-hW + lo) / 2, T, lo + hW, C.WALL_H)
    env.add_box('wall', 0.0, (hi + hW) / 2, T, hW - hi, C.WALL_H)
    for x_face, sgn in ((-T / 2, -1), (T / 2, +1)):           # furniture faces on the dividing wall
        for a, b in ((-hW, lo), (hi, hW)):
            if b - a > 1.5:
                p0, p1 = env.P(x_face, a + 0.3), env.P(x_face, b - 0.3)
                env.faces.append((*p0, *p1, float(sgn), 0.0))
    env.add_rect(xL, -hW, -T / 2, hW)
    env.add_rect(T / 2, -hW, xR, hW)
    env.params.update({'room_width': round(W, 3), 'door_width': round(dw, 3), 'door_y': round(env.m * yd, 3),
                       'room1': round(-xL, 2), 'room2': round(xR, 2)})
    lim = hW - 0.8
    door = env.P(0.0, yd)
    for k, v in enumerate(speeds):
        for _ in range(60):
            start = env.P(-(C.BEFORE_EVENT_SEC * v + 1.0), rng.uniform(-lim, lim))
            goal = env.P(C.AFTER_EVENT_SEC * v + C.END_TRIGGER_M + 1.0, rng.uniform(-lim, lim))
            route = env.plan(start, goal)
            if route is None or detour_m(route) < C.DETOUR_MIN_M:
                continue
            s_cross = crossing_s(route, 0.0)
            if s_cross is None:
                continue
            ep = new_episode(env, k, start, goal, v, route, {'type': 'door', 'position': list(door), 's': s_cross})
            if timing_ok(ep):
                env.episodes.append(ep)
                break
        else:
            return None
    return True


def crossing_s(route, x_line):
    pl = M.Polyline(route)
    for i in range(len(route) - 1):
        (x0, _), (x1, _) = route[i], route[i + 1]
        if (x0 - x_line) * (x1 - x_line) <= 0 and x0 != x1:
            t = (x_line - x0) / (x1 - x0)
            return float(pl.s[i] + t * pl.seg_len[i])
    return None


def episodes_C(env, rng):
    speeds = speeds_for(rng)
    vmax = max(speeds)
    length = C.BEHIND_START_M + C.BEFORE_EVENT_SEC * vmax + 1.0 + max(C.STOP_GAP_RANGE)
    w = template_T1(env, rng, length)
    hw = w / 2
    lim = hw - 0.6
    for k, v in enumerate(speeds):
        gap = U(rng, C.STOP_GAP_RANGE)
        stop = {'slowdown_m': round(U(rng, C.SLOWDOWN_DISTANCE_RANGE), 2), 'stand_sec': C.STAND_SEC,
                'gap_to_wall': round(gap, 2)}
        y = rng.uniform(-lim, lim)
        x_stop = length - gap
        # walking time to the stop = (L - Ds)/v + 2 Ds/v  >=  BEFORE_EVENT_SEC
        L = C.BEFORE_EVENT_SEC * v - stop['slowdown_m'] + 0.5
        start, goal = env.P(x_stop - L, y), env.P(x_stop, y)
        route = env.plan(start, goal)
        if route is None:
            return None
        tl = M.ped_timeline(M.Polyline(route), v, stop)
        t_stop = tl['t_end'] - stop['stand_sec']
        ep = new_episode(env, k, start, goal, v, route,
                         {'type': 'stop', 'position': list(goal), 's': M.Polyline(route).length}, stop=stop)
        ep['t_before_sec'] = round(t_stop, 2)
        env.episodes.append(ep)
    return True


def episodes_E_straight(env, rng):
    speeds = speeds_for(rng)
    vmax = max(speeds)
    total = (C.BEFORE_EVENT_SEC + C.AFTER_EVENT_SEC) * vmax + C.END_TRIGGER_M + 1.0
    length = C.BEHIND_START_M + total + 1.5
    w = template_T1(env, rng, length)
    lim = w / 2 - 0.6
    x_goal = length - 1.5
    for k, v in enumerate(speeds):
        y = rng.uniform(-lim, lim)
        L = (C.BEFORE_EVENT_SEC + C.AFTER_EVENT_SEC) * v + C.END_TRIGGER_M + 1.0
        start, goal = env.P(x_goal - L, y), env.P(x_goal, y)
        route = env.plan(start, goal)
        if route is None or len(route) != 2:
            return None
        env.episodes.append(new_episode(env, k, start, goal, v, route, None))
    env.params['shape'] = 'straight'
    return True


def episodes_E_arc(env, rng):
    speeds = speeds_for(rng)
    vmax = max(speeds)
    Lp = (C.BEFORE_EVENT_SEC + C.AFTER_EVENT_SEC) * vmax + C.END_TRIGGER_M + 1.0
    # CHANGED (2026-11-03): the curve starts within ARC_LEAD_MAX_M of the path start, so every kept
    # sample's history already shows it (an arc whose onset lies in the future is not explained by anything)
    sweep = U(rng, C.ARC_SWEEP_RANGE)
    lead = U(rng, (0.0, C.ARC_LEAD_MAX_M))
    R = (Lp - lead) / sweep
    lead = Lp - R * sweep                         # straight lead-in, then the arc
    pts = []
    for s in np.arange(0.0, Lp + 1e-9, C.ARC_WAYPOINT_STEP_M):
        if s <= lead:
            pts.append((s, 0.0))
        else:
            a = (s - lead) / R
            pts.append((lead + R * math.sin(a), R * (1 - math.cos(a))))
    pts = [(x + C.BEHIND_START_M + 1.0, y) for x, y in pts]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx = 2.5
    x0, x1 = 0.0, max(xs) + mx
    y0, y1 = min(ys) - mx, max(ys) + mx
    env.hwall(x0, x1, y1, +1)
    env.hwall(x0, x1, y0, -1)
    env.vwall(y0, y1, x0, -1)
    env.vwall(y0, y1, x1, +1)
    env.add_rect(x0, y0, x1, y1)
    path_o = [env.P(x, y) for x, y in pts]
    path_pl = M.Polyline(path_o)
    # columns on a grid, away from the path and the walls
    S = U(rng, C.COLUMN_SPACING_RANGE)
    r = U(rng, C.COLUMN_R_RANGE)
    n_col = 0
    for cx in np.arange(x0 + S / 2, x1, S):
        for cy in np.arange(y0 + S / 2, y1, S):
            q = env.P(cx, cy)
            if path_pl.distance_to(q) >= 1.5 + r and min(cx - x0, x1 - cx, cy - y0, y1 - cy) >= 0.8 + r \
                    and rng.random() < 0.7:
                env.add_cyl('column', cx, cy, r, C.WALL_H)
                n_col += 1
    env.params.update({'shape': 'arc', 'radius': round(R, 2), 'sweep_deg': round(math.degrees(sweep), 1),
                       'lead_in': round(lead, 2), 'hall': [round(x1 - x0, 2), round(y1 - y0, 2)],
                       'columns': n_col, 'column_spacing': round(S, 2),
                       'turn': 'right' if env.m < 0 else 'left'})
    for k, v in enumerate(speeds):
        L = (C.BEFORE_EVENT_SEC + C.AFTER_EVENT_SEC) * v + C.END_TRIGGER_M + 1.0
        s0 = path_pl.length - L
        i0 = int(np.searchsorted(path_pl.s, s0))
        route = [tuple(path_pl.at([s0])[0])] + [tuple(p) for p in path_o[i0:] if True]
        route = [route[0]] + [p for p in route[1:] if math.dist(p, route[0]) > 0.05]
        if min(env.clear_at(*p) for p in path_pl.at(np.arange(s0, path_pl.length, 0.1))) < CLEARANCE_M:
            return None
        env.episodes.append(new_episode(env, k, route[0], route[-1], v, route, None, path=True))
    return True


# ===========================================================================
# Furniture (distractors against the walls)
# ===========================================================================
def add_furniture(env, rng):
    n_want = rng.randint(*C.FURNITURE_PER_ENV)
    routes = [M.Polyline(ep['route']) for ep in env.episodes]
    ends = [tuple(ep['start']) for ep in env.episodes] + [tuple(ep['goal']) for ep in env.episodes]
    event_obs = [o for o in env.obstacles if o.role == 'event']
    added = []
    for _ in range(40):
        if len(added) >= n_want or not env.faces:
            break
        x0, y0, x1, y1, nx, ny = rng.choice(env.faces)
        L = math.hypot(x1 - x0, y1 - y0)
        kind = rng.choice(('cabinet', 'cabinet', 'bin', 'vending'))
        if kind == 'cabinet':
            along, depth = rng.uniform(0.6, 1.5), rng.uniform(0.4, 0.6)
        elif kind == 'vending':
            along, depth = 0.9, 0.8
        else:
            along = depth = 2 * rng.uniform(0.2, 0.35)
        if L < along + 0.6:
            continue
        t = rng.uniform(0.3 + along / 2, L - 0.3 - along / 2) / L
        cx = x0 + t * (x1 - x0) + nx * (depth / 2 + 0.02)
        cy = y0 + t * (y1 - y0) + ny * (depth / 2 + 0.02)
        h = U(rng, C.FURNITURE_H_RANGE)
        if kind == 'bin':
            o = Obstacle(f'furniture_{env._n:02d}', 'cylinder', cx, cy, radius=along / 2)
        else:
            yaw = math.atan2(y1 - y0, x1 - x0)
            o = Obstacle(f'furniture_{env._n:02d}', 'box', cx, cy, yaw, along, depth)
        o.height, o.role = h, 'furniture'
        if any(min(o.distance(*q) for q in pl.at(np.arange(0, pl.length, 0.1))) < C.FURNITURE_ROUTE_CLEAR_M
               for pl in routes):
            continue
        if any(o.distance(*q) < 1.0 for q in ends):
            continue
        if any(math.hypot(o.x - e.x, o.y - e.y) < 2.5 for e in event_obs):
            continue
        if any(o.distance(a.x, a.y) < 0.5 for a in env.obstacles if a.role in ('furniture', 'column', 'event')):
            continue
        env._n += 1
        env.obstacles.append(o)
        env._wcache = None
        # the bridge must still plan the same routes
        same = True
        for ep in env.episodes:
            if ep['path_mode']:
                continue
            r = env.plan(tuple(ep['start']), tuple(ep['goal']))
            if r is None or len(r) != len(ep['route']) or \
                    max(math.dist(a, b) for a, b in zip(r, ep['route'])) > 0.05:
                same = False
                break
        if not same:
            env.remove(o)
            continue
        o.kind_label = kind
        added.append(o)
    env.params['furniture'] = len(added)


# ===========================================================================
# Robot placement
# ===========================================================================
def robot_candidates(env, route_pl, ep, rng):
    pts = []
    g = C.ROBOT_CANDIDATE_GRID_M
    ends = [tuple(ep['start']), tuple(ep['goal'])]
    for x0, y0, x1, y1 in env.rects:
        for x in np.arange(x0 + g / 2, x1, g):
            for y in np.arange(y0 + g / 2, y1, g):
                pts.append((float(x), float(y)))
    rng.shuffle(pts)
    out = []
    for p in pts:
        if len(out) >= C.ROBOT_CANDIDATE_TRIES:
            break
        if env.clear_at(*p) < C.ROBOT_WALL_CLEAR_M:
            continue
        if route_pl.distance_to(p) < C.ROBOT_ROUTE_CLEAR_M:
            continue
        if min(math.dist(p, q) for q in ends) < C.ROBOT_START_CLEAR_M:
            continue
        out.append(p)
    return out


def event_time(tl, ep):
    ev = ep['event']
    if ev is None:
        return None
    if ev['type'] == 'stop':
        return tl['t_end'] - ep['stop']['stand_sec']
    return M.time_at_s(tl, ev['s'])


def plan_ok(tl, ep, seen, st):
    """NEW (2026-11-03): a plan counts only if the whole walk reaches USABLE_TARGET AND the samples
    around the event (just before and while the pedestrian passes the door / corner / obstacle) are
    usable too. Before, a robot that saw only the first half of the walk (the door beyond its 8 m
    range) passed with a good overall share."""
    if st[2] < C.USABLE_TARGET:
        return False
    t_ev = event_time(tl, ep)
    if t_ev is None:
        return True
    n, _, share = M.event_usable_stats(tl, seen, t_ev)
    ep['_ev'] = (n, share)            # the last call before a placement is accepted belongs to that placement
    return n >= C.EVENT_MIN_SAMPLES and share >= C.EVENT_USABLE_TARGET


def place_robot(env, ep, rng):
    """Choose a robot behaviour for the episode with planned usable share >= USABLE_TARGET."""
    route_pl = M.Polyline(ep['route'])
    tl = M.ped_timeline(route_pl, ep['speed'], ep['stop'])
    modes = list(C.MODE_SHARE)
    first = rng.choices(modes, weights=[C.MODE_SHARE[m] for m in modes])[0]
    order = [first] + rng.sample([m for m in modes if m != first], len(modes) - 1)
    T = M.kept_anchor_times(tl['t_end'])
    if len(T) == 0:
        return None
    probe_t = [T[0], T[len(T) // 3], T[2 * len(T) // 3], T[-1]]
    cands = None
    for mode in order:
        if mode == 'follow':
            dists = list(np.arange(C.FOLLOW_DIST_RANGE[0], C.FOLLOW_DIST_RANGE[1] + 1e-9, 0.5))
            rng.shuffle(dists)
            for d in dists:
                p0 = route_pl.at([-d])[0]
                if env.clear_at(*p0) < C.ROBOT_WALL_CLEAR_M:
                    continue
                robot = {'mode': 'follow', 'dist': float(d)}
                rxy, rh, seen, st = M.evaluate(tl, route_pl, robot, env.obstacles)
                if min(env.clear_at(*q) for q in rxy[::3]) < C.ROBOT_PATH_CLEAR_M:
                    break                     # the follower would scrape a wall (door): no follow here
                if plan_ok(tl, ep, seen, st):
                    robot['start_pose'] = [round(float(rxy[0, 0]), 3), round(float(rxy[0, 1]), 3), round(float(rh[0]), 4)]
                    return robot, st, mode != first
            continue
        if cands is None:
            cands = robot_candidates(env, route_pl, ep, rng)
        for p in cands:
            if mode == 'watch':
                h0 = math.atan2(tl['xy'][0, 1] - p[1], tl['xy'][0, 0] - p[0])
                robot = {'mode': 'watch', 'pose': (p[0], p[1], h0)}
                rxy, rh, seen, st = M.evaluate(tl, route_pl, robot, env.obstacles)
                if plan_ok(tl, ep, seen, st):
                    robot['start_pose'] = [round(p[0], 3), round(p[1], 3), round(h0, 4)]
                    del robot['pose']
                    return robot, st, mode != first
            else:
                heads = []
                for t in probe_t:
                    q = tl['xy'][min(int(round(t / tl['dt'])), len(tl['t']) - 1)]
                    heads.append(math.atan2(q[1] - p[1], q[0] - p[0]))
                if ep['event']:
                    q = ep['event']['position']
                    heads.append(math.atan2(q[1] - p[1], q[0] - p[0]))
                for h in heads:
                    robot = {'mode': 'still', 'pose': (p[0], p[1], h)}
                    rxy, rh, seen, st = M.evaluate(tl, route_pl, robot, env.obstacles)
                    if plan_ok(tl, ep, seen, st):
                        robot['start_pose'] = [round(p[0], 3), round(p[1], 3), round(h, 4)]
                        del robot['pose']
                        return robot, st, mode != first
    return None


def finish_episode(env, ep, robot, st):
    route_pl = M.Polyline(ep['route'])
    tl = M.ped_timeline(route_pl, ep['speed'], ep['stop'])
    ep['robot'] = robot
    ep['planned'] = {'t_end_sec': round(tl['t_end'], 2), 'n_kept': st[0], 'n_usable': st[1],
                     'usable_share': round(st[2], 3)}
    if '_ev' in ep:
        ep['planned']['event_samples'] = int(ep['_ev'][0])
        ep['planned']['event_usable_share'] = round(float(ep['_ev'][1]), 3)
        del ep['_ev']
    if ep['event'] is not None:
        if ep['event']['type'] == 'stop':
            ep['event']['t_planned_sec'] = round(tl['t_end'] - ep['stop']['stand_sec'], 2)
        else:
            ep['event']['t_planned_sec'] = round(M.time_at_s(tl, ep['event']['s']), 2)


# ===========================================================================
# One environment
# ===========================================================================
EPISODE_BUILDERS = {'A': episodes_A, 'B': episodes_B, 'C': episodes_C, 'D': episodes_D}
TEMPLATES = {'A': 'T1_hallway', 'B': 'T2_corner', 'C': 'T1_dead_end', 'D': 'T3_rooms_door'}


def generate_env(scenario, idx, seed, log=print):
    for attempt in range(30):
        rng = random.Random(seed * 1000 + attempt)
        env_id = f'env_{scenario}_{idx:03d}'
        if scenario == 'E':
            arc = idx % 2 == 1
            env = Env(env_id, 'E', 'T4_hall_arc' if arc else 'T1_hallway', rng, rng.random() < 0.5)
            ok = (episodes_E_arc if arc else episodes_E_straight)(env, rng)
        else:
            env = Env(env_id, scenario, TEMPLATES[scenario], rng, rng.random() < 0.5)
            ok = EPISODE_BUILDERS[scenario](env, rng)
        if not ok:
            continue
        add_furniture(env, rng)
        fallbacks = 0
        for ep in env.episodes:
            res = place_robot(env, ep, rng)
            if res is None:
                ok = False
                break
            robot, st, fb = res
            fallbacks += fb
            finish_episode(env, ep, robot, st)
        if not ok:
            log(f'  {env_id}: attempt {attempt + 1}: an episode had no robot placement with '
                f'{C.USABLE_TARGET:.0%} usable, regenerating')
            continue
        env.params.update({'seed': seed, 'attempt': attempt, 'mode_fallbacks': fallbacks})
        x_min = min(r[0] for r in env.rects) - C.WALL_T
        env.translate(C.ENV_OFFSET_X - x_min, 0.0)
        return env
    raise RuntimeError(f'{scenario} {idx}: no valid environment in 30 attempts')


# ===========================================================================
# Output
# ===========================================================================
def scene_yaml(env):
    lines = [f'# {env.env_id} ({C.SCENARIOS[env.scenario]}, template {env.template}) -- generated by '
             f'environment_generator.py {GENERATOR_VERSION}',
             '# WORLD frame (Gazebo). The same obstacles in the odom/RViz frame are in environment.json.',
             f'# Load:  python3 spawn_obstacles.py <this file>   ({len(env.obstacles)} obstacles)',
             'obstacles:']
    for o in env.world_obstacles():
        lines += [f'  - name: {o.name}', f'    type: {o.kind}', f'    x: {o.x:.4f}', f'    y: {o.y:.4f}']
        if o.kind == 'box':
            lines += [f'    yaw: {math.atan2(o.s, o.c):.5f}', f'    size_x: {2 * o.hx:.4f}', f'    size_y: {2 * o.hy:.4f}']
        else:
            lines += [f'    radius: {o.r:.4f}']
        lines += [f'    height: {o.height:.3f}']
    return '\n'.join(lines) + '\n'


def obstacle_dict(o):
    d = {'name': o.name, 'role': o.role, 'type': o.kind, 'x': round(o.x, 4), 'y': round(o.y, 4),
         'height': round(o.height, 3)}
    if o.kind == 'box':
        d.update({'yaw': round(o.yaw, 5), 'size_x': round(2 * o.hx, 4), 'size_y': round(2 * o.hy, 4)})
    else:
        d['radius'] = round(o.r, 4)
    return d


def env_json(env):
    return {'environment_id': env.env_id, 'scenario': env.scenario, 'scenario_name': C.SCENARIOS[env.scenario],
            'template': env.template, 'generator_version': GENERATOR_VERSION, 'frame': 'odom (RViz)',
            'params': env.params, 'rects': [[round(v, 3) for v in r] for r in env.rects],
            'obstacles': [obstacle_dict(o) for o in env.obstacles],
            'episodes': env.episodes}


def draw(env, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

    fig, ax = plt.subplots(figsize=(12, 7))
    for x0, y0, x1, y1 in env.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color='#f2f2f2', zorder=0))
    colors = {'wall': '#444444', 'event': '#d62728', 'furniture': '#8c564b', 'column': '#7f7f7f'}
    for o in env.obstacles:
        c = colors.get(o.role, 'k')
        if o.kind == 'box':
            ax.add_patch(Polygon(o.corners(0.0), closed=True, color=c, zorder=2))
        else:
            ax.add_patch(Circle((o.x, o.y), o.r, color=c, zorder=2))
    cmap = plt.get_cmap('tab10')
    for ep in env.episodes:
        c = cmap(ep['index'] % 10)
        r = np.array(ep['route'])
        ax.plot(r[:, 0], r[:, 1], '-', color=c, lw=1.5, zorder=3)
        ax.plot(*ep['start'], 'o', color=c, ms=6, zorder=4)
        ax.plot(*ep['goal'], 'x', color=c, ms=8, mew=2, zorder=4)
        if ep['event']:
            ax.plot(*ep['event']['position'], '*', color=c, ms=12, mec='k', zorder=5)
        rb = ep['robot']
        pl = M.Polyline(ep['route'])
        tl = M.ped_timeline(pl, ep['speed'], ep['stop'])
        if rb['mode'] == 'follow':
            rxy, rh = M.robot_follow(tl, pl, rb['dist'])
            ax.plot(rxy[:, 0], rxy[:, 1], ':', color=c, lw=1.2, zorder=3)
            k = len(rxy) // 2
            rx, ry, h = rxy[k, 0], rxy[k, 1], rh[k]
        else:
            rx, ry, h = rb['start_pose']
            if rb['mode'] == 'watch':
                _, hh = M.robot_watch(tl, (rx, ry), h)
                h = hh[len(hh) // 2]
        ax.add_patch(Circle((rx, ry), C.ROBOT_RADIUS_M, fc='none', ec=c, lw=2, zorder=6))
        ax.add_patch(Wedge((rx, ry), C.CAM_RANGE_M, math.degrees(h - C.CAM_HALF_FOV),
                           math.degrees(h + C.CAM_HALF_FOV), color=c, alpha=0.07, zorder=1))
        ax.plot([], [], '-', color=c, label=f"ep{ep['index']}: {ep['speed']:.2f} m/s, {rb['mode']}"
                f"{' d=%.1f' % rb['dist'] if rb['mode'] == 'follow' else ''}, usable {ep['planned']['usable_share']:.0%}")
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f'{env.env_id}  {C.SCENARIOS[env.scenario]}  {env.template}  (odom/RViz frame, m)\n'
                 f'o start  x goal  * event  circle = robot (wedge: camera, at the middle of the walk)', fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def write_env(env, out_dir, preview=True):
    d = out_dir / env.env_id
    d.mkdir(parents=True, exist_ok=True)
    (d / 'scene.yaml').write_text(scene_yaml(env))
    (d / 'environment.json').write_text(json.dumps(env_json(env), indent=1))
    if preview:
        draw(env, d / 'preview.png')
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=list(C.SCENARIOS))
    ap.add_argument('--all', action='store_true', help='all scenarios A-E')
    ap.add_argument('--envs', type=int, default=3, help='environments per scenario')
    ap.add_argument('--first', type=int, default=0, help='index of the first environment')
    ap.add_argument('--seed', type=int, default=1, help='base seed (same seed = same environments)')
    ap.add_argument('--dataset', default='dev', help='dataset folder name under datasets/ (default: dev)')
    ap.add_argument('--out', default=None, help='explicit output folder (overrides --dataset)')
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--stand-sec', type=float, default=None,
                    help='scenario C: standing time at the stop in s (default: STAND_SEC in mvp_config.py, 10). '
                         'Use e.g. 20 for a test that needs many standing samples.')
    args = ap.parse_args()
    if args.stand_sec is not None:
        C.STAND_SEC = args.stand_sec
    if not args.all and not args.scenario:
        sys.exit('Give --scenario A|B|C|D|E or --all')
    scen = list(C.SCENARIOS) if args.all else [args.scenario]
    if args.out is None:
        args.out = str(DATASETS_DIR / args.dataset)
    for si, s in enumerate(scen):
        out = Path(args.out) / C.SCENARIOS[s]
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        print(f'== {C.SCENARIOS[s]}: {args.envs} environment(s) -> {out}')
        for i in range(args.first, args.first + args.envs):
            t0 = time.time()
            seed = args.seed * 100000 + (ord(s) - ord('A')) * 10000 + i
            env = generate_env(s, i, seed)
            write_env(env, out, preview=not args.no_preview)
            for ep in env.episodes:
                rows.append([env.env_id, ep['episode_id'], ep['speed'], ep['robot']['mode'],
                             ep['planned']['n_kept'], ep['planned']['usable_share'],
                             ep['t_before_sec'], ep['t_after_sec']])
            shares = [ep['planned']['usable_share'] for ep in env.episodes]
            modes = [ep['robot']['mode'] for ep in env.episodes]
            print(f'  {env.env_id}: {len(env.obstacles)} obstacles, {len(env.episodes)} episodes, modes '
                  f'{"/".join(m[0] for m in modes)}, planned usable {min(shares):.0%}-{max(shares):.0%}, '
                  f'{time.time() - t0:.1f} s')
        summary = out / 'summary.csv'
        new = not summary.exists()
        with open(summary, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['environment', 'episode', 'speed', 'robot_mode', 'planned_kept', 'planned_usable_share',
                            't_before_sec', 't_after_sec'])
            w.writerows(rows)


if __name__ == '__main__':
    main()