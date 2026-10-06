#!/usr/bin/env python3
"""
world_obstacles.py -- the obstacles the pedestrian knows about, and how it
walks around them (2026-10-05). Pure Python + numpy, no ROS: the bridge
(hunav_model_bridge_nav.py) and the checks import it.

WHY
HuNav only avoids obstacles it is TOLD about: every /compute_agents request
carries, per agent, a list of nearby obstacle points (Agent.closest_obs), and
the social-force model pushes the agent away from them. Our bridge never
filled that list, so the pedestrian walked straight through a box.

Filling it is necessary but NOT enough. The social force only pushes AWAY
from the nearest point of an obstacle. Walking straight at the flat face of a
box, that push points straight back along the path: no sideways component,
so the agent slows down and stops in front of the box forever (a simulation
of HuNav's own equations, dt 0.1 s, confirmed this for every start offset
smaller than the half-width of the box). People do not do that: they see the
box from far away and pick a way around it. So this module does both:

  1. OBSTACLE POINTS for HuNav (closest_points): one point per obstacle, the
     closest point of its outline to the agent, for obstacles within
     OBSTACLE_RANGE_M -- the same thing the official HuNav Gazebo plugin
     sends (HuNavSystemPlugin_fortress.cpp, getObstacles). This keeps the
     close-range behaviour realistic (keeps some distance when passing).
  2. A ROUTE (plan_route): the shortest path from the agent to its goal that
     keeps CLEARANCE_M between the walking line and every obstacle, found on
     a visibility graph over the corners of the obstacles grown by
     CLEARANCE_M. The bridge then gives HuNav the route's next corner as its
     goal instead of the far goal (RouteFollower), and moves on to the
     following corner as soon as that one can be seen -- so corners are cut
     smoothly, like a person does, instead of being touched.

OBSTACLE FILE (obstacles.yaml, WORLD frame = Gazebo's coordinates, the numbers
Gazebo's GUI shows for a model's pose):

    obstacles:
      - name: box1
        type: box          # box: x, y, yaw (rad, optional), size_x, size_y
        x: 3.0
        y: 0.0
        yaw: 0.0
        size_x: 1.0
        size_y: 1.0
      - name: pillar
        type: cylinder     # cylinder: x, y, radius
        x: 5.0
        y: 2.0
        radius: 0.3

The file is the ground truth only if it matches what is placed in Gazebo.
analyze_obstacle_clearance.py checks that against the occupancy grid.
"""
import heapq
import math
from pathlib import Path

import numpy as np

CLEARANCE_M = 0.6          # PREFERRED gap walking line <-> obstacle outline, metres.
                           # A person's half-width is ~0.25 m, so this leaves
                           # ~0.35 m of air between shoulder and box.
MIN_CLEARANCE_M = 0.35     # the least a person accepts (squeezing through a
                           # door or a narrow gap: ~0.1 m of air at the shoulder).
                           # A gap narrower than 2 x this (0.7 m) is not used.
DETOUR_FACTOR = 1.3        # take the narrow way when the comfortable route is
                           # more than this much longer (people do not walk 30%
                           # further just to keep extra distance from a box).
OBSTACLE_RANGE_M = 2.0     # obstacle points further than this are not sent to
                           # HuNav. The official plugin uses 5 m, but HuNav
                           # AVERAGES the force over all points it gets, so far
                           # points (force ~0) would only weaken the near one.
CHECK_STEP_M = 0.05        # sampling step when testing a straight segment


class Obstacle:
    """One obstacle in the world frame: a (possibly rotated) box or a cylinder."""

    def __init__(self, name, kind, x, y, yaw=0.0, size_x=0.0, size_y=0.0, radius=0.0):
        if kind not in ('box', 'cylinder'):
            raise ValueError(f'obstacle {name}: type must be box or cylinder, not {kind!r}')
        if kind == 'box' and (size_x <= 0 or size_y <= 0):
            raise ValueError(f'obstacle {name}: a box needs size_x and size_y > 0')
        if kind == 'cylinder' and radius <= 0:
            raise ValueError(f'obstacle {name}: a cylinder needs radius > 0')
        self.name, self.kind = name, kind
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.hx, self.hy, self.r = 0.5 * float(size_x), 0.5 * float(size_y), float(radius)
        self.c, self.s = math.cos(self.yaw), math.sin(self.yaw)

    def closest_point(self, px, py):
        """Closest point of the outline (or interior) to (px, py), world frame."""
        if self.kind == 'cylinder':
            dx, dy = px - self.x, py - self.y
            d = math.hypot(dx, dy)
            if d <= self.r:
                return px, py
            return self.x + dx * self.r / d, self.y + dy * self.r / d
        dx, dy = px - self.x, py - self.y
        lx, ly = self.c * dx + self.s * dy, -self.s * dx + self.c * dy
        lx, ly = min(max(lx, -self.hx), self.hx), min(max(ly, -self.hy), self.hy)
        return self.x + self.c * lx - self.s * ly, self.y + self.s * lx + self.c * ly

    def distance(self, px, py):
        """Distance from (px, py) to the obstacle; 0 inside it."""
        cx, cy = self.closest_point(px, py)
        return math.hypot(px - cx, py - cy)

    def corners(self, grow):
        """Route nodes around the obstacle grown by `grow`: their distance to
        the obstacle is >= grow, and the straight lines between neighbouring
        nodes stay >= grow away too."""
        if self.kind == 'box':
            out = []
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                lx, ly = sx * (self.hx + grow), sy * (self.hy + grow)
                out.append((self.x + self.c * lx - self.s * ly, self.y + self.s * lx + self.c * ly))
            return out
        n = 8                                      # octagon AROUND the grown circle
        R = (self.r + grow) / math.cos(math.pi / n)
        return [(self.x + R * math.cos(2 * math.pi * k / n + math.pi / n),
                 self.y + R * math.sin(2 * math.pi * k / n + math.pi / n)) for k in range(n)]

    def describe(self):
        if self.kind == 'box':
            return (f'{self.name}: box at ({self.x:.2f}, {self.y:.2f}) yaw {math.degrees(self.yaw):.0f} deg, '
                    f'{2 * self.hx:.2f} x {2 * self.hy:.2f} m')
        return f'{self.name}: cylinder at ({self.x:.2f}, {self.y:.2f}), radius {self.r:.2f} m'


def load_obstacles(path):
    """Read obstacles.yaml. Missing file -> []. A broken entry raises ValueError."""
    import yaml
    path = Path(path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    out = []
    for i, e in enumerate(data.get('obstacles') or []):
        name = str(e.get('name', f'obstacle{i + 1}'))
        kind = str(e.get('type', 'box')).lower()
        try:
            out.append(Obstacle(name, kind, e['x'], e['y'], e.get('yaw', 0.0),
                                e.get('size_x', 0.0), e.get('size_y', 0.0), e.get('radius', 0.0)))
        except KeyError as k:
            raise ValueError(f'obstacle {name}: missing {k}') from None
    return out


def closest_points(px, py, obstacles, max_range=OBSTACLE_RANGE_M):
    """The points to put into Agent.closest_obs: one per obstacle in range."""
    pts = []
    for o in obstacles:
        cx, cy = o.closest_point(px, py)
        if math.hypot(px - cx, py - cy) <= max_range:
            pts.append((cx, cy))
    return pts


def min_distance(px, py, obstacles):
    """(distance, obstacle) of the nearest obstacle, (inf, None) if none."""
    best = (math.inf, None)
    for o in obstacles:
        d = o.distance(px, py)
        if d < best[0]:
            best = (d, o)
    return best


def segment_clear(a, b, obstacles, clearance, relax_a=False, relax_b=False):
    """True if every point of a->b keeps `clearance` from every obstacle.
    relax_a / relax_b: the end point itself may already be closer than
    `clearance` (the agent standing next to a box, a goal next to a wall);
    then only require that the segment gets no closer than that end does."""
    ax, ay = a
    bx, by = b
    L = math.hypot(bx - ax, by - ay)
    n = max(1, int(math.ceil(L / CHECK_STEP_M)))
    for o in obstacles:
        need = clearance
        if relax_a:
            need = min(need, o.distance(ax, ay) - 1e-3)
        if relax_b:
            need = min(need, o.distance(bx, by) - 1e-3)
        if need <= 0:
            need = 1e-3                     # still never go THROUGH it
        for k in range(n + 1):
            t = k / n
            if o.distance(ax + t * (bx - ax), ay + t * (by - ay)) < need:
                return False
    return True


def plan_route(start, goal, obstacles, clearance=CLEARANCE_M):
    """Shortest route start -> goal keeping `clearance` from every obstacle.
    Returns the list of points to walk to AFTER start, ending with goal
    ([goal] when the straight line is free), or None when there is no route
    (goal inside an obstacle, or walled in)."""
    if not obstacles or segment_clear(start, goal, obstacles, clearance, True, True):
        return [tuple(goal)]
    if min_distance(*goal, obstacles)[0] <= 0.05:
        return None
    grow = clearance * 1.02                 # nodes a hair outside the limit
    nodes = [tuple(start), tuple(goal)]
    for o in obstacles:
        for p in o.corners(grow):
            if min_distance(*p, obstacles)[0] >= clearance:   # not inside a neighbour
                nodes.append(p)
    n = len(nodes)
    dist = [math.inf] * n
    prev = [-1] * n
    dist[0] = 0.0
    heap = [(0.0, 0)]
    done = [False] * n
    while heap:
        d, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        if i == 1:
            break
        for j in range(1, n):
            if done[j]:
                continue
            nd = d + math.dist(nodes[i], nodes[j])
            if nd >= dist[j]:
                continue
            if segment_clear(nodes[i], nodes[j], obstacles, clearance, relax_a=(i == 0), relax_b=(j == 1)):
                dist[j], prev[j] = nd, i
                heapq.heappush(heap, (nd, j))
    if not done[1]:
        return None
    path, k = [], 1
    while k != 0:
        path.append(nodes[k])
        k = prev[k]
    return path[::-1]


def route_length(start, route):
    pts = [tuple(start)] + list(route)
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def plan_human_route(start, goal, obstacles, clearance=CLEARANCE_M, min_clearance=MIN_CLEARANCE_M):
    """Comfortable route if there is one that is not much longer than the
    tightest possible one; otherwise the tight one. Returns (route, clearance
    used) or (None, None)."""
    wide = plan_route(start, goal, obstacles, clearance)
    if min_clearance >= clearance:
        return (wide, clearance) if wide else (None, None)
    tight = plan_route(start, goal, obstacles, min_clearance)
    if tight is None:
        return (wide, clearance) if wide else (None, None)
    if wide is None or route_length(start, wide) > DETOUR_FACTOR * route_length(start, tight):
        return tight, min_clearance
    return wide, clearance


class RouteFollower:
    """Turns a list of goals into the single point HuNav should walk to now.

    goals      : [(x, y)] world frame, the goals the user asked for
    cyclic     : after the last goal start again at the first (like HuNav)
    reach_m    : a GOAL counts as reached within this distance (HuNav's own
                 goal radius + 0.1, so HuNav and this class agree)
    Waypoints (route corners) are passed as soon as the next point of the
    route is visible with full clearance -- that cuts corners smoothly -- or
    when the agent is within reach_m of them.
    """

    def __init__(self, goals, obstacles, cyclic, reach_m, clearance=CLEARANCE_M,
                 min_clearance=MIN_CLEARANCE_M, log=None):
        self.goals = [tuple(g) for g in goals]
        self.obstacles = list(obstacles)
        self.cyclic = cyclic
        self.reach = reach_m
        self.pref_clearance = clearance
        self.min_clearance = min_clearance
        self.clearance = clearance          # the one the current route was planned with
        self.log = log or (lambda lvl, msg: None)
        self.gi = 0                 # index of the goal being walked to
        self.route = None           # remaining route points, last = goals[gi]
        self.finished = False

    def set_obstacles(self, obstacles):
        self.obstacles = list(obstacles)
        self.route = None           # re-plan on the next update

    def _plan(self, p):
        g = self.goals[self.gi]
        r, c = plan_human_route(p, g, self.obstacles, self.pref_clearance, self.min_clearance)
        if r is None:
            self.log('error', f'no route to goal {self.gi + 1} ({g[0]:.2f}, {g[1]:.2f}): it is inside or '
                              f'walled in by an obstacle. Walking straight to it.')
            r, c = [g], self.min_clearance
        elif len(r) > 1:
            self.log('info', f'route to goal {self.gi + 1}: {len(r) - 1} corner(s) around obstacles, '
                             f'keeping {c:.2f} m')
        self.route, self.clearance = r, c

    def update(self, px, py):
        """Call every tick with the agent's position. Returns the (x, y) HuNav should walk to."""
        if not self.goals:
            return None
        p = (px, py)
        if self.route is None:
            self._plan(p)
        if not self.finished and math.dist(p, self.goals[self.gi]) <= self.reach:
            if self.gi + 1 < len(self.goals) or self.cyclic:
                if len(self.goals) > 1:
                    self.gi = (self.gi + 1) % len(self.goals)
                    self._plan(p)
            else:
                self.finished = True
        while len(self.route) > 1:
            if math.dist(p, self.route[0]) <= self.reach or \
                    segment_clear(p, self.route[1], self.obstacles, self.clearance,
                                  relax_a=True, relax_b=(len(self.route) == 2)):
                self.route.pop(0)
            else:
                break
        return self.route[0]

    def remaining_route(self):
        return list(self.route or [])