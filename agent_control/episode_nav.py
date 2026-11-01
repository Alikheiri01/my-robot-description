#!/usr/bin/env python3
"""
episode_nav.py -- pure logic (no ROS) for the bridge's EPISODE MODE (2026-11-01).
hunav_model_bridge_nav.py imports it; it can be tested without a simulation.

  PathFollower   follows a dense path (E arcs) with a "carrot": HuNav's goal is always
                 a point LOOKAHEAD metres ahead along the path. HuNav holds position for
                 one tick whenever it REACHES a goal; the carrot is never reached, so the
                 pedestrian walks the curve at a steady speed. Only the last point (the
                 real goal) is ever reached.
  route_remaining  metres still to walk along the planner's route (A/B/C/D).
  stop_speed     scenario C, the designed slowdown: v(d) = v0 * sqrt(d / Ds) inside Ds
                 of the stop point (constant deceleration from v0 to 0 over Ds).
All in the WORLD frame (like the rest of the bridge).
"""
import math

STOP_MIN_SPEED = 0.1      # m/s: below this HuNav barely moves; the goal radius finishes the stop


class PathFollower:
    def __init__(self, path, lookahead):
        self.p = [tuple(map(float, q)) for q in path]
        self.L = lookahead
        self.cum = [0.0]
        for a, b in zip(self.p, self.p[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.length = self.cum[-1]
        self.i = 0            # segment index of the last projection (only moves forward)
        self.s = 0.0

    def _at(self, s):
        s = min(max(s, 0.0), self.length)
        for i in range(len(self.p) - 1):
            if self.cum[i + 1] >= s:
                seg = self.cum[i + 1] - self.cum[i]
                t = 0.0 if seg < 1e-9 else (s - self.cum[i]) / seg
                a, b = self.p[i], self.p[i + 1]
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return self.p[-1]

    def update(self, px, py):
        """Agent at (px, py) -> (carrot point, metres of path left)."""
        best, best_s, best_i = math.inf, self.s, self.i
        for i in range(max(0, self.i - 1), min(len(self.p) - 1, self.i + 40)):
            a, b = self.p[i], self.p[i + 1]
            abx, aby = b[0] - a[0], b[1] - a[1]
            L2 = abx * abx + aby * aby
            t = 0.0 if L2 < 1e-12 else min(max(((px - a[0]) * abx + (py - a[1]) * aby) / L2, 0.0), 1.0)
            d = math.hypot(a[0] + t * abx - px, a[1] + t * aby - py)
            if d < best:
                best, best_s, best_i = d, self.cum[i] + t * (self.cum[i + 1] - self.cum[i]), i
        if best_s >= self.s:                 # never go backwards along the path
            self.s, self.i = best_s, best_i
        carrot = self._at(self.s + self.L)
        return carrot, max(0.0, self.length - self.s)


def route_remaining(px, py, route):
    """route: remaining points of the RouteFollower (last = goal)."""
    if not route:
        return 0.0
    d = math.dist((px, py), route[0])
    for a, b in zip(route, route[1:]):
        d += math.dist(a, b)
    return d


def stop_speed(v0, remaining, slowdown_m):
    if slowdown_m <= 0 or remaining >= slowdown_m:
        return v0
    return max(STOP_MIN_SPEED, v0 * math.sqrt(max(remaining, 0.0) / slowdown_m))