#!/usr/bin/env python3
"""
plan_model.py -- the offline model used to PLAN episodes before Gazebo runs them
(2026-11-01). Pure numpy, no ROS.

It answers one question per episode: "with this route, this speed and this robot
behaviour, which share of the saved samples will be USABLE?" (usable = the
training filter: pedestrian seen at the anchor and in >= 6 of the 8 history steps).

The model, kept simple on purpose:
  * pedestrian: walks the planned route at constant speed (scenario C: the designed
    slowdown, then standing). t = 0 is the walk start, where the runner publishes the
    episode id; the episode ends ('end' id) END_TRIGGER_M before the goal (C: after
    standing STAND_SEC).
  * saved samples: anchors every 0.5 s from START_MARGIN_SEC after the walk start to
    END_MARGIN_SEC before the end (the recorder drops every window that touches an
    episode id change).
  * seen: the pedestrian is inside the camera wedge (73 deg, 8 m, minus planning
    margins) and the sight line from the camera is not blocked by any wall or obstacle.
  * robot: still (fixed pose), watch (fixed position, turns toward the pedestrian with
    a turn-rate limit), follow (drives along the pedestrian's route a few metres behind,
    with a speed limit).
The real numbers come from the recordings (record_session.py reports "usable X %");
this model only decides where to put the robot so the real number is high.
"""
import math

import numpy as np

import mvp_config as C


# ---------------------------------------------------------------------------
# polylines
# ---------------------------------------------------------------------------
class Polyline:
    def __init__(self, pts):
        self.p = np.asarray(pts, dtype=float)
        seg = np.diff(self.p, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])

    def at(self, s):
        """Points at arc lengths s (array). Beyond the ends: straight extension."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        out = np.empty((len(s), 2))
        d0 = (self.p[1] - self.p[0]) / max(self.seg_len[0], 1e-9)
        d1 = (self.p[-1] - self.p[-2]) / max(self.seg_len[-1], 1e-9)
        lo, hi = s < 0, s > self.length
        mid = ~(lo | hi)
        out[lo] = self.p[0] + np.outer(s[lo], d0)
        out[hi] = self.p[-1] + np.outer(s[hi] - self.length, d1)
        if mid.any():
            out[mid, 0] = np.interp(s[mid], self.s, self.p[:, 0])
            out[mid, 1] = np.interp(s[mid], self.s, self.p[:, 1])
        return out

    def project(self, q):
        """Arc length of the point of the polyline closest to q."""
        q = np.asarray(q, dtype=float)
        best, best_s = math.inf, 0.0
        for i in range(len(self.p) - 1):
            a, b = self.p[i], self.p[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-12 else min(max(float((q - a) @ ab) / L2, 0.0), 1.0)
            d = float(np.hypot(*(a + t * ab - q)))
            if d < best:
                best, best_s = d, self.s[i] + t * self.seg_len[i]
        return best_s, best

    def distance_to(self, q):
        return self.project(q)[1]


# ---------------------------------------------------------------------------
# pedestrian timeline
# ---------------------------------------------------------------------------
def ped_timeline(route, speed, stop=None, dt=C.PLAN_DT):
    """route: Polyline. stop: None or {'slowdown_m': Ds, 'stand_sec': S} (scenario C: the
    route ends at the stop point). Returns dict with t, s(t), xy(t), t_end."""
    L = route.length
    if stop is None:
        s_end = L - C.END_TRIGGER_M
        t_end = s_end / speed
        t = np.arange(0.0, t_end + 1e-9, dt)
        s = speed * t
    else:
        Ds = min(stop['slowdown_m'], L)
        t_walk = (L - Ds) / speed
        t_dec = 2.0 * Ds / speed                    # constant deceleration from speed to 0 over Ds
        t_end = t_walk + t_dec + stop['stand_sec']
        t = np.arange(0.0, t_end + 1e-9, dt)
        s = np.where(t <= t_walk, speed * t, 0.0)
        dec = (t > t_walk) & (t <= t_walk + t_dec)
        tau = t[dec] - t_walk
        s[dec] = (L - Ds) + speed * tau - 0.5 * (speed / t_dec) * tau ** 2
        s[t > t_walk + t_dec] = L
    return {'t': t, 's': s, 'xy': route.at(s), 't_end': float(t_end), 'dt': dt}


def time_at_s(tl, s_target):
    """First time the pedestrian reaches arc length s_target."""
    idx = np.searchsorted(tl['s'], s_target)
    idx = min(idx, len(tl['t']) - 1)
    return float(tl['t'][idx])


def kept_anchor_times(t_end):
    first = math.ceil(C.START_MARGIN_SEC / C.PERIOD_SEC) * C.PERIOD_SEC
    last = t_end - C.END_MARGIN_SEC
    if last < first:
        return np.zeros(0)
    return np.arange(first, last + 1e-9, C.PERIOD_SEC)


# ---------------------------------------------------------------------------
# robot timelines (heading and position for every model time step)
# ---------------------------------------------------------------------------
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def robot_still(tl, pose):
    n = len(tl['t'])
    return np.tile([pose[0], pose[1]], (n, 1)), np.full(n, pose[2])


def robot_watch(tl, pos, h0=None, dt=C.PLAN_DT):
    n = len(tl['t'])
    xy = np.tile([pos[0], pos[1]], (n, 1))
    h = np.empty(n)
    ped = tl['xy']
    cur = math.atan2(ped[0, 1] - pos[1], ped[0, 0] - pos[0]) if h0 is None else h0
    for i in range(n):
        want = math.atan2(ped[i, 1] - pos[1], ped[i, 0] - pos[0])
        err = (want - cur + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > C.WATCH_DEADBAND * 0.5:
            cur += max(-C.ROBOT_WMAX * dt, min(C.ROBOT_WMAX * dt, C.WATCH_GAIN * err * dt))
        h[i] = cur
    return xy, h


def robot_follow(tl, route, dist, dt=C.PLAN_DT):
    """Robot on the pedestrian's route (extended straight behind the start), dist metres behind."""
    n = len(tl['t'])
    s_r = np.empty(n)
    cur = -dist
    for i in range(n):
        err = (tl['s'][i] - dist) - cur
        v = max(0.0, min(C.ROBOT_VMAX, C.FOLLOW_GAIN * err))
        cur += v * dt
        s_r[i] = cur
    xy = route.at(s_r)
    ahead = route.at(s_r + C.FOLLOW_LOOKAHEAD_M)
    h = np.arctan2(ahead[:, 1] - xy[:, 1], ahead[:, 0] - xy[:, 0])
    return xy, h


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------
def _segments_hit_box(a, b, o, pad=0.0):
    """Vectorised: does segment a[i]->b[i] cross box obstacle o? (slab test in box frame)"""
    c, s = o.c, o.s
    ax = c * (a[:, 0] - o.x) + s * (a[:, 1] - o.y)
    ay = -s * (a[:, 0] - o.x) + c * (a[:, 1] - o.y)
    bx = c * (b[:, 0] - o.x) + s * (b[:, 1] - o.y)
    by = -s * (b[:, 0] - o.x) + c * (b[:, 1] - o.y)
    dx, dy = bx - ax, by - ay
    t0, t1 = np.zeros(len(a)), np.ones(len(a))
    hit = np.ones(len(a), dtype=bool)
    for p, d, h in ((ax, dx, o.hx + pad), (ay, dy, o.hy + pad)):
        par = np.abs(d) < 1e-12
        hit &= ~(par & ((p < -h) | (p > h)))
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (-h - p) / d
            tb = (h - p) / d
        lo = np.where(par, -np.inf, np.minimum(ta, tb))
        hi = np.where(par, np.inf, np.maximum(ta, tb))
        t0 = np.maximum(t0, lo)
        t1 = np.minimum(t1, hi)
    return hit & (t0 <= t1)


def _segments_hit_circle(a, b, o, pad=0.0):
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((o.x - a[:, 0]) * ab[:, 0] + (o.y - a[:, 1]) * ab[:, 1]) / L2, 0, 1)
    px, py = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    return np.hypot(px - o.x, py - o.y) < o.r + pad


def sight_blocked(a, b, obstacles):
    blocked = np.zeros(len(a), dtype=bool)
    for o in obstacles:
        blocked |= _segments_hit_box(a, b, o) if o.kind == 'box' else _segments_hit_circle(a, b, o)
    return blocked


def seen_series(ped_xy, robot_xy, robot_h, obstacles):
    """1/0 per model step: is the pedestrian seen by the camera (planning margins)?"""
    cam = robot_xy + C.CAM_OFFSET_M * np.stack([np.cos(robot_h), np.sin(robot_h)], 1)
    v = ped_xy - cam
    d = np.hypot(v[:, 0], v[:, 1])
    ang = np.abs(wrap(np.arctan2(v[:, 1], v[:, 0]) - robot_h))
    ok = (d <= C.CAM_RANGE_M - C.PLAN_RANGE_MARGIN_M) & (ang <= C.CAM_HALF_FOV - C.PLAN_FOV_MARGIN)
    # sight line ends PED_BODY_M before the pedestrian centre
    shrink = np.clip((d - C.PED_BODY_M) / np.maximum(d, 1e-9), 0, 1)[:, None]
    end = cam + v * shrink
    idx = np.nonzero(ok)[0]
    if len(idx):
        ok[idx] &= ~sight_blocked(cam[idx], end[idx], obstacles)
    return ok


def usable_stats(tl, seen):
    """Planned (n_kept, n_usable, share) with the training filter."""
    T = kept_anchor_times(tl['t_end'])
    if len(T) == 0:
        return 0, 0, 0.0
    dt = tl['dt']

    def at(t):
        return seen[np.clip(np.round(t / dt).astype(int), 0, len(seen) - 1)]

    hist = np.stack([at(T - k * C.PERIOD_SEC) for k in range(C.HISTORY_STEPS)], 1)   # k=0 is the anchor
    usable = hist[:, 0] & (hist.sum(1) >= C.USABLE_MIN_HISTORY_SEEN)
    return len(T), int(usable.sum()), float(usable.mean())


def event_usable_stats(tl, seen, t_event):
    """NEW (2026-11-03): like usable_stats, but only for the anchors around the event
    (t_event - EVENT_WINDOW_SEC[0] <= anchor <= t_event + EVENT_WINDOW_SEC[1]): the moments just
    before the pedestrian goes through the door / round the corner / past the obstacle.
    Returns (n_anchors, n_usable, share)."""
    T = kept_anchor_times(tl['t_end'])
    T = T[(T >= t_event - C.EVENT_WINDOW_SEC[0] - 1e-9) & (T <= t_event + C.EVENT_WINDOW_SEC[1] + 1e-9)]
    if len(T) == 0:
        return 0, 0, 0.0
    dt = tl['dt']

    def at(t):
        return seen[np.clip(np.round(t / dt).astype(int), 0, len(seen) - 1)]

    hist = np.stack([at(T - k * C.PERIOD_SEC) for k in range(C.HISTORY_STEPS)], 1)
    usable = hist[:, 0] & (hist.sum(1) >= C.USABLE_MIN_HISTORY_SEEN)
    return len(T), int(usable.sum()), float(usable.mean())


def evaluate(tl, route, robot, obstacles):
    """robot: {'mode': 'still'|'watch'|'follow', 'pose': (x,y,h)} or follow {'dist': d}.
    Returns (robot_xy, robot_h, seen, (n_kept, n_usable, share))."""
    if robot['mode'] == 'still':
        rxy, rh = robot_still(tl, robot['pose'])
    elif robot['mode'] == 'watch':
        rxy, rh = robot_watch(tl, robot['pose'][:2], robot['pose'][2])
    else:
        rxy, rh = robot_follow(tl, route, robot['dist'])
    seen = seen_series(tl['xy'], rxy, rh, obstacles)
    return rxy, rh, seen, usable_stats(tl, seen)