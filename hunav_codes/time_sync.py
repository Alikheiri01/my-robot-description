#!/usr/bin/env python3
"""
time_sync.py -- "where was it AT time t?" lookups, shared by every node that
combines depth-derived data with poses.

WHY THIS EXISTS (measured 2026-09-28 with latency_probe.py)
A depth-derived message describes the scene at its own header.stamp, and by
the time it reaches a consumer it is already old: the fused cloud and base
grid ~0.2-0.4 s of SIM time, the pedestrian crop ~0.4-0.8 s (at a real-time
factor of ~0.32, that is 1-2.5 s on screen). /odom and /people are only a few
milliseconds old. Every consumer used to combine a depth message with the
LATEST pose -- i.e. a pose from a different moment than the depth data:
  - build_occupancy_grid_dynamic.py drew the exclusion circle where the
    pedestrian is NOW, not where they were in the cloud -> they stayed in the
    grid as an obstacle;
  - the crop viewer / dataset recorder cut the crop at the latest position and
    heading from an older grid -> position/heading/scene from three moments.
The fix is to keep a short history of each pose stream and look the pose up
at the depth message's OWN timestamp. Any remaining pipeline delay then only
makes things display later; it can no longer pair data from different moments.

REQUIRES all streams on the SAME clock. Depth data is stamped with Gazebo's
simulation time; /people is too only if hunav_model_bridge.py runs with
use_sim_time:=true (the actor launch does). clocks_match() lets consumers
detect and loudly report a mismatch instead of silently failing every lookup.

WHY LINEAR INTERPOLATION IS THE RIGHT MODEL FOR /people
HuNav steps the pedestrian in straight Euler steps, and HuNavActorDriver
(interpolate mode) walks the Gazebo actor along the straight line between
consecutive poses, arriving at each one when /people publishes it. So the
body the cameras see at time t is the linear interpolation of the /people
samples around t, by their stamps.
"""
import bisect
import math

WALL_STAMP_THRESHOLD = 1e9  # stamps after 2001 are wall-clock, sim time starts near 0


def stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time -> float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def is_wall_clock(t: float) -> bool:
    return t > WALL_STAMP_THRESHOLD


def clocks_match(t_a: float, t_b: float) -> bool:
    """True if both stamps are simulation time or both are wall-clock time."""
    return is_wall_clock(t_a) == is_wall_clock(t_b)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class StampedPoseHistory:
    """
    Short history of (t, x, y, yaw) samples in arrival order, with lookup at
    any time t by linear interpolation (yaw along the shortest arc).

    Outside the stored range, at() HOLDS the nearest end sample for at most
    `max_hold` seconds and returns None beyond that -- it never extrapolates,
    so a lookup can be late but never invented.
    """

    def __init__(self, max_age_sec: float = 20.0):
        self.max_age = max_age_sec
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []

    def __len__(self):
        return len(self.t)

    def add(self, t: float, x: float, y: float, yaw: float = 0.0):
        if self.t and t <= self.t[-1]:
            if t == self.t[-1]:  # same instant republished: keep the newest values
                self.x[-1], self.y[-1], self.yaw[-1] = x, y, yaw
            elif self.t[-1] - t > 5.0:
                # clock went backwards a lot (e.g. simulation restarted): start over
                self.clear()
                self._append(t, x, y, yaw)
            return  # otherwise out of order: ignore
        self._append(t, x, y, yaw)
        # prune old samples
        cutoff = t - self.max_age
        n_old = bisect.bisect_left(self.t, cutoff)
        if n_old > 0:
            del self.t[:n_old], self.x[:n_old], self.y[:n_old], self.yaw[:n_old]

    def _append(self, t, x, y, yaw):
        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.yaw.append(yaw)

    def clear(self):
        self.t.clear()
        self.x.clear()
        self.y.clear()
        self.yaw.clear()

    def first_time(self):
        return self.t[0] if self.t else None

    def latest_time(self):
        return self.t[-1] if self.t else None

    def latest(self):
        if not self.t:
            return None
        return self.x[-1], self.y[-1], self.yaw[-1]

    def at(self, t: float, max_hold: float = 0.0):
        """(x, y, yaw) at time t, or None if t is outside the history by more than max_hold."""
        if not self.t:
            return None
        if t <= self.t[0]:
            return (self.x[0], self.y[0], self.yaw[0]) if self.t[0] - t <= max_hold else None
        if t >= self.t[-1]:
            return (self.x[-1], self.y[-1], self.yaw[-1]) if t - self.t[-1] <= max_hold else None
        i = bisect.bisect_right(self.t, t)  # self.t[i-1] <= t < self.t[i]
        t0, t1 = self.t[i - 1], self.t[i]
        a = (t - t0) / (t1 - t0)
        x = self.x[i - 1] + a * (self.x[i] - self.x[i - 1])
        y = self.y[i - 1] + a * (self.y[i] - self.y[i - 1])
        yaw = wrap_angle(self.yaw[i - 1] + a * wrap_angle(self.yaw[i] - self.yaw[i - 1]))
        return x, y, yaw

    def step_yaw_at(self, t: float, max_hold: float = 0.0):
        """
        Yaw of the straight segment that contains time t, or None if t is
        outside the history by more than max_hold: the yaw stored with the
        first sample stamped at or after t. This relies on the convention that
        a sample's yaw is the direction of the segment that ENDS at that sample
        (heading_smoother.py publishes it that way).

        NOT blended between samples, unlike at(): the pedestrian faces one way
        for a whole straight segment, and blending yaws across a reversal would
        sweep the crop through every angle in between while the body walks
        straight.
        """
        if not self.t:
            return None
        if t <= self.t[0]:
            return self.yaw[0] if self.t[0] - t <= max_hold else None
        if t > self.t[-1]:
            return self.yaw[-1] if t - self.t[-1] <= max_hold else None
        return self.yaw[bisect.bisect_left(self.t, t)]

    def max_gap(self, t_start: float, t_end: float) -> float:
        """Largest spacing between consecutive samples covering [t_start, t_end]
        (inf if the history does not cover it) -- to detect dropouts."""
        if not self.t or t_start < self.t[0] or t_end > self.t[-1]:
            return math.inf
        i0 = max(bisect.bisect_right(self.t, t_start) - 1, 0)
        i1 = min(bisect.bisect_left(self.t, t_end), len(self.t) - 1)
        if i1 <= i0:
            return 0.0
        return max(self.t[k + 1] - self.t[k] for k in range(i0, i1))


def _self_test():
    h = StampedPoseHistory(max_age_sec=10.0)
    h.add(10.0, 0.0, 0.0, math.radians(170))
    h.add(10.5, 1.0, 2.0, math.radians(-170))
    x, y, yaw = h.at(10.25)
    assert abs(x - 0.5) < 1e-9 and abs(y - 1.0) < 1e-9
    assert abs(abs(math.degrees(yaw)) - 180.0) < 1e-6, math.degrees(yaw)  # shortest arc, not 0
    assert h.at(9.9) is None and h.at(9.9, max_hold=0.2) == (0.0, 0.0, math.radians(170))
    assert h.at(10.6) is None and h.at(10.6, max_hold=0.2)[0] == 1.0
    # step lookup: the yaw of the sample that ENDS the segment containing t, no blending
    assert abs(h.step_yaw_at(10.25) - math.radians(-170)) < 1e-12
    assert abs(h.step_yaw_at(10.5) - math.radians(-170)) < 1e-12   # arrival instant: segment just ended
    assert abs(h.step_yaw_at(10.0) - math.radians(170)) < 1e-12
    assert h.step_yaw_at(9.9) is None and abs(h.step_yaw_at(9.9, 0.2) - math.radians(170)) < 1e-12
    assert h.step_yaw_at(10.6) is None and abs(h.step_yaw_at(10.6, 0.2) - math.radians(-170)) < 1e-12
    h.add(10.4, 9.0, 9.0)            # out of order: ignored
    assert h.at(10.45)[0] < 1.0
    for k in range(1, 40):
        h.add(10.5 + 0.5 * k, float(k), 0.0)
    assert h.first_time() >= h.latest_time() - 10.0 - 1e-9  # pruned
    assert abs(h.max_gap(h.first_time(), h.latest_time()) - 0.5) < 1e-9
    assert h.max_gap(0.0, 1.0) == math.inf
    assert clocks_match(12.3, 99.0) and not clocks_match(12.3, 1.79e9)
    print('time_sync self-test OK')


if __name__ == '__main__':
    _self_test()