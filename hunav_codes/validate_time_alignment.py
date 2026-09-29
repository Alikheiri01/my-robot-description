#!/usr/bin/env python3
"""
validate_time_alignment.py -- measures, with numbers, whether the pedestrian
pose we look up for a depth cloud is where the CAMERA actually sees the
pedestrian in that cloud.

WHY THIS EXISTS
Looking at RViz / the crop viewer cannot answer "is it aligned?": the display
lags by seconds, and a 0.2 m error is invisible by eye. The camera data
itself is the independent ground truth. For every fused depth cloud this
script
  1. finds the pedestrian's points in the cloud (mean-shift on the points
     around the predicted position -- no other assumption),
  2. takes their centre (the "camera position"),
  3. compares it with what each lookup method predicts for the cloud's own
     timestamp, and reports the error in metres.

METHODS COMPARED (all use only data that was available when the cloud arrived)
  A  latest /people pose               -- the OLD behaviour
  B  /people interpolated at the cloud's stamp + POSE_LOOKUP_OFFSET_SEC
     (hunav_config.py); if that time is newer than the newest /people sample,
     hold that sample   -- what build_occupancy_grid_dynamic.py /
     pedestrian_crop_view_dynamic.py do now
  C  like B, but beyond the newest sample, extrapolate with /people's velocity

The report's "timing offset" is what is STILL missing after B's/C's own
offset; the verdict prints the POSE_LOOKUP_OFFSET_SEC to use if it is not ~0.
Run it again after changing that constant to confirm.

WHAT IT REPORTS
  - error of each method (median / 90th percentile), split into "cloud inside
    the /people range" vs "cloud newer than the newest /people sample";
  - a TIMING OFFSET per method: how many seconds to add to the cloud's stamp
    before looking the pose up so it matches the camera (0 = perfect). Found
    by regression that separates timing error (flips sign with walking
    direction) from the constant "camera only sees the near surface of the
    body" shift k (always points toward the robot);
  - what share of the pedestrian's points the exclusion circle would miss;
  - how big the pedestrian really is (radius needed for the exclusion);
  - how well the smoothed heading matches the instantaneous heading.

USAGE (simulation running, pedestrian walking in view of the cameras)
    python3 validate_time_alignment.py
Let it run 2-3 minutes so it sees several turnarounds, then Ctrl+C (NOT
Ctrl+Z) for the report. Remove other objects near the pedestrian's path first
(clouds where another object is within ~1 m of the pedestrian are skipped and
counted, but too many skipped clouds leaves too little data).
"""
import bisect
import collections
import math
import sys

import numpy as np

from time_sync import StampedPoseHistory, clocks_match, wrap_angle

# --- tuning of the measurement itself (not of the system under test) ----------
GATE_R = 0.6            # mean-shift radius: covers a body incl. swinging arms
BODY_R = 0.7            # points within this of the centre count as "the pedestrian"
RING_LO, RING_HI = 0.8, 1.4   # annulus checked for OTHER objects near the pedestrian
ISOLATION_MAX = 0.35    # ring points / body points above this -> not isolated, skip
MIN_POINTS, MAX_POINTS = 25, 2500
MS_ITERS = 8
MS_CONVERGED = 0.02     # m
HOLD_MAX = 0.6          # s, same as the consumers under test
EXTRAP_MAX = 0.6        # s
MIN_SPEED_FOR_FIT = 0.2  # m/s; a standing pedestrian says nothing about timing
MAX_COND = 30.0         # regression needs walking in both directions
MIN_USED = 15
PASS_P90_M = 0.15       # about one grid cell (0.1 m) plus margin
PASS_OFFSET_S = 0.10


def _seg_velocity(hist, t):
    """Velocity of the /people segment containing t (last segment beyond the end)."""
    n = len(hist.t)
    if n < 2:
        return 0.0, 0.0
    i = min(max(bisect.bisect_right(hist.t, t) - 1, 0), n - 2)
    dt = hist.t[i + 1] - hist.t[i]
    if dt <= 1e-9:
        return 0.0, 0.0
    return (hist.x[i + 1] - hist.x[i]) / dt, (hist.y[i + 1] - hist.y[i]) / dt


def _fit(R, H, V):
    """Least squares  R_i ~= -k*H_i + delta*V_i  ->  (k, delta, rms, cond)."""
    n = len(R)
    A = np.zeros((2 * n, 2))
    A[0::2, 0], A[1::2, 0] = -H[:, 0], -H[:, 1]
    A[0::2, 1], A[1::2, 1] = V[:, 0], V[:, 1]
    b = R.reshape(-1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    res = b - A @ sol
    return float(sol[0]), float(sol[1]), float(math.sqrt(np.mean(res ** 2) * 2)), float(np.linalg.cond(A))


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float('nan')


class AlignmentAnalyzer:
    """Pure logic (no ROS): feed events in ARRIVAL order, then call report()."""

    def __init__(self, exclusion_radius, spawn_yaw=0.0, lookup_offset=0.0):
        self.excl_r = exclusion_radius
        self.spawn_yaw = spawn_yaw
        self.lookup_offset = lookup_offset   # POSE_LOOKUP_OFFSET_SEC the nodes under test apply (B and C)
        self.odom = StampedPoseHistory(max_age_sec=1e9)
        self.ppl = StampedPoseHistory(max_age_sec=1e9)
        self.vel = {}                 # round(stamp, 3) -> (vx, vy)
        self.last_vel = (0.0, 0.0)
        self.records = []
        self.rejects = collections.Counter()
        self.n_clouds = 0
        self.heading_err = []         # (abs error deg, speed)

    # --- events ------------------------------------------------------------
    def on_odom(self, t, x, y, yaw):
        self.odom.add(t, x, y, yaw)

    def on_people(self, t, x, y, vx, vy):
        self.ppl.add(t, x, y, 0.0)
        self.vel[round(t, 3)] = (vx, vy)
        self.last_vel = (vx, vy)

    def on_smoothed(self, t, yaw_odom_frame):
        v = self.vel.get(round(t, 3))
        if v is None or math.hypot(*v) < 0.3:
            return
        raw = math.atan2(v[1], v[0])
        smooth_world = yaw_odom_frame + self.spawn_yaw
        self.heading_err.append((abs(math.degrees(wrap_angle(smooth_world - raw))), math.hypot(*v)))

    def on_cloud(self, t_c, pts_b):
        """pts_b: Nx2 obstacle points in base_link (ground already removed)."""
        self.n_clouds += 1
        robot = self.odom.at(t_c, max_hold=0.2)
        if robot is None:
            self.rejects['no robot pose at cloud time'] += 1
            return
        if len(self.ppl) == 0:
            self.rejects['no /people yet'] += 1
            return
        last_t = self.ppl.latest_time()
        if not clocks_match(t_c, last_t):
            self.rejects['CLOCK MISMATCH cloud vs /people'] += 1
            return
        t_l = t_c + self.lookup_offset      # the time the nodes under test look the pedestrian up at
        pB_full = self.ppl.at(t_l, max_hold=HOLD_MAX)
        if pB_full is None:
            self.rejects['no /people pose near cloud time'] += 1
            return
        pB = np.array(pB_full[:2])
        hold = t_l > last_t
        pA = np.array(self.ppl.latest()[:2])
        pC = pB + np.array(self.last_vel) * min(t_l - last_t, EXTRAP_MAX) if hold else pB

        rx, ry, ryaw = robot
        c, s = math.cos(ryaw), math.sin(ryaw)
        xw = rx + c * pts_b[:, 0] - s * pts_b[:, 1]
        yw = ry + s * pts_b[:, 0] + c * pts_b[:, 1]

        ctr = pB.copy()
        shift = 1.0
        for _ in range(MS_ITERS):
            m = np.hypot(xw - ctr[0], yw - ctr[1]) <= GATE_R
            if np.count_nonzero(m) < MIN_POINTS:
                break
            new = np.array([xw[m].mean(), yw[m].mean()])
            shift = float(np.hypot(*(new - ctr)))
            ctr = new
            if shift < MS_CONVERGED:
                break
        d = np.hypot(xw - ctr[0], yw - ctr[1])
        n_gate = int(np.count_nonzero(d <= GATE_R))
        if n_gate < MIN_POINTS:
            self.rejects[f'pedestrian not visible (<{MIN_POINTS} points)'] += 1
            return
        if n_gate > MAX_POINTS:
            self.rejects['too many points near pedestrian (large object?)'] += 1
            return
        if shift >= MS_CONVERGED:
            self.rejects['centre search did not converge'] += 1
            return
        n_ring = int(np.count_nonzero((d > RING_LO) & (d <= RING_HI)))
        if n_ring > ISOLATION_MAX * n_gate:
            self.rejects['another object within ~1 m of the pedestrian'] += 1
            return

        body = d <= BODY_R
        rec = {'t': t_c, 'tl': t_l, 'c': ctr, 'robot': np.array([rx, ry]), 'hold': hold, 'n': int(body.sum()),
               'extent90': _pct(d[body], 90), 'extent_max': float(d[body].max())}
        for key, p in (('A', pA), ('B', pB), ('C', pC)):
            rec['p' + key] = p
            dist = np.hypot(xw[body] - p[0], yw[body] - p[1])
            rec['left' + key] = float(np.mean(dist > self.excl_r))
        self.records.append(rec)

    # --- report ------------------------------------------------------------
    def report(self):
        L = []
        w = L.append
        w('=' * 100)
        w('TIME-ALIGNMENT VALIDATION')
        used = len(self.records)
        w(f'Clouds received: {self.n_clouds}    used: {used}    /people samples: {len(self.ppl)}')
        for reason, cnt in self.rejects.most_common():
            w(f'   skipped {cnt:4d}: {reason}')
        if self.rejects.get('CLOCK MISMATCH cloud vs /people'):
            w('\n  !! Cloud stamps and /people stamps are on different clocks. Run '
              'hunav_model_bridge.py with use_sim_time:=true.')
        if used < MIN_USED:
            w(f'\nToo few usable clouds ({used} < {MIN_USED}) for statistics. Keep the pedestrian '
              f'in the cameras\' view, remove nearby objects and run longer.')
            w('=' * 100)
            return '\n'.join(L), {}

        c = np.array([r['c'] for r in self.records])
        robot = np.array([r['robot'] for r in self.records])
        tl = np.array([r['tl'] for r in self.records])
        hold = np.array([r['hold'] for r in self.records])
        v = np.array([_seg_velocity(self.ppl, ti) for ti in tl])
        NW = 58   # width of the method-name column
        off_txt = f'{self.lookup_offset:+.2f} s'
        speed = np.hypot(v[:, 0], v[:, 1])
        H = c - robot
        H = H / np.maximum(np.hypot(H[:, 0], H[:, 1]), 1e-6)[:, None]
        moving = speed >= MIN_SPEED_FOR_FIT

        def resid(key):
            return c - np.array([r['p' + key] for r in self.records])

        def fit_subset(key, mask):
            m = mask & moving
            if np.count_nonzero(m) < 8:
                return None
            k, dl, rms, cond = _fit(resid(key)[m], H[m], v[m])
            return None if cond > MAX_COND else (k, dl, rms, cond)

        # constant near-surface shift, from the cleanest data (B, cloud inside the range)
        ref = fit_subset('B', ~hold) or fit_subset('B', np.ones(used, bool))
        k_ref = ref[0] if ref else 0.0
        w(f'\nCamera-vs-body shift k = {k_ref:.2f} m (the depth camera only sees the near side of the '
          f'body, so its centre sits this far toward the robot; removed from the errors below)')

        w(f'\nLookup offset applied to B and C: {off_txt} (POSE_LOOKUP_OFFSET_SEC in hunav_config.py); '
          f'A ignores it.')
        w('\nPOSITION ERROR: where the camera sees the pedestrian vs the looked-up position')
        w('  (timing offset = seconds still to ADD to that method\'s lookup time; 0 is perfect)')
        w(f'  {"method":{NW}s} {"clouds":>6s} {"median":>8s} {"p90":>8s} {"timing offset":>14s}')
        names = [('A', 'A  latest /people pose (old behaviour)'),
                 ('B', f'B  pose at cloud stamp {off_txt}, hold at end (current)'),
                 ('C', f'C  pose at cloud stamp {off_txt}, extrapolate at end')]
        best = {}
        for key, name in names:
            e = np.hypot(*(resid(key) + k_ref * H).T)
            f = fit_subset(key, np.ones(used, bool))
            off = 'n/a' if key == 'A' or not f else f'{f[1]:+.2f} s'
            w(f'  {name:{NW}s} {used:6d} {_pct(e, 50):8.3f} {_pct(e, 90):8.3f} {off:>14s}')
            best[key] = (_pct(e, 90), f[1] if f else float('nan'))
            if key in ('B', 'C'):
                for gname, mask in (('cloud inside the /people range', ~hold),
                                    ('cloud newer than newest /people', hold)):
                    if np.count_nonzero(mask) >= 5:
                        fg = fit_subset(key, mask)
                        offg = f'{fg[1]:+.2f} s' if fg else 'n/a'
                        w(f'     {gname:{NW - 3}s} {np.count_nonzero(mask):6d} {_pct(e[mask], 50):8.3f} '
                          f'{_pct(e[mask], 90):8.3f} {offg:>14s}')
                        best[key + ('h' if mask is hold else 'i')] = _pct(e[mask], 90)

        w(f'\nEXCLUSION (radius {self.excl_r:.2f} m): share of the pedestrian\'s camera points NOT removed')
        w(f'  {"method":{NW}s} {"median":>8s} {"p90":>8s} {"clouds >15% left":>18s}')
        for key, name in names:
            left = np.array([r['left' + key] for r in self.records])
            w(f'  {name:{NW}s} {100 * _pct(left, 50):7.0f}% {100 * _pct(left, 90):7.0f}% '
              f'{100 * np.mean(left > 0.15):17.0f}%')
        ext = np.array([r['extent90'] for r in self.records])
        extmax = np.array([r['extent_max'] for r in self.records])
        w(f'\nPEDESTRIAN SIZE: 90% of its points lie within {_pct(ext, 50):.2f} m (median) / '
          f'{_pct(ext, 90):.2f} m (p90 over clouds) of its camera centre; farthest point '
          f'{_pct(extmax, 50):.2f} m median.')
        w(f'  -> exclusion radius needed ~ (size + k + alignment error) = '
          f'{_pct(ext, 90):.2f} + {k_ref:.2f} + {min(best["B"][0], best["C"][0]):.2f} = '
          f'{_pct(ext, 90) + k_ref + min(best["B"][0], best["C"][0]):.2f} m   (configured: {self.excl_r:.2f} m)')

        if self.heading_err:
            he = np.array([h[0] for h in self.heading_err])
            w(f'\nHEADING: smoothed heading vs instantaneous heading in the same /people message '
              f'({len(he)} samples while walking)')
            w(f'  median error {np.median(he):.0f} deg,  more than 45 deg off: {100 * np.mean(he > 45):.0f}%,  '
              f'more than 120 deg off: {100 * np.mean(he > 120):.0f}%')
            w('  (large errors right after each turnaround = the 3-sample smoothing still averaging '
              'the old direction)')

        w('\nVERDICT')
        p90_b, p90_c = best['B'][0], best['C'][0]
        f_in = fit_subset('B', ~hold)
        off_in = f_in[1] if f_in else float('nan')
        cur_ok = p90_b <= PASS_P90_M and (math.isnan(off_in) or abs(off_in) <= PASS_OFFSET_S)
        if cur_ok:
            w(f'  PASS: the current lookup (B) is within {p90_b:.2f} m of the camera for 90% of clouds '
              f'(limit {PASS_P90_M} m).')
            if not math.isnan(off_in):
                w(f'        Residual timing offset (clouds inside the /people range): {off_in:+.2f} s '
                  f'(limit +-{PASS_OFFSET_S} s). Alignment is sound.')
        else:
            w(f'  NOT YET: method B\'s 90th-percentile error is {p90_b:.2f} m (limit {PASS_P90_M} m).')
            hb = best.get('Bh')
            ib = best.get('Bi')
            if hb is not None and ib is not None and hb > ib + 0.05:
                w(f'   - clouds newer than the newest /people sample are the problem (p90 {hb:.2f} m vs '
                  f'{ib:.2f} m inside the range): holding the last pose is not enough.')
                if p90_c < p90_b - 0.03:
                    w(f'     Extrapolating with the velocity (C) improves it to {p90_c:.2f} m -> adopt C.')
                else:
                    w('     Extrapolation does not help either -> publish the NEXT pose early '
                      '(stamped with the time the actor will reach it) and interpolate to it.')
            if not math.isnan(off_in) and abs(off_in) > PASS_OFFSET_S:
                w(f'   - constant timing offset {off_in:+.2f} s remains (clouds inside the /people range): '
                  f'set POSE_LOOKUP_OFFSET_SEC = {self.lookup_offset + off_in:+.2f} in hunav_config.py '
                  f'(currently {self.lookup_offset:+.2f}), restart the nodes and run this again.')
        # exclusion radius: independent of alignment, reported whenever it matters
        left_b = np.array([r['leftB'] for r in self.records])
        share = float(np.mean(left_b > 0.15))
        need = _pct(ext, 90) + k_ref + min(best['B'][0], best['C'][0])
        if share > 0.05:
            w(f'  EXCLUSION: {100 * share:.0f}% of clouds still leave >15% of the pedestrian\'s points '
              f'(radius needed ~{need:.2f} m, configured {self.excl_r:.2f} m)'
              + (' -> raise PEDESTRIAN_EXCLUSION_RADIUS in hunav_config.py.' if need > self.excl_r + 0.02
                 else '; the radius is not the cause.'))
        else:
            w(f'  EXCLUSION: only {100 * share:.0f}% of clouds leave >15% of the pedestrian\'s points '
              f'-- radius {self.excl_r:.2f} m is adequate.')
        w('=' * 100)
        return '\n'.join(L), {'best': best, 'k': k_ref, 'used': used}


# =============================================================================
# ROS wiring
# =============================================================================
def run_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    import sensor_msgs_py.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseArray
    from people_msgs.msg import People
    from hunav_config import (odom_to_world, yaw_from_quaternion, PEDESTRIAN_EXCLUSION_RADIUS,
                              SPAWN_YAW, POSE_LOOKUP_OFFSET_SEC)
    from time_sync import stamp_to_sec

    class Validator(Node):
        def __init__(self):
            super().__init__('validate_time_alignment')
            self.an = AlignmentAnalyzer(PEDESTRIAN_EXCLUSION_RADIUS, SPAWN_YAW, POSE_LOOKUP_OFFSET_SEC)
            self.create_subscription(Odometry, '/odom', self._odom, qos_profile_sensor_data)
            self.create_subscription(People, '/people', self._people, 10)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._smoothed, 10)
            self.create_subscription(PointCloud2, '/depth_cam/fused/points', self._cloud, qos_profile_sensor_data)
            self.create_timer(10.0, self._progress)
            print('validate_time_alignment: collecting... (Ctrl+C for the report, NOT Ctrl+Z)', flush=True)

        def _odom(self, msg):
            p = msg.pose.pose.position
            x, y, yaw = odom_to_world(p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation))
            self.an.on_odom(stamp_to_sec(msg.header.stamp), x, y, yaw)

        def _people(self, msg):
            if msg.people:
                p = msg.people[0]
                self.an.on_people(stamp_to_sec(msg.header.stamp), p.position.x, p.position.y,
                                  p.velocity.x, p.velocity.y)

        def _smoothed(self, msg):
            if msg.poses:
                o = msg.poses[0].orientation
                self.an.on_smoothed(stamp_to_sec(msg.header.stamp), 2.0 * math.atan2(o.z, o.w))

        def _cloud(self, msg):
            s = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if s.shape[0] == 0:
                pts = np.zeros((0, 2))
            else:
                keep = np.asarray(s['z']) > 0.0   # above the floor band
                pts = np.column_stack([np.asarray(s['x'])[keep], np.asarray(s['y'])[keep]]).astype(np.float64)
            self.an.on_cloud(stamp_to_sec(msg.header.stamp), pts)

        def _progress(self):
            a = self.an
            top = a.rejects.most_common(1)
            print(f'  ... {a.n_clouds} clouds, {len(a.records)} usable'
                  + (f', most skipped: {top[0][1]}x "{top[0][0]}"' if top else ''), flush=True)

    rclpy.init()
    node = Validator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(node.an.report()[0], flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    run_node()