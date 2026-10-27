#!/usr/bin/env python3
"""
Phase D dataset recorder -- assembles (trajectory_history, crop, future_target)
training samples from the live pipeline and writes them out as .npz files.

REWRITTEN 2026-09-30: TIME-ALIGNED, SIMULATION-CLOCK, EVENT-DRIVEN
------------------------------------------------------------------
The first version sampled on a WALL-clock timer using whatever the latest
/odom, /people_smoothed_pose and /occupancy_grid/base messages happened to be.
Two things were wrong with that once the simulation ran slower than real time
(real-time factor ~0.32) and the depth pipeline was measured to trail the
pose streams:
  1. A 0.5 s WALL period is only ~0.16 s of SIMULATION time, so consecutive
     history/future rows were ~0.16 s apart instead of 0.5 s (the "8 steps =
     4 s of history" in hunav_config.py was really ~1.3 s), and the same
     pedestrian pose was recorded about three times over.
  2. The crop was cut from the newest grid at the newest pedestrian pose --
     a pose from a different moment than the depth data in that grid
     (validate_time_alignment.py: ~0.26 m error at walking speed).
Now everything is keyed by SIMULATION timestamps:

  * ANCHOR TIME. For a depth-derived grid stamped t_g, the pedestrian the
    cameras see is where the pose stream put them at  T = t_g + POSE_LOOKUP_OFFSET_SEC.
    (2026-10-01: that stream is now the plugin's APPLIED pose, stamped with the
    simulation time of the step the actor was moved in, relayed by
    applied_pose_relay.py as /people_smoothed_pose, so the offset is 0.0. Until
    then it was HuNav's /people, the command, with a measured offset of -0.10 ..
    -0.29 s that changed between sessions. Samples carry lookup_offset_sec.) The
    sample's anchor is that T, and its crop is cut from THAT grid at the
    pedestrian pose at T, with the heading of the segment containing T
    (time_sync.StampedPoseHistory.step_yaw_at).
  * SAMPLE LATTICE. Anchors are spaced DATASET_SAMPLE_PERIOD_SEC of simulation
    time apart (0.5 s = one HuNav step): for each target time the grid whose T
    is nearest is used, so the crop is at most half a grid interval (~0.14 s)
    from the ideal lattice time and is always centred exactly on the body the
    camera saw.
  * HISTORY / FUTURE. Read from the /people_smoothed_pose history at exactly
    T - (H-1-i)*period and T + (k+1)*period (linear position interpolation,
    heading of the containing segment) -- true 0.5 s spacing whatever the
    real-time factor, and no duplicated poses. Future rows need pose data up
    to T + F*period, so a sample is written that much simulation time after
    its anchor.
  * NO GUESSING. The pedestrian pose is only ever INTERPOLATED (never held or
    extrapolated). A sample is dropped, and counted by reason, if the pose
    history does not cover its window, has a dropout longer than
    MAX_POSE_GAP_SEC inside it, if no grid lies near the target time, if the
    robot pose is missing, or if the clocks of the streams do not match.

ONE CONSISTENT LOCAL FRAME PER SAMPLE
------------------------------------------------------------------
Unchanged: history and future rows are expressed relative to the ANCHOR
position + heading (extract_pedestrian_crop.base_to_local), so the last
history row is exactly (0, 0, 0). All poses (/odom, /people_smoothed_pose)
share the 'odom' frame after heading_smoother.py's own frame fix.

Sample layout (each is one .npz file):
    trajectory_history : float32 (TRAJECTORY_HISTORY_LENGTH, 3)
                          [local_x, local_y, local_heading] per step,
                          oldest first, LAST row is the anchor step itself
                          (local_x=local_y=local_heading=0 by construction)
    crop                : int8 (80, 80) -- the anchor's own crop,
                          same -1/0/100 convention as the base grid
    future_target       : float32 (FUTURE_HORIZON_LENGTH, 3) -- same
                          convention, the TRUE continuation after the anchor
                          (from the /people pose stream, not the lagged depth)
    agent_id            : str, currently always 'agent1' (single-agent)
    sample_period_sec   : float, DATASET_SAMPLE_PERIOD_SEC at record time
    wall_time           : float, time.time() when the sample was emitted
  NEW (for checking a recording frame by frame):
    schema              : 'time_aligned_v2'
    anchor_time_sim     : float64, T (simulation seconds)
    grid_stamp_sim      : float64, t_g of the grid the crop was cut from
    lookup_offset_sec   : float, POSE_LOOKUP_OFFSET_SEC used (T = t_g + this)
    history_times       : float64 (H,) simulation time of each history row
    future_times        : float64 (F,) simulation time of each future row
    robot_pose_odom     : float64 (3,) robot x, y, yaw (odom frame) at t_g
    ped_anchor_odom     : float64 (3,) pedestrian x, y, yaw (odom) at T
    max_pose_gap_sec    : float, largest spacing of pose samples in the window
  NEW 2026-10-24 (could the ROBOT see the pedestrian? see analyze_ped_visibility.py):
    ped_seen_anchor     : int8, 1 = the pedestrian's spot in this sample's crop is
                          known (camera saw it), 0 = grey (out of view / hidden)
    ped_seen_history    : int8 (H,) the same for every history row, judged on the
                          grid nearest that time; -1 = no grid within
                          MAX_GRID_MISMATCH_SEC. Last entry = ped_seen_anchor.
    ped_seen_future     : int8 (F,) the same for the future rows (information only;
                          the last ones can be -1 if their grids had not arrived)
  Nothing is dropped for visibility: training decides (proposed: anchor seen and
  >= 6 of 8 history steps seen).
    gated               : bool, True = the start gate (collection_gate.py) was
                          heard and the whole window was open (2026-10-24)
  NEW 2026-10-27 (robot track + episodes; the schema string stays 'time_aligned_v2',
  fields are only added, nothing old changed):
    robot_history_odom  : float64 (H, 3) robot x, y, yaw (odom frame) at each history time
                          (history_times). The last row is the robot at T
                          (robot_pose_odom is the robot at the grid stamp t_g, ~0.05 s earlier).
    robot_future_odom   : float64 (F, 3) robot x, y, yaw at each future time (future_times).
                          NOT a network input (the robot's future is not known when
                          predicting); kept for analysis: did the pedestrian react to the
                          robot, how close did they get, closed-loop evaluation later.
                          A sample is never saved with a missing robot pose (no NaN): it is
                          dropped instead ('robot pose dropout ...'), like the pedestrian.
    episode_id          : str, the episode the anchor T belongs to ('' = no episode runner).
                          Split train/test by it.
    episode_time_sec    : float, T minus the episode's start time (-1.0 if no episode).

EPISODES (2026-10-27)
------------------------------------------------------------------
An episode runner (written later) publishes the current episode id as a plain
std_msgs/String on /dataset/episode (it may repeat it, e.g. once a second; only a
CHANGE counts). The recorder notes the simulation time of each change (the newest
pedestrian pose time when the message arrives) and
  * labels each sample with the episode its anchor T falls in;
  * DROPS a sample whose window (first history row .. last future row, +-EPISODE_MARGIN_SEC)
    contains an episode change: such a window mixes two episodes (reset, new layout).
Everything else about an episode (scenario type, layout, parameters, event time) lives
in one metadata file per episode written by the runner. Without a runner nothing
changes: episode_id is '' and nothing is dropped for episodes.
You can test it by hand:  ros2 topic pub --once /dataset/episode std_msgs/String "data: 'ep_001'"

RUN INFO (2026-10-27)
------------------------------------------------------------------
At start-up the recorder writes recorder_info.json next to the samples folder: every
constant it uses (this file, hunav_config.py, build_occupancy_grid.py,
extract_pedestrian_crop.py). A dataset then always says how it was made (crop size,
resolution, offsets, period ...), which the parameter study needs.

START GATE (2026-10-24)
------------------------------------------------------------------
collection_gate.py publishes on /dataset/gate the CLOSED time intervals (not
started yet, teleport, obstacles changed, simulation restart). A sample is saved
only if its whole window (first history row .. last future row) touches none of
them; otherwise it is dropped and counted as 'start gate: <reason>'. The recorder
waits until the gate has reported GATE_LAG_SEC past the window's end (a scene
change is marked a little into the past). If no gate is heard
within GATE_WAIT_SEC of the first pedestrian pose, samples are saved UNGATED
(gated=False, the status line and record_session.py say so). If the gate goes
silent later, windows it cannot judge are dropped ('gate silent').

REQUIRED: the full pipeline must be running (robot + depth pipeline +
occupancy grid + HuNav driving the pedestrian + heading_smoother.py), with
hunav_model_bridge.py on the simulation clock (use_sim_time:=true). This node
is purely a consumer of already-published topics. It does not need
use_sim_time itself: it never reads its own clock for sampling, only message
stamps (a wall-clock timer is used only for the status line).
"""
import math
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

# hunav_config.py, time_sync.py and the scene-codes modules (build_occupancy_grid,
# extract_pedestrian_crop) live in sibling directories -- see
# pedestrian_crop_view_dynamic.py's docstring for why they are not copied here.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))
sys.path.insert(0, str(_THIS_DIR.parent / 'scene codes'))

import numpy as np

from hunav_config import (
    POSE_LOOKUP_OFFSET_SEC,
    TRAJECTORY_HISTORY_LENGTH, FUTURE_HORIZON_LENGTH, DATASET_SAMPLE_PERIOD_SEC,
)
from time_sync import StampedPoseHistory, clocks_match, stamp_to_sec, wrap_angle
from analyze_ped_visibility import crop_seen   # same "seen" rule as the analyzer and the viewer
import json
try:   # 2026-10-13: robot pose looked up at the same corrected time as the pedestrian (see hunav_config.py)
    from hunav_config import ROBOT_POSE_LOOKUP_OFFSET_SEC
except ImportError:   # line not in hunav_config.py yet -> old behaviour
    ROBOT_POSE_LOOKUP_OFFSET_SEC = 0.0

DEFAULT_OUTPUT_DIR = _THIS_DIR / 'samples'

# Largest allowed spacing between consecutive pedestrian pose samples inside a
# sample's window (normal spacing is 0.5 s). One missed HuNav tick (1.0 s) already
# makes the path between the surviving poses a guess, so such windows are dropped.
MAX_POSE_GAP_SEC = 0.8
# Largest distance between the target lattice time and the chosen grid's own T.
# Grids arrive every ~0.27 s of simulation time, so the nearest is normally < 0.14 s away.
MAX_GRID_MISMATCH_SEC = 0.30
# Robot pose lookup at the grid stamp (odom arrives ~30x per sim second).
ODOM_MAX_HOLD_SEC = 0.2
# If the next grid is this far past the target time, the lattice restarts there.
LATTICE_RESTART_SEC = 10.0
POSE_HISTORY_KEEP_SEC = 40.0
MAX_PENDING = 100
# Two samples whose anchors are closer than this are near-duplicates (same scene,
# same poses). The grid nearest to a lattice time is used, so when grids arrive
# unevenly two consecutive targets can pick grids only ~0.1 s apart (seen: 0.03-0.07 s).
# Such a candidate is skipped and counted instead. 0.3 s = 60% of the sample period.
MIN_ANCHOR_SPACING_SEC = 0.3
# Grids kept for judging visibility at history/future times (2026-10-24). Must cover the
# whole sample window (history 3.5 s + future 6 s) plus the pose stream's delay.
GRID_KEEP_SEC = 20.0
# Start gate (2026-10-24, see collection_gate.py)
GATE_TOPIC = '/dataset/gate'
EPISODE_TOPIC = '/dataset/episode'   # std_msgs/String: the current episode id (published by the episode runner)
EPISODE_MARGIN_SEC = 0.5   # s: an episode change this close to a window also drops it (message delay)
ROBOT_MAX_GAP_SEC = 0.5    # s: largest allowed hole in the robot pose stream inside a window (normal: 0.02 s)
ROBOT_WAIT_SEC = 1.0       # s: wait this long (pedestrian time) for the robot pose to reach the window end
RECORDER_VERSION = '2026-10-27'
GATE_WAIT_SEC = 15.0       # sim s after the first pedestrian pose: no gate heard by then -> ungated
GATE_STALE_SEC = 5.0       # sim s: gate reports this far behind the poses -> it is considered silent
# A scene change is noticed up to ~2 s late (bridge and gate each read once a second), so the
# gate closes from 2 s BEFORE it noticed. Wait until the gate has reported this far past a
# window's end before saving it, so such a late-marked interval still catches the window.
GATE_LAG_SEC = 2.5


class GateTracker:
    """Collects the closed intervals reported by collection_gate.py (simulation time)."""

    def __init__(self):
        self.heard = False
        self.covered = None          # the gate has judged everything up to this time
        self.intervals = {}          # (start, reason) -> end (None = still closed)
        self.open = False            # the gate's state right now (for the status line)
        self.reason = ''

    def on_message(self, d):
        now = d.get('now')
        if now is None:
            return
        if self.covered is not None and now < self.covered - 1.0:      # simulation restarted
            self.intervals.clear()
        self.heard = True
        self.covered = now
        self.open = bool(d.get('open'))
        self.reason = d.get('reason', '')
        for a, b, r in d.get('closed', []):
            self.intervals[(round(a, 3), r)] = b
        for k in [k for k, b in self.intervals.items() if b is not None and b < now - 180.0]:
            del self.intervals[k]

    def closed_reason(self, t0, t1):
        """Reason of a closed interval touching [t0, t1], or '' if none."""
        for (a, r), b in self.intervals.items():
            end = b if b is not None else float('inf')
            if a <= t1 and end >= t0:
                return r
        return ''


def world_pose_to_base_link(x, y, yaw, robot_pos, robot_yaw):
    """Identical to pedestrian_crop_view_dynamic.py's own version -- same
    convention, deliberately kept in sync with it rather than reimplemented
    differently."""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    dx, dy = x - robot_pos[0], y - robot_pos[1]
    base_x = c * dx + s * dy
    base_y = -s * dx + c * dy
    base_yaw = yaw - robot_yaw
    return base_x, base_y, base_yaw


class AlignedSampler:
    """
    All the sampling logic, with no ROS in it (so it can be tested with
    synthetic data). Feed it events -- on_odom / on_ped / on_grid -- in arrival
    order; finished samples go to `sink(sample_dict)`.

    crop_fn(grid_array, ped_x_bl, ped_y_bl, ped_heading_bl) -> crop
        (extract_pedestrian_crop.extract_pedestrian_crop)
    to_local_fn(x, y, anchor_x, anchor_y, anchor_yaw) -> (local_x, local_y)
        (extract_pedestrian_crop.base_to_local)
    """

    def __init__(self, crop_fn, to_local_fn, sink, lookup_offset=POSE_LOOKUP_OFFSET_SEC,
                 history_len=TRAJECTORY_HISTORY_LENGTH, future_len=FUTURE_HORIZON_LENGTH,
                 period=DATASET_SAMPLE_PERIOD_SEC, max_gap=MAX_POSE_GAP_SEC, log=None, gate=None):
        self.crop_fn = crop_fn
        self.to_local_fn = to_local_fn
        self.sink = sink
        self.off = lookup_offset
        self.H, self.F, self.P = history_len, future_len, period
        self.max_gap = max_gap
        self.log = log or (lambda level, msg: None)
        self.gate = gate               # GateTracker, or None = no gate at all (tests)
        self.t_first_ped = None
        self.episode_id = ''           # the current episode (from /dataset/episode)
        self.initial_episode = ''      # the episode before the first recorded change
        self.episode_changes = []      # [(sim time of the change, new id)], oldest first
        self.ungated = False
        self.n_gate_dropped = 0

        self.ped = StampedPoseHistory(max_age_sec=POSE_HISTORY_KEEP_SEC)
        self.odom = StampedPoseHistory(max_age_sec=POSE_HISTORY_KEEP_SEC)
        self.pending = []          # anchors waiting for pose data
        self.prev = None           # (t_g, T, loader) of the previous grid
        self.next_target = None    # next lattice time (in anchor time T)
        self.last_used_tg = None
        self.last_anchor_T = None
        self.grids = deque()       # (t_g, T, loader) of recent grids, for visibility flags
        self._grid_cache = {}      # t_g -> loaded array (only grids actually used)
        self.drops = Counter()
        self.n_saved = 0
        self.n_anchors = 0
        self.n_seen_anchor = 0

    # --- events -------------------------------------------------------------
    def on_odom(self, t, x, y, yaw):
        self.odom.add(t, x, y, yaw)

    def on_episode(self, eid, t=None):
        """A message on /dataset/episode. Only a change of the id counts. t = simulation time of
        the change (default: the newest pedestrian pose time, i.e. 'now' on the sim clock)."""
        eid = (eid or '').strip().replace(';', ',')
        if eid == self.episode_id:
            return
        if t is None:
            t = self.ped.latest_time()
        self.episode_id = eid
        if t is None:                  # nothing recorded yet: this is the starting episode
            self.initial_episode = eid
            return
        self.episode_changes.append((t, eid))
        self.log('info', f'episode changed to {eid!r} at {t:.2f} s (windows around it are dropped)')

    def episode_at(self, t):
        """(id, start time or None) of the episode simulation time t falls in."""
        eid, start = self.initial_episode, None
        for tc, e in self.episode_changes:
            if tc > t:
                break
            eid, start = e, tc
        return eid, start

    def episode_change_in(self, t0, t1):
        return any(t0 <= tc <= t1 for tc, _ in self.episode_changes)

    def on_ped(self, t, x, y, yaw):
        if self.t_first_ped is None:
            self.t_first_ped = t
        self.ped.add(t, x, y, yaw)
        self._advance()

    def on_grid(self, t_g, loader):
        """loader() -> the grid as an int8 array (called only if the grid is used)."""
        if self.prev is not None and t_g <= self.prev[0]:
            if t_g < self.prev[0] - 1.0:
                self._reset('simulation time went backwards (restart?)')
            else:
                return  # duplicate / reordered grid
        last_ped = self.ped.latest_time()
        if last_ped is None:
            self.drops['no pedestrian pose yet'] += 1
            return
        if not clocks_match(t_g, last_ped):
            self.drops['CLOCK MISMATCH grid vs pedestrian pose (run the bridge with use_sim_time:=true)'] += 1
            self.log('error', f'grid stamp {t_g:.2f} and pedestrian pose stamp {last_ped:.2f} are on '
                              f'DIFFERENT clocks -- nothing can be recorded')
            return

        T_new = t_g + self.off
        cur = (t_g, T_new, loader)
        self.grids.append(cur)
        while self.grids and self.grids[0][1] < T_new - GRID_KEEP_SEC:
            self._grid_cache.pop(self.grids.popleft()[0], None)
        if self.next_target is None:
            self.next_target = T_new
        elif T_new - self.next_target > LATTICE_RESTART_SEC:
            self.drops['grid gap: lattice restarted'] += 1
            self.next_target = T_new

        while T_new >= self.next_target - 1e-9:
            best = min([cur] + ([self.prev] if self.prev else []),
                       key=lambda c: abs(c[1] - self.next_target))
            if abs(best[1] - self.next_target) > MAX_GRID_MISMATCH_SEC:
                self.drops['no grid near the sample time'] += 1
            elif best[0] == self.last_used_tg:
                self.drops['grid already used (grids sparser than the sample period)'] += 1
            elif self.last_anchor_T is not None and 0 <= best[1] - self.last_anchor_T < MIN_ANCHOR_SPACING_SEC:
                self.drops['too close to the previous anchor (uneven grids)'] += 1
            else:
                self._start_anchor(best)
            self.next_target += self.P
        self.prev = cur
        self._advance()

    # --- internals -----------------------------------------------------------
    def _reset(self, why):
        self.log('warn', f'resetting sampler: {why}')
        self.pending.clear()
        self.prev = None
        self.next_target = None
        self.last_used_tg = None
        self.last_anchor_T = None
        self.grids.clear()
        self._grid_cache.clear()
        self.initial_episode = self.episode_id      # old change times belong to the old clock
        self.episode_changes = []

    def _start_anchor(self, cand):
        t_g, T, loader = cand
        robot = self.odom.at(t_g + ROBOT_POSE_LOOKUP_OFFSET_SEC, max_hold=ODOM_MAX_HOLD_SEC)
        if robot is None:
            self.drops['no robot pose (/odom) at the grid time'] += 1
            return
        if len(self.pending) >= MAX_PENDING:
            self.pending.pop(0)
            self.drops['pending queue overflow'] += 1
        self.pending.append({'T': T, 't_g': t_g, 'grid': loader(), 'robot': robot, 'crop': None})
        self.last_used_tg = t_g
        self.last_anchor_T = T
        self.n_anchors += 1

    def _advance(self):
        latest = self.ped.latest_time()
        if latest is None or not self.pending:
            return
        remaining = []
        for a in self.pending:
            if a['crop'] is None:
                if latest < a['T']:
                    remaining.append(a)     # the pose at the anchor time has not arrived yet
                    continue
                if not self._cut_crop(a):
                    continue
            t_end = a['T'] + self.F * self.P
            odom_latest = self.odom.latest_time()
            robot_ready = (odom_latest is not None and odom_latest >= t_end - 1e-9) or \
                latest >= t_end + ROBOT_WAIT_SEC
            if latest >= t_end - 1e-9 and robot_ready:
                verdict = self._gate_check(a, latest)
                if verdict == 'wait':
                    remaining.append(a)
                elif verdict:
                    self.drops[f'start gate: {verdict}'] += 1
                    self.n_gate_dropped += 1
                else:
                    self._emit(a, latest)
            else:
                remaining.append(a)
        self.pending = remaining

    def _cut_crop(self, a):
        """Pedestrian pose AT the anchor time (interpolated, never held) + crop. False if impossible."""
        p = self.ped.at(a['T'])
        yaw = self.ped.step_yaw_at(a['T'])
        if p is None or yaw is None:
            self.drops['no pedestrian pose at the anchor time'] += 1
            return False
        x, y = p[0], p[1]
        rx, ry, ryaw = a['robot']
        bx, by, byaw = world_pose_to_base_link(x, y, yaw, (rx, ry), ryaw)
        a['crop'] = self.crop_fn(a['grid'], bx, by, byaw)
        a['ped'] = (x, y, yaw)
        a['grid'] = None
        return True

    def _gate_check(self, a, latest):
        """'' = save, 'wait' = gate has not reported that far yet, else the reason to drop."""
        g = self.gate
        if g is None:
            return ''
        t0, t1 = a['T'] - (self.H - 1) * self.P, a['T'] + self.F * self.P
        if not g.heard:
            if self.t_first_ped is not None and latest - self.t_first_ped < GATE_WAIT_SEC:
                return 'wait'
            if not self.ungated:
                self.ungated = True
                self.log('warn', 'no start gate heard on /dataset/gate: saving samples UNGATED '
                                 '(start collection_gate.py for real data)')
            return ''
        if g.covered < t1 + GATE_LAG_SEC:
            if latest - g.covered > GATE_STALE_SEC:
                return 'gate silent (is collection_gate.py running?)'
            return 'wait'
        return g.closed_reason(t0, t1)

    def _seen_at(self, t):
        """1 / 0: was the pedestrian's spot known in the grid nearest time t (the same rule as the
        crop), -1: no grid near t, or no robot / pedestrian pose for it."""
        if not self.grids:
            return -1
        t_g, T, loader = min(self.grids, key=lambda g: abs(g[1] - t))
        if abs(T - t) > MAX_GRID_MISMATCH_SEC:
            return -1
        robot = self.odom.at(t_g + ROBOT_POSE_LOOKUP_OFFSET_SEC, max_hold=ODOM_MAX_HOLD_SEC)
        p = self.ped.at(T)
        yaw = self.ped.step_yaw_at(T)
        if robot is None or p is None or yaw is None:
            return -1
        grid = self._grid_cache.get(t_g)
        if grid is None:
            grid = self._grid_cache[t_g] = loader()
        bx, by, byaw = world_pose_to_base_link(p[0], p[1], yaw, (robot[0], robot[1]), robot[2])
        return 1 if crop_seen(self.crop_fn(grid, bx, by, byaw))[0] else 0

    def _emit(self, a, latest):
        T, P, H, F = a['T'], self.P, self.H, self.F
        t0, t1 = T - (H - 1) * P, T + F * P
        first = self.ped.first_time()
        if first is None or t0 < first - 1e-9:
            self.drops['pose history does not reach back far enough yet (start-up)'] += 1
            return
        gap = self.ped.max_gap(t0, min(t1, latest))
        if gap > self.max_gap:
            self.drops['pose dropout inside the sample window'] += 1
            return
        if self.episode_change_in(t0 - EPISODE_MARGIN_SEC, t1 + EPISODE_MARGIN_SEC):
            self.drops['episode changed inside the window'] += 1
            return
        if self.odom.max_gap(t0, t1) > ROBOT_MAX_GAP_SEC:
            self.drops['robot pose dropout inside the sample window'] += 1
            return

        ax, ay, ayaw = a['ped']

        def row(t):
            t = min(max(t, first), latest)
            pos = self.ped.at(t)
            yaw = self.ped.step_yaw_at(t)
            lx, ly = self.to_local_fn(pos[0], pos[1], ax, ay, ayaw)
            return float(lx), float(ly), float(wrap_angle(yaw - ayaw))

        times_h = np.array([T - (H - 1 - i) * P for i in range(H)], dtype=np.float64)
        times_f = np.array([T + (k + 1) * P for k in range(F)], dtype=np.float64)
        history = np.array([row(t) for t in times_h], dtype=np.float32)
        future = np.array([row(t) for t in times_f], dtype=np.float32)
        if not (np.isfinite(history).all() and np.isfinite(future).all()):
            self.drops['non-finite pose values'] += 1
            return
        robot_rows = [self.odom.at(float(t), max_hold=ODOM_MAX_HOLD_SEC) for t in np.concatenate([times_h, times_f])]
        if any(r is None for r in robot_rows):
            self.drops['robot pose dropout: no robot pose for a history/future step'] += 1
            return
        robot_rows = np.array(robot_rows, dtype=np.float64)
        robot_hist, robot_fut = robot_rows[:H], robot_rows[H:]
        eid, e_start = self.episode_at(T)
        seen_anchor = 1 if crop_seen(a['crop'])[0] else 0
        seen_h = np.array([self._seen_at(t) for t in times_h[:-1]] + [seen_anchor], dtype=np.int8)
        seen_f = np.array([self._seen_at(t) for t in times_f], dtype=np.int8)
        self.n_seen_anchor += seen_anchor

        self.sink({
            'trajectory_history': history,
            'crop': np.asarray(a['crop']),
            'future_target': future,
            'agent_id': 'agent1',
            'sample_period_sec': P,
            'wall_time': time.time(),
            'schema': 'time_aligned_v2',
            'anchor_time_sim': float(T),
            'grid_stamp_sim': float(a['t_g']),
            'lookup_offset_sec': float(self.off),
            'history_times': times_h,
            'future_times': times_f,
            'robot_pose_odom': np.array(a['robot'], dtype=np.float64),
            'ped_anchor_odom': np.array(a['ped'], dtype=np.float64),
            'max_pose_gap_sec': float(gap),
            'ped_seen_anchor': np.int8(seen_anchor),
            'ped_seen_history': seen_h,
            'ped_seen_future': seen_f,
            'gated': bool(self.gate is not None and self.gate.heard and not self.ungated),
            'robot_history_odom': robot_hist,
            'robot_future_odom': robot_fut,
            'episode_id': eid,
            'episode_time_sec': float(T - e_start) if (eid and e_start is not None) else -1.0,
        })
        self.n_saved += 1

    def status(self):
        drops = ', '.join(f'{n}x {r}' for r, n in self.drops.most_common(4)) or 'none'
        seen = f' ({100 * self.n_seen_anchor / self.n_saved:.0f}% with the pedestrian seen)' if self.n_saved else ''
        if self.gate is None or (not self.gate.heard and not self.ungated):
            gate = 'gate: not heard yet'
        elif self.ungated:
            gate = 'gate: NOT RUNNING (samples are ungated)'
        else:
            state = 'OPEN' if self.gate.open else f'CLOSED: {self.gate.reason}'
            gate = f'gate: on ({state}), {self.n_gate_dropped} samples dropped by it'
        ep = f'episode: {self.episode_id!r} ({len(self.episode_changes)} changes)' if \
            (self.episode_id or self.episode_changes) else 'episode: none'
        # the gate part must stay LAST (record_session.py reads it up to the end of the line)
        return (f'{self.n_saved} samples saved{seen}, {len(self.pending)} waiting for future poses, '
                f'{self.n_anchors} anchors started; dropped: {drops}; {ep}; {gate}')


def collect_constants(modules):
    """{module: {NAME: value}} of every UPPER_CASE number / string / bool / short tuple in each module."""
    out = {}
    for name, mod in modules.items():
        if mod is None:
            continue
        vals = {}
        for k in dir(mod):
            if not k.isupper() or k.startswith('_'):
                continue
            v = getattr(mod, k)
            if isinstance(v, (bool, int, float, str)):
                vals[k] = v
            elif isinstance(v, (tuple, list)) and len(v) <= 20 and all(isinstance(e, (bool, int, float, str)) for e in v):
                vals[k] = list(v)
            elif isinstance(v, Path):
                vals[k] = str(v)
        out[name] = vals
    return out


# =============================================================================
# ROS wiring
# =============================================================================
def run_node(output_dir: Path):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid, Odometry
    from geometry_msgs.msg import PoseArray
    from std_msgs.msg import String

    import build_occupancy_grid as bog
    from extract_pedestrian_crop import extract_pedestrian_crop, base_to_local
    from hunav_config import yaw_from_quaternion, ensure_single_instance  # noqa: F401

    class DatasetRecorder(Node):
        def __init__(self):
            super().__init__('dataset_recorder')
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.gate = GateTracker()
            self.sampler = AlignedSampler(
                extract_pedestrian_crop, base_to_local, self._write,
                log=self._log, gate=self.gate)
            self.create_subscription(String, GATE_TOPIC, self._gate_cb, 10)
            self.create_subscription(String, EPISODE_TOPIC, self._episode_cb, 10)

            self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
            self.create_subscription(PoseArray, '/people_smoothed_pose', self._ped_cb, 10)
            self.create_subscription(OccupancyGrid, '/occupancy_grid/base', self._grid_cb, 10)
            self.create_timer(10.0, self._status)   # wall-clock timer: status line only
            self._write_info()

            H, F, P = TRAJECTORY_HISTORY_LENGTH, FUTURE_HORIZON_LENGTH, DATASET_SAMPLE_PERIOD_SEC
            self.get_logger().info(
                f'dataset_recorder (time-aligned) started. Writing to {self.output_dir} '
                f'(run id {self.run_id}). history={H}, future={F}, period={P:.2f} s of SIMULATION '
                f'time, anchor = grid stamp {POSE_LOOKUP_OFFSET_SEC:+.2f} s. First sample after '
                f'~{(H - 1) * P + F * P:.1f} s of simulation time.')

        def _log(self, level, msg):
            # one call line per severity (rclpy refuses a severity change on the same line)
            if level == 'error':
                self.get_logger().error(msg, throttle_duration_sec=5.0)
            elif level == 'warn':
                self.get_logger().warn(msg, throttle_duration_sec=5.0)
            else:
                self.get_logger().info(msg, throttle_duration_sec=5.0)

        def _gate_cb(self, msg):
            try:
                self.gate.on_message(json.loads(msg.data))
            except (ValueError, TypeError, KeyError) as e:
                self.get_logger().warn(f'bad /dataset/gate message: {e}', throttle_duration_sec=10.0)

        def _episode_cb(self, msg):
            self.sampler.on_episode(msg.data)

        def _write_info(self):
            info_dir = self.output_dir.parent if self.output_dir.name == 'samples' else self.output_dir
            info = {'recorder_version': RECORDER_VERSION, 'run_id': self.run_id,
                    'started_wall': datetime.now().isoformat(timespec='seconds'),
                    'samples_dir': str(self.output_dir),
                    'constants': collect_constants({'dataset_recorder': sys.modules[__name__],
                                                    'hunav_config': sys.modules.get('hunav_config'),
                                                    'build_occupancy_grid': bog,
                                                    'extract_pedestrian_crop': sys.modules.get('extract_pedestrian_crop')})}
            try:
                (info_dir / 'recorder_info.json').write_text(json.dumps(info, indent=2, default=str))
            except OSError as e:
                self.get_logger().warn(f'could not write recorder_info.json: {e}')

        def _odom_cb(self, msg):
            p = msg.pose.pose.position
            self.sampler.on_odom(stamp_to_sec(msg.header.stamp), p.x, p.y,
                                 yaw_from_quaternion(msg.pose.pose.orientation))

        def _ped_cb(self, msg):
            if not msg.poses:
                return
            pose = msg.poses[0]  # single-agent, matches every other assumption in this codebase
            yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
            self.sampler.on_ped(stamp_to_sec(msg.header.stamp), pose.position.x, pose.position.y, yaw)

        def _grid_cb(self, msg):
            if msg.info.width != bog.GRID_N or msg.info.height != bog.GRID_N:
                self.get_logger().warn(
                    f'received base grid is {msg.info.width}x{msg.info.height}, '
                    f'expected {bog.GRID_N}x{bog.GRID_N} -- ignoring this message.',
                    throttle_duration_sec=5.0)
                return
            self.sampler.on_grid(
                stamp_to_sec(msg.header.stamp),
                lambda m=msg: np.array(m.data, dtype=np.int8).reshape((m.info.height, m.info.width)))

        def _write(self, sample):
            out_path = self.output_dir / f'sample_{self.run_id}_{self.sampler.n_saved:06d}.npz'
            sample = dict(sample)
            sample['crop'] = sample['crop'].astype(np.int8)
            np.savez(out_path, **sample)
            n = self.sampler.n_saved + 1
            if n == 1 or n % 20 == 0:
                self.get_logger().info(f'{n} samples saved -> {self.output_dir}')

        def _status(self):
            self.get_logger().info(self.sampler.status())

    ensure_single_instance('dataset_recorder')
    rclpy.init()
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Stopped. {node.sampler.status()}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    if len(sys.argv) == 2:
        output_dir = Path(sys.argv[1]).expanduser().resolve()
    run_node(output_dir)


if __name__ == '__main__':
    main()