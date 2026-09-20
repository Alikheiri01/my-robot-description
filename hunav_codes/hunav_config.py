#!/usr/bin/env python3
"""
Single shared source of truth for every tunable constant across the HuNav
integration. Created specifically to close a known fragility: SPAWN_X/
SPAWN_Y/SPAWN_YAW used to be hardcoded and duplicated in both
hunav_model_bridge.py and build_occupancy_grid_dynamic.py, one edit away
from silently going out of sync. Every script in codes/ that needs any of
these values should import them from here, not redefine its own copy.

Lives in the same codes/ directory as everything that imports it, so a
plain `import hunav_config` works with no packaging changes -- when Python
runs a script directly (as our ExecuteProcess launch actions do), the
script's own directory is automatically on sys.path.
"""

# =============================================================================
# Robot spawn pose -- MUST match spawn_entity's -x/-y/-Y arguments in
# my_robot_hunav_launch.launch.py exactly. DiffDrive's simulated odometry
# starts at (0,0,0) regardless of the robot's true spawn pose (confirmed
# bug, see hunav_model_bridge.py's own history) -- these three values are
# what let every consumer correct /odom back into real world-frame pose.
# =============================================================================
SPAWN_X = 0.0
SPAWN_Y = 0.0
SPAWN_YAW = 0.7854

import math as _math

_COS_SPAWN_YAW = _math.cos(SPAWN_YAW)
_SIN_SPAWN_YAW = _math.sin(SPAWN_YAW)


def yaw_from_quaternion(q) -> float:
    """Standard quaternion -> yaw (Z-axis Euler) extraction. Used everywhere
    an /odom message needs its orientation reduced to a single planar angle
    -- centralized here since it was independently duplicated in
    hunav_model_bridge.py and build_occupancy_grid_dynamic.py."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return _math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float):
    """Returns (x, y, z, w) for a pure yaw rotation -- same convention used
    throughout this project."""
    return (0.0, 0.0, _math.sin(yaw / 2.0), _math.cos(yaw / 2.0))


def odom_to_world(odom_x: float, odom_y: float, odom_yaw: float):
    """
    DiffDrive's odometry starts at (0,0,0) regardless of the robot's true
    world spawn pose (confirmed empirically). Composes odom's own relative
    motion with the robot's known true spawn transform to recover genuine
    world-frame pose. Needed wherever something has to be physically
    correct in actual Gazebo coordinates -- SFM computation (hunav_model_bridge.py's
    /compute_agents calls) and moving the pedestrian_standin model
    (set_pose, which is real Gazebo-world-frame, no notion of 'odom').
    """
    world_x = SPAWN_X + odom_x * _COS_SPAWN_YAW - odom_y * _SIN_SPAWN_YAW
    world_y = SPAWN_Y + odom_x * _SIN_SPAWN_YAW + odom_y * _COS_SPAWN_YAW
    world_yaw = SPAWN_YAW + odom_yaw
    return world_x, world_y, world_yaw


def world_to_odom(world_x: float, world_y: float, world_yaw: float):
    """
    Inverse of odom_to_world() above -- needed for the opposite direction:
    taking something already expressed in true Gazebo world coordinates
    (e.g. the pedestrian's position/yaw from /people, which is world-frame
    because it feeds set_pose) and converting it INTO the odom-frame
    convention that RViz's own TF tree actually uses for everything else
    (the depth-camera point cloud, the robot model, etc.) -- there is no
    real localization stack here, so 'map' is just a static identity link
    to 'odom', and 'odom' itself is rotated/translated from true world by
    exactly the SPAWN_X/SPAWN_Y/SPAWN_YAW transform (same confirmed bug as
    above). Publishing a world-frame value under frame_id='map' without
    this conversion silently mismatches with everything else RViz draws --
    not noise, not latency, a genuine frame convention mismatch. Anything
    published purely for RViz visualization (not physics/Gazebo) should go
    through this first.
    """
    dx = world_x - SPAWN_X
    dy = world_y - SPAWN_Y
    odom_x = dx * _COS_SPAWN_YAW + dy * _SIN_SPAWN_YAW
    odom_y = -dx * _SIN_SPAWN_YAW + dy * _COS_SPAWN_YAW
    odom_yaw = world_yaw - SPAWN_YAW
    return odom_x, odom_y, odom_yaw


# =============================================================================
# Gazebo world / pedestrian stand-in model
# =============================================================================
WORLD_NAME = 'empty'
MODEL_NAME = 'pedestrian_standin'

# Height at which the pedestrian_standin model's root is placed -- matches
# HuNavSystemPlugin_fortress.cpp's own updateGazeboPedestrians(), which sets
# actorPose.Pos().Z(0.8) for this exact mesh family. Confirmed from source,
# not guessed -- do not change without a reason grounded the same way.
STANDIN_Z = 0.8

# =============================================================================
# hunav_model_bridge.py
# =============================================================================
BRIDGE_UPDATE_RATE_HZ = 2.0  # deliberately conservative -- see that file's
                              # own docstring for the subprocess-overhead reasoning
SET_POSE_TIMEOUT_MS = 300

# Diagnostic move/freeze duty cycle -- added to debug the RViz position/
# orientation offset (pedestrian moves too fast to visually inspect whether
# the arrow and the depth-camera obstacle actually settle onto each other).
# When PAUSE_PHASE_ENABLED is True, the bridge alternates between a MOVE
# phase (calls /compute_agents normally, exactly as before) and a FREEZE
# phase (skips the call entirely -- pedestrian pose, /people, and
# /people_smoothed_pose all stay frozen at their last value). If the offset
# shrinks/vanishes during a freeze, that's a clean confirmation the mismatch
# is pipeline latency on a moving target, not a coordinate/orientation bug.
# Set back to False once the experiment is done -- this is a debugging aid,
# not a permanent behavior.
PAUSE_PHASE_ENABLED = True
PAUSE_PHASE_MOVE_SEC = 2.5    # how long the pedestrian walks before freezing
PAUSE_PHASE_FREEZE_SEC = 4.0  # how long it holds still -- generous, to give
                               # RViz's slower depth pipeline time to fully
                               # catch up and settle

# =============================================================================
# Occupancy grid live exclusion (build_occupancy_grid_dynamic.py)
# =============================================================================
# Matches (with a small safety margin) thesis_static_agent.yaml's agent
# radius (0.4m). NOTE: still a separate hand-set value, not read from the
# YAML directly -- update by hand if that scenario's radius ever changes.
PEDESTRIAN_EXCLUSION_RADIUS = 0.5

# =============================================================================
# Heading smoothing (heading_smoother.py)
# =============================================================================
# Number of recent /people readings averaged together for the smoothed
# heading estimate. At BRIDGE_UPDATE_RATE_HZ=2.0, 3 samples = 1.5 seconds of
# averaging -- long enough to filter per-frame noise, short enough to still
# track a real direction change (e.g. the pedestrian reversing at a goal)
# reasonably promptly. Worth tuning empirically once the live crop
# visualizer (planned next) lets us actually watch behavior during a turn.
HEADING_SMOOTHING_WINDOW = 3

# Below this speed (m/s), heading is ill-defined (atan2 of a near-zero
# vector is mostly noise) -- the smoother holds the last reliable heading
# instead of updating it. thesis_static_agent.yaml's current max_vel is
# 1.0 m/s; 0.1 m/s is roughly 10% of that.
MIN_SPEED_FOR_HEADING_UPDATE = 0.1

# =============================================================================
# Trajectory / crop dataset pairing (Phase D, dataset export)
# =============================================================================
# Matches common convention in pedestrian-trajectory-prediction literature
# (Social-LSTM/Trajectron++-era work typically uses ~8 historical steps) --
# a defensible starting point, not an arbitrary guess. At
# BRIDGE_UPDATE_RATE_HZ=2.0 this is 4 seconds of history per sample.
TRAJECTORY_HISTORY_LENGTH = 8

# How many future steps the network is trained to predict, at the same
# sample cadence as history (DATASET_SAMPLE_PERIOD_SEC below). 12 steps at
# that cadence is 6 seconds -- deliberately longer than the 4s of history,
# a common obs:predict ratio in the literature (e.g. Social-LSTM's 3.2s
# observed / 4.8s predicted on ETH/UCY). Tune once real training starts.
FUTURE_HORIZON_LENGTH = 12

# Dataset samples are taken on a fixed timer, NOT per /occupancy_grid/base
# message -- that topic publishes at camera rate, far faster than the
# pedestrian actually moves (BRIDGE_UPDATE_RATE_HZ=2.0), so sampling on
# every grid frame would just duplicate near-identical poses. Sampling at
# the pedestrian's own update rate is what makes TRAJECTORY_HISTORY_LENGTH's
# "4 seconds of history" comment above actually true.
DATASET_SAMPLE_PERIOD_SEC = 1.0 / BRIDGE_UPDATE_RATE_HZ

# A pose/grid older than this is treated as stale -- same margin reasoning
# as pedestrian_crop_view_dynamic.py's own STALE_THRESHOLD_SEC (comfortably
# above PAUSE_PHASE_FREEZE_SEC so an intentional bridge freeze isn't
# mistaken for a dead feed). A stale reading resets the sample buffer
# rather than silently splicing across the gap into a fake trajectory.
DATASET_STALE_THRESHOLD_SEC = PAUSE_PHASE_FREEZE_SEC + 3.0

# One .npz file per training sample (trajectory_history, crop, future_target,
# timestamp, agent_id as named arrays) -- simplest format that loads
# directly into a PyTorch Dataset's __getitem__ via np.load(), no new
# dependencies beyond numpy (already used throughout this codebase).
DATASET_SAMPLE_FORMAT = 'npz'


# =============================================================================
# Single-instance enforcement
# =============================================================================
# Added after repeated duplicate-process incidents (a manually-started
# heading_smoother.py surviving Ctrl+Z or a closed terminal, running
# alongside the launch-file-started copy, silently producing two competing
# smoothing histories on the same data). ROS2 itself only warns on a
# duplicate node name, it doesn't prevent one -- so this is a real,
# filesystem-level lock instead of relying on remembering to pkill
# everything correctly every time.
#
# REVISED: the original version stored the PID in a plain text file and
# checked liveness with os.kill(pid, 0) -- "does any process with this
# PID number currently exist." That has a real false-positive failure
# mode: cleanup_hunav.sh (correctly) uses `pkill -9`, and a SIGKILL can't
# be caught, so the atexit handler that deletes the lock file never runs.
# The stale lock file survives with the dead process's old PID still
# written in it. Once the OS eventually reuses that PID number for any
# unrelated process, os.kill(pid, 0) succeeds again, and the check can't
# tell the difference -- it reports "already running" when nothing of the
# kind is true (confirmed happening in practice: `ps aux` showed no such
# process, yet the old check still refused to start).
#
# Fixed by switching to fcntl.flock() on the lock file itself, held for as
# long as the file descriptor stays open. The OS releases an flock the
# instant its owning process exits, by ANY means -- normal exit, an
# uncaught exception, or SIGKILL -- with zero PID bookkeeping and zero
# PID-reuse ambiguity: the check is simply "is this file currently locked
# by some living process," never "does this PID number happen to belong
# to something." No behavior change from the caller's side -- still just
# call this once at the top of main().
import fcntl
import os
import sys

_lock_file_handles = {}  # lock_name -> open file object, kept alive so the flock isn't released early


def ensure_single_instance(lock_name: str):
    """
    Call this as the very first line of main(), before rclpy.init(), in
    any script that must never run twice at once. If another instance is
    already alive, prints a clear message and exits immediately -- refuses
    to start, rather than silently running alongside the existing one.
    A lock from a process that's gone -- however it ended, including
    kill -9 -- is released automatically by the OS; no manual staleness
    detection needed.
    """
    lock_path = f'/tmp/hunav_{lock_name}.lock'

    f = open(lock_path, 'w')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        print(
            f'ERROR: {lock_name} is already running (another process holds '
            f'{lock_path}). Refusing to start a second instance.',
            file=sys.stderr,
        )
        sys.exit(1)

    f.write(str(os.getpid()))
    f.flush()
    # Keep the file object alive for the life of the process -- closing it
    # (including via garbage collection) would release the flock early.
    _lock_file_handles[lock_name] = f