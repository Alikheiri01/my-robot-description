#!/usr/bin/env python3
"""
mvp_config.py -- every number the MVP collection uses, in one place (2026-11-01).

The environment generator, the offline check, the robot driver and the episode
runner all import from here, so a value is changed once. Everything is a
placeholder for the MVP: widths and sizes get narrowed with the faculty
measurements later, the rest goes into the parameter study.
"""
import math

# ----------------------------------------------------------------------------
# Recorder rules (must match dataset_recorder.py; only copied here for planning)
# ----------------------------------------------------------------------------
PERIOD_SEC = 0.5                 # DATASET_SAMPLE_PERIOD_SEC
HISTORY_STEPS = 8                # TRAJECTORY_HISTORY_LENGTH
FUTURE_STEPS = 12                # FUTURE_HORIZON_LENGTH
EPISODE_MARGIN_SEC = 0.5         # dataset_recorder.EPISODE_MARGIN_SEC
USABLE_MIN_HISTORY_SEEN = 6      # training filter: anchor seen and >= 6 of 8 history steps seen
# first kept anchor after the walk starts / last kept anchor before the episode ends
START_MARGIN_SEC = (HISTORY_STEPS - 1) * PERIOD_SEC + EPISODE_MARGIN_SEC   # 4.0 s
END_MARGIN_SEC = FUTURE_STEPS * PERIOD_SEC + EPISODE_MARGIN_SEC            # 6.5 s

# ----------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------
EPISODES_PER_ENV = 6
SPEED_RANGE = (0.6, 1.4)         # m/s, uniform per episode
BEFORE_EVENT_SEC = 12.0          # walking time from the walk start to the event (>=)
AFTER_EVENT_SEC = 8.0            # walking time from the event to the episode end (>=)
# The episode ends ('end' id published) while the pedestrian still walks at full speed:
# HuNav brakes only ~0.2-0.3 s (0.5-0.65 m) before its goal (measured, T05/T01/T06), so
# ending 1.5 m before the goal keeps every saved window free of the unexplained stop.
END_TRIGGER_M = 1.5
DETOUR_MIN_M = 0.4               # A and D: the route must leave the straight line by at least this
EVENT_OBSTACLE_GAP_M = 1.1       # A: the free side of the obstacle is at least this wide

# Scenario C (designed slowdown, see the spec): v(d) = v0 * sqrt(d / SLOWDOWN_DISTANCE_M)
SLOWDOWN_DISTANCE_RANGE = (1.0, 2.5)
STOP_GAP_RANGE = (0.5, 0.9)      # pedestrian centre -> wall face at the stop point
STAND_SEC = 7.5                  # standing time before 'end': >= 6 s future + margins

# ----------------------------------------------------------------------------
# Camera (depth_camera.xacro) and the "seen" model used for planning
# ----------------------------------------------------------------------------
CAM_OFFSET_M = 0.37              # camera ahead of the robot centre
CAM_HALF_FOV = math.radians(36.5)
CAM_RANGE_M = 8.0
PLAN_FOV_MARGIN = math.radians(5.0)   # planning is pessimistic: the pedestrian must be well inside
PLAN_RANGE_MARGIN_M = 0.7
PED_BODY_M = 0.3                 # the last 0.3 m of the sight line is the pedestrian itself

# ----------------------------------------------------------------------------
# Robot behaviour
# ----------------------------------------------------------------------------
ROBOT_RADIUS_M = 0.35            # footprint radius (CHECK with the robot's real size)
ROBOT_WALL_CLEAR_M = 0.7         # parked robot centre -> any wall/obstacle (2026-11-02: was 0.5; a watching robot drifts ~0.5 m while it turns, skid steering)
ROBOT_PATH_CLEAR_M = 0.45        # follow: the robot's track keeps this from every wall (radius + 0.1)
ROBOT_ROUTE_CLEAR_M = 1.2        # parked robot centre -> pedestrian route (HuNav reacts to the robot)
ROBOT_START_CLEAR_M = 1.5        # robot centre -> pedestrian start / goal
ROBOT_VMAX = 1.5                 # m/s (follow mode) (2026-11-02: was 0.8, an assumption; DiffDrive allows 4.0, pedestrians walk up to 1.4)
ROBOT_WMAX = 1.0                 # rad/s (watch and follow)
WATCH_DEADBAND = math.radians(10.0)   # watch: turn when the pedestrian is further than this off axis
WATCH_GAIN = 2.0                 # rad/s per rad of heading error
FOLLOW_DIST_RANGE = (3.0, 5.0)   # m behind the pedestrian, along its route
FOLLOW_LOOKAHEAD_M = 1.0
FOLLOW_GAIN = 1.0                # 1/s, speed = gain * (distance error)
BEHIND_START_M = 6.0             # free hallway behind the pedestrian start (room for a follower)
MODE_SHARE = {'still': 0.3, 'watch': 0.3, 'follow': 0.4}
USABLE_TARGET = 0.6              # planned usable share; the real target is >= 0.5 (margin for the model)
ROBOT_CANDIDATE_GRID_M = 0.3
ROBOT_CANDIDATE_TRIES = 250

# ----------------------------------------------------------------------------
# Geometry (placeholders until the faculty measurements)
# ----------------------------------------------------------------------------
WALL_T = 0.2
WALL_H = 2.0
FURNITURE_H_RANGE = (1.0, 2.0)
EVENT_OBSTACLE_H_RANGE = (1.0, 1.8)
HALL_WIDTH_RANGE = (1.8, 3.0)     # hallway
DOOR_WIDTH_RANGE = (1.0, 1.4)     # 0.9 only after a test shows HuNav passes it (agent radius 0.4)
ROOM_WIDTH_RANGE = (5.0, 8.0)     # D: the two rooms
BAY_W, BAY_D = 1.8, 1.4           # wall niche where a parked robot can stand
BAYS_PER_HALL = (2, 4)
COLUMN_R_RANGE = (0.2, 0.35)      # E hall columns
COLUMN_SPACING_RANGE = (4.0, 6.0)
ARC_RADIUS_RANGE = (6.0, 12.0)    # E arcs
ARC_WAYPOINT_STEP_M = 0.5
FURNITURE_PER_ENV = (0, 3)
FURNITURE_ROUTE_CLEAR_M = 0.9     # furniture never changes a planned route (route keeps 0.6 m)

# A event obstacle
A_BOX_ALONG_RANGE = (0.4, 0.8)    # depth along the walking direction
A_BOX_ACROSS_RANGE = (0.4, 1.2)
A_CYL_R_RANGE = (0.15, 0.4)
A_BOX_SHARE = 0.6

# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------
SCENARIOS = {
    'A': 'A_detour',
    'B': 'B_corner',
    'C': 'C_stop',
    'D': 'D_door',
    'E': 'E_walk',
}
PLAN_DT = 0.1                     # time step of the planning model
# The bridge plans a route only when it gets a goal (the pedestrian stands at its start then)
# or when the obstacles change (never during an episode), so a slow plan delays the walk
# start but never shows up in the data. Above this it is reported as a problem.
PLANNER_SLOW_SEC = 1.0
# Every environment is shifted so it starts this far from the odom origin (+x): the robot
# (odom 0,0) and the pedestrian (launch spawn, odom ~2,0) are then OUTSIDE the walls when
# an environment is spawned, whatever the order of teleport and spawn.
ENV_OFFSET_X = 4.0

# ----------------------------------------------------------------------------
# Wall-clock budget estimate (check_environments.py); refine after the first runs
# ----------------------------------------------------------------------------
RTF = 0.37                        # measured real-time factor of the full simulation
RESET_SIM_SEC = 5.0               # teleport + waiting for the gate to open again, per episode
ENV_OVERHEAD_WALL_SEC = 120.0     # spawn, wait for the bridge, start/stop record_session, its report

# ----------------------------------------------------------------------------
# Episode runner, bridge episode mode, robot driver (P3-P5)
# ----------------------------------------------------------------------------
PATH_LOOKAHEAD_M = 1.5            # path mode (E arcs): HuNav's goal = this far ahead on the path
WALK_START_SPEED = 0.3            # m/s: the walk has started -> publish the episode id
ARRIVE_SPEED = 0.1                # m/s: below this within ARRIVE_DIST_M of the goal = arrived
ARRIVE_DIST_M = 1.0
STUCK_SEC = 5.0                   # sim s below ARRIVE_SPEED away from the goal = stuck (abort)
TIMEOUT_FACTOR = 2.0              # abort after this x the planned episode time (+ 20 s)
RESET_SETTLE_SIM_SEC = 1.0        # after a teleport, before the goal is sent
GATE_WAIT_WALL_SEC = 60.0         # wait at most this long for the start gate to open again
SCENE_WAIT_WALL_SEC = 60.0        # wait at most this long for the bridge to report the new scene
RECORD_START_WALL_SEC = 12.0      # time given to record_session.py to start its recorder
RECORD_STOP_WALL_SEC = 300.0      # time given to record_session.py to write its report
ROBOT_NAME = 'my_robot'
ROBOT_MIN_PED_DIST_M = 1.5        # follow: the robot stops when it is this close to the pedestrian
FOLLOW_TURN_FIRST_RAD = 0.6       # follow: turn in place when the lookahead point is this far off