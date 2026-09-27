#!/usr/bin/env python3
"""
check_actor_motion.py -- v2: pose TRUSTWORTHINESS checker.

WHY v2 EXISTS
v1 threw away the evidence it was supposed to collect. It throttled its
status line to once per second while running goal-arrival checks on every
message, which aliased a rapidly alternating pose stream into what looked
like a single smooth trajectory -- and simultaneously produced an arrival
log that contradicted its own position readout. The contradiction was the
real finding; the throttle hid it.

This version logs EVERY message and makes the pose stream's internal
consistency the primary output, not goal progress. Specifically it answers:

  1. How many people are in each message, and what are their names?
     If this is ever greater than 1, or the name changes between messages,
     then indexing people[0] is mixing two different entities' states into
     one apparent trajectory -- the leading hypothesis for the alternating
     positions seen in the v1 log.

  2. Does the REPORTED velocity agree with the velocity DERIVED from
     successive positions, in both magnitude and direction? hunav_agent_manager
     fills the velocity field itself; deriving it independently from position
     deltas is a free cross-check on that claim. Large disagreement means the
     position and velocity fields describe different motions.

  3. Is the state ever frozen (identical position across messages) while
     still claiming nonzero velocity? That pair is self-contradictory and
     was clearly present in the v1 log.

  4. Are there physically impossible jumps -- displacement per message
     implying a speed far above the scenario's own max_vel?

SCOPE -- READ THIS BEFORE TRUSTING THE OUTPUT
/people comes from hunav_agent_manager, the same component that computes
the motion. This tool can therefore prove that the pose stream is
self-inconsistent (which is decisive), but it CANNOT prove the stream
matches what Gazebo actually renders. That is a separate link, through
hunav_model_bridge.py's set_pose calls. Confirming it needs a second,
independently-routed source read from Gazebo itself -- deliberately not
attempted here until we know which Gazebo topic actually carries the
actor's pose.

USAGE
    python3 check_actor_motion.py [path/to/scenario.yaml]
Finish with Ctrl+C -- NOT Ctrl+Z. Ctrl+Z only suspends the process; it
keeps its subscriptions alive in the background and never prints the
summary, which is what happened on the previous run.
"""
import math
import sys
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from people_msgs.msg import People


DEFAULT_SCENARIO_YAML = (
    '/home/ali/ros2_ws/src/my_robot_description/'
    'hunav_assets/scenarios/thesis_static_agent.yaml'
)

# Log every single message. This is the whole point of v2 -- do not add
# throttling back without also logging the full series somewhere else.
LOG_EVERY_MESSAGE = True

# Positions closer than this across consecutive messages count as "frozen".
FROZEN_EPSILON_M = 1e-4

# Derived speed above max_vel * this factor is flagged as a teleport rather
# than motion. 1.5 leaves headroom for jitter in message arrival times.
JUMP_SPEED_FACTOR = 1.5

# Reported-vs-derived speed disagreement above this is flagged.
SPEED_DISAGREE_TOL = 0.25      # m/s
HEADING_DISAGREE_TOL = 45.0    # degrees

GOAL_EXIT_MULTIPLIER = 1.5


def load_scenario(path):
    """Fails loudly rather than defaulting -- a checker validating against
    the wrong numbers is worse than one that refuses to start."""
    with open(path, 'r') as f:
        doc = yaml.safe_load(f)
    params = doc['hunav_loader']['ros__parameters']
    agent_name = params['agents'][0]
    agent = params[agent_name]
    global_goals = params['global_goals']
    goals = []
    for gid in agent['goals']:
        g = global_goals.get(gid, global_goals.get(str(gid)))
        if g is None:
            raise ValueError(f'Goal id {gid} missing from global_goals')
        goals.append({'id': gid, 'x': float(g['x']), 'y': float(g['y'])})
    return {
        'agent_name': agent_name,
        'goals': goals,
        'goal_radius': float(agent.get('goal_radius', 0.3)),
        'max_vel': float(agent.get('max_vel', 1.0)),
        'cyclic': bool(agent.get('cyclic_goals', False)),
    }


def angle_diff_deg(a_deg, b_deg):
    """Smallest absolute difference between two headings, in degrees."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(d)


class PoseConsistencyChecker(Node):
    def __init__(self, scenario):
        super().__init__('check_actor_motion')
        self.s = scenario

        self.n_msgs = 0
        self.first_t = None
        self.last_t = None
        self.last_pos = None

        self.names_seen = {}          # name -> count
        self.people_counts = {}       # len(msg.people) -> count

        self.frozen_count = 0
        self.frozen_run = 0
        self.frozen_run_max = 0
        self.frozen_with_velocity = 0

        self.jump_count = 0
        self.max_derived_speed = 0.0

        self.speed_disagreements = 0
        self.heading_disagreements = 0
        self.sum_abs_speed_diff = 0.0
        self.speed_samples = 0

        self.path_length = 0.0
        self.min_x = self.min_y = float('inf')
        self.max_x = self.max_y = float('-inf')

        self.goal_stats = {
            g['id']: {'closest': float('inf'), 'arrivals': 0, 'inside': False}
            for g in scenario['goals']
        }

        # BEST_EFFORT is compatible with either a BEST_EFFORT or a RELIABLE
        # publisher, so this subscription cannot silently fail to connect
        # the way a RELIABLE subscriber can against a BEST_EFFORT publisher.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(People, '/people', self._cb, qos)

        print('=' * 100)
        print(f'agent="{scenario["agent_name"]}"  goal_radius={scenario["goal_radius"]}m  '
              f'max_vel={scenario["max_vel"]}m/s  cyclic={scenario["cyclic"]}')
        for g in scenario['goals']:
            print(f'   goal {g["id"]}: ({g["x"]:.2f}, {g["y"]:.2f})')
        print('Logging EVERY /people message. Ctrl+C (not Ctrl+Z) for the summary.')
        print('=' * 100)
        print('   idx      t     dt   n  name       pos                 '
              'v_rep  v_der   hdg_rep  hdg_der   flags')

    def _cb(self, msg: People):
        now = time.time()
        if self.first_t is None:
            self.first_t = now

        n_people = len(msg.people)
        self.people_counts[n_people] = self.people_counts.get(n_people, 0) + 1
        if n_people == 0:
            print(f'  {self.n_msgs:5d}  {now - self.first_t:6.2f}  '
                  f'--  0  (empty people array)')
            self.n_msgs += 1
            return

        person = msg.people[0]
        self.names_seen[person.name] = self.names_seen.get(person.name, 0) + 1

        x, y = person.position.x, person.position.y
        vx, vy = person.velocity.x, person.velocity.y
        v_rep = math.hypot(vx, vy)
        hdg_rep = math.degrees(math.atan2(vy, vx)) if v_rep > 1e-6 else float('nan')

        dt = (now - self.last_t) if self.last_t is not None else float('nan')
        v_der = float('nan')
        hdg_der = float('nan')
        flags = []

        if self.last_pos is not None and dt > 1e-6:
            dx, dy = x - self.last_pos[0], y - self.last_pos[1]
            step = math.hypot(dx, dy)
            self.path_length += step
            v_der = step / dt
            self.max_derived_speed = max(self.max_derived_speed, v_der)

            if step < FROZEN_EPSILON_M:
                self.frozen_count += 1
                self.frozen_run += 1
                self.frozen_run_max = max(self.frozen_run_max, self.frozen_run)
                flags.append('FROZEN')
                if v_rep > 0.01:
                    self.frozen_with_velocity += 1
                    flags.append('FROZEN_BUT_CLAIMS_VELOCITY')
            else:
                self.frozen_run = 0
                hdg_der = math.degrees(math.atan2(dy, dx))

            if v_der > self.s['max_vel'] * JUMP_SPEED_FACTOR:
                self.jump_count += 1
                flags.append(f'JUMP({v_der:.1f}m/s)')

            if not math.isnan(v_der):
                self.sum_abs_speed_diff += abs(v_rep - v_der)
                self.speed_samples += 1
                if abs(v_rep - v_der) > SPEED_DISAGREE_TOL:
                    self.speed_disagreements += 1
                    flags.append('SPEED_MISMATCH')

            if not math.isnan(hdg_rep) and not math.isnan(hdg_der):
                if angle_diff_deg(hdg_rep, hdg_der) > HEADING_DISAGREE_TOL:
                    self.heading_disagreements += 1
                    flags.append('HEADING_MISMATCH')

        self.last_pos = (x, y)
        self.last_t = now
        self.n_msgs += 1
        self.min_x, self.max_x = min(self.min_x, x), max(self.max_x, x)
        self.min_y, self.max_y = min(self.min_y, y), max(self.max_y, y)

        radius = self.s['goal_radius']
        for g in self.s['goals']:
            d = math.hypot(x - g['x'], y - g['y'])
            st = self.goal_stats[g['id']]
            st['closest'] = min(st['closest'], d)
            if d <= radius and not st['inside']:
                st['inside'] = True
                st['arrivals'] += 1
                flags.append(f'GOAL{g["id"]}')
            elif d > radius * GOAL_EXIT_MULTIPLIER:
                st['inside'] = False

        if LOG_EVERY_MESSAGE:
            print(f'  {self.n_msgs - 1:5d}  {now - self.first_t:6.2f}  '
                  f'{dt:5.2f}  {n_people:1d}  {person.name[:9]:9s}  '
                  f'({x:6.2f},{y:6.2f})   '
                  f'{v_rep:5.2f}  {v_der:5.2f}   '
                  f'{hdg_rep:7.1f}  {hdg_der:7.1f}   '
                  f'{" ".join(flags)}')

    def print_summary(self):
        print()
        print('=' * 100)
        print('SUMMARY')
        if self.n_msgs == 0:
            print('  No /people messages received at all.')
            print('=' * 100)
            return

        duration = (self.last_t - self.first_t) if self.first_t else 0.0
        rate = self.n_msgs / duration if duration > 0 else 0.0
        print(f'  Messages / duration / rate : {self.n_msgs} over {duration:.1f}s '
              f'= {rate:.2f} Hz')
        print(f'  people[] lengths seen      : {self.people_counts}')
        print(f'  names seen in people[0]    : {self.names_seen}')
        print()
        print('  CONSISTENCY CHECKS')
        print(f'    frozen samples           : {self.frozen_count} '
              f'(longest run {self.frozen_run_max})')
        print(f'    frozen but v is nonzero  : {self.frozen_with_velocity}')
        print(f'    impossible jumps         : {self.jump_count} '
              f'(max derived speed {self.max_derived_speed:.2f} m/s '
              f'vs max_vel {self.s["max_vel"]:.2f})')
        mean_diff = (self.sum_abs_speed_diff / self.speed_samples
                     if self.speed_samples else float("nan"))
        print(f'    reported vs derived speed: {self.speed_disagreements} '
              f'disagreements, mean abs diff {mean_diff:.2f} m/s')
        print(f'    reported vs derived hdg  : {self.heading_disagreements} '
              f'disagreements over {HEADING_DISAGREE_TOL:.0f} deg')
        print()
        print(f'  Distance walked            : {self.path_length:.2f} m')
        print(f'  Area covered               : x [{self.min_x:.2f}, {self.max_x:.2f}]  '
              f'y [{self.min_y:.2f}, {self.max_y:.2f}]')
        for g in self.s['goals']:
            st = self.goal_stats[g['id']]
            print(f'  goal {g["id"]} ({g["x"]:.2f},{g["y"]:.2f}): '
                  f'arrivals={st["arrivals"]:3d}  closest={st["closest"]:.2f}m')
        print()

        bad = (self.jump_count > 0 or self.frozen_with_velocity > 0
               or len(self.names_seen) > 1
               or any(k > 1 for k in self.people_counts))
        if bad:
            print('  VERDICT: pose stream is NOT trustworthy. Likely causes, in order:')
            if any(k > 1 for k in self.people_counts) or len(self.names_seen) > 1:
                print('    - more than one entity is present in /people, so indexing')
                print('      people[0] interleaves two different states into one series.')
            if self.jump_count > 0:
                print('    - displacement per message implies speeds above max_vel,')
                print('      i.e. teleports rather than motion.')
            if self.frozen_with_velocity > 0:
                print('    - state freezes while still reporting nonzero velocity.')
        else:
            print('  VERDICT: pose stream is internally self-consistent.')
            print('           This does NOT yet confirm Gazebo renders the same pose.')
        print('=' * 100)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENARIO_YAML
    try:
        scenario = load_scenario(path)
    except Exception as e:
        print(f'ERROR: could not read scenario from {path}: {e}', file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = PoseConsistencyChecker(scenario)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()