#!/usr/bin/env python3
"""
hunav_model_bridge_ctrl.py -- hunav_model_bridge.py plus live control of the
pedestrian's goals and walking speed (2026-10-05).

It IS hunav_model_bridge.py (imported, not copied: same stepping, same pose
delivery, same lock name, so the two can never run together) with three
additions:

  subscribes  /hunav/agent_goals   geometry_msgs/PoseArray, WORLD frame
                  non-empty -> these become the agent's goals, in this order
                  (the agent loops through them if cyclic_goals is true in the
                  yaml; one goal = walk there and stay)
                  empty     -> back to the goals of the yaml
              /hunav/agent_speed   std_msgs/Float32, m/s
                  > 0 -> new walking speed;  <= 0 -> back to the yaml speed
  publishes   /hunav/current_goals PoseArray, WORLD frame, frame_id 'world':
                  the goal list HuNav is really using (from every response)
              /hunav/current_speed Float32: the speed this bridge asks for

agent_control.py (RViz clicks + keys) is the human-friendly front end.

NEEDS the local HuNav patch (patch_hunav_agent_manager.py). Without it HuNav
ignores new goals: this bridge notices (the response never shows them) and
says so after a few tries instead of failing silently. Speed changes are not
visible in the response, so they cannot be confirmed the same way -- watch
hunav_agent_manager's output for "desired velocity set to".

HOW A GOAL CHANGE TRAVELS: the new list is put into the next /compute_agents
request; the patched manager sees a list different from its own and takes it.
From then on this bridge just echoes whatever the manager returns, so HuNav's
own goal cycling continues untouched. The change counts as confirmed when the
returned list holds the same goals (in any order, since the agent may already
have reached the first one).

Lives in my_robot_description/agent_control/ and imports hunav_model_bridge.py and
hunav_config.py from ../hunav_codes. Run exactly like hunav_model_bridge.py
(my_robot_hunav_full.launch.py does it):
    python3 hunav_model_bridge_ctrl.py --pose-mode topic --ros-args -p use_sim_time:=true
"""
import sys
import os as _os
# This file lives in my_robot_description/agent_control/; the shared code
# (hunav_config.py, hunav_model_bridge.py, time_sync.py) is in ../hunav_codes.
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'hunav_codes'))

MAX_ATTEMPTS = 4          # requests carrying a new goal list before giving up on confirmation
MATCH_TOL = 1e-3          # m


def _same_goal_set(a, b, tol=MATCH_TOL):
    """a, b: lists of (x, y). Same goals, order ignored."""
    if len(a) != len(b):
        return False
    rest = list(b)
    for p in a:
        for i, q in enumerate(rest):
            if abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol:
                del rest[i]
                break
        else:
            return False
    return True


class GoalCommander:
    """Pure logic (no ROS): what to put into the next request, and whether HuNav took it."""

    def __init__(self, yaml_goals, yaml_speed, log=print):
        self.yaml_goals = list(yaml_goals)       # [(x, y)] world frame
        self.yaml_speed = float(yaml_speed)
        self.speed = None                        # None = yaml speed (nothing sent)
        self.pending = None                      # [(x, y)] waiting for HuNav to take it
        self.attempts = 0
        self.log = log

    def command_goals(self, goals):
        self.pending = list(goals) if goals else list(self.yaml_goals)
        self.attempts = 0
        self.log('info', f'new goal list ({len(self.pending)}): '
                         + ', '.join(f'({x:.2f}, {y:.2f})' for x, y in self.pending)
                         + (' [yaml goals]' if not goals else ''))

    def command_speed(self, v):
        self.speed = float(v) if v > 0 else None
        self.log('info', f'walking speed -> {self.effective_speed():.2f} m/s'
                         + (' [yaml speed]' if self.speed is None else ''))

    def effective_speed(self):
        return self.speed if self.speed is not None else self.yaml_speed

    def goals_for_request(self):
        """The goal list to put into the next request, or None to echo the last response."""
        if self.pending is None:
            return None
        self.attempts += 1
        return self.pending

    def speed_for_request(self):
        """desired_velocity to put into every request (always set: responses carry 0)."""
        return self.effective_speed()

    def on_response(self, goals):
        """goals: [(x, y)] returned by HuNav. Returns 'confirmed', 'failed' or None."""
        if self.pending is None:
            return None
        if _same_goal_set(goals, self.pending):
            self.pending = None
            self.log('info', f'HuNav is now using the new goals ({len(goals)}).')
            return 'confirmed'
        if self.attempts >= MAX_ATTEMPTS:
            self.pending = None
            self.log('error', 'HuNav ignored the new goals. Is hunav_agent_manager patched '
                              '(patch_hunav_agent_manager.py) and rebuilt?')
            return 'failed'
        return None


def main():
    import argparse
    import rclpy
    from geometry_msgs.msg import Pose, PoseArray
    from std_msgs.msg import Float32
    from hunav_config import ensure_single_instance
    from hunav_model_bridge import HunavModelBridge

    class HunavModelBridgeCtrl(HunavModelBridge):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.cmd = None
            self.goals_pub = self.create_publisher(PoseArray, '/hunav/current_goals', 10)
            self.speed_pub = self.create_publisher(Float32, '/hunav/current_speed', 10)
            self.create_subscription(PoseArray, '/hunav/agent_goals', self._goals_cb, 10)
            self.create_subscription(Float32, '/hunav/agent_speed', self._speed_cb, 10)

        def initialize_agents(self):
            ok = super().initialize_agents()
            if ok and self.current_agents.agents:
                a = self.current_agents.agents[0]          # single agent, like the rest of the project
                goals = [(g.position.x, g.position.y) for g in a.goals]
                self.cmd = GoalCommander(goals, a.desired_velocity,
                                         log=lambda lvl, m: getattr(self.get_logger(), lvl)(m))
                self.get_logger().info(
                    f'goal control ready: yaml goals {goals}, yaml speed {a.desired_velocity:.2f} m/s. '
                    f'Listening on /hunav/agent_goals and /hunav/agent_speed.')
            return ok

        def _goals_cb(self, msg):
            if self.cmd is not None:
                self.cmd.command_goals([(p.position.x, p.position.y) for p in msg.poses])

        def _speed_cb(self, msg):
            if self.cmd is not None:
                self.cmd.command_speed(msg.data)

        def _tick(self):
            if self.cmd is not None and self.current_agents is not None and self.current_agents.agents \
                    and not self._compute_agents_pending:
                a = self.current_agents.agents[0]
                goals = self.cmd.goals_for_request()
                if goals is not None:
                    a.goals = []
                    for x, y in goals:
                        p = Pose()
                        p.position.x, p.position.y = float(x), float(y)
                        p.orientation.w = 1.0
                        a.goals.append(p)
                a.desired_velocity = float(self.cmd.speed_for_request())
            super()._tick()

        def _on_compute_agents_response(self, future):
            super()._on_compute_agents_response(future)
            if self.cmd is None or self.current_agents is None or not self.current_agents.agents:
                return
            a = self.current_agents.agents[0]
            goals = [(g.position.x, g.position.y) for g in a.goals]
            self.cmd.on_response(goals)
            out = PoseArray()
            out.header.stamp = self.current_agents.header.stamp
            out.header.frame_id = 'world'
            out.poses = list(a.goals)
            self.goals_pub.publish(out)
            self.speed_pub.publish(Float32(data=float(self.cmd.effective_speed())))

    ensure_single_instance('hunav_model_bridge')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pose-mode', choices=['set_pose', 'topic'], default='set_pose')
    parser.add_argument('--debug-raw', action='store_true')
    args, _ = parser.parse_known_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = HunavModelBridgeCtrl(pose_mode=args.pose_mode, debug_raw=args.debug_raw)
    if not node.initialize_agents():
        node.get_logger().error('Failed to initialize agents from /get_agents. Exiting.')
        node.destroy_node()
        rclpy.shutdown()
        return
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()