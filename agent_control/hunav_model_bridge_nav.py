#!/usr/bin/env python3
"""
hunav_model_bridge_nav.py -- hunav_model_bridge_ctrl.py plus OBSTACLES: the
pedestrian sees the obstacles listed in obstacles.yaml and walks around them
(2026-10-05).

WHAT IS NEW COMPARED TO hunav_model_bridge_ctrl.py
  * Every /compute_agents request carries the closest point of each nearby
    obstacle (Agent.closest_obs), like the official HuNav Gazebo plugin does.
    HuNav's social force then keeps the agent away from them at close range.
    (Before, that list was always empty: HuNav saw an empty world, and the
    pedestrian walked through a box.)
  * The bridge plans a ROUTE around the obstacles to each goal (see
    world_obstacles.py) and gives HuNav the route's next corner as its goal,
    one at a time, moving on as soon as the next corner can be seen. Without
    this the social force alone makes the agent stop in front of a box it
    walks straight at, forever.
  * obstacles.yaml is re-read whenever it changes on disk: edit it while the
    simulation runs and the route is re-planned within a second.
  * RViz: /hunav/nav_markers (MarkerArray, frame map) shows the obstacles as
    HuNav knows them (red outlines), the planned route (cyan) and the corner
    the agent is walking to now (cyan ball). Add it once in RViz:
    Add -> By topic -> /hunav/nav_markers -> MarkerArray, then Ctrl+S.

UNCHANGED (agent_control.py works exactly as before)
  subscribes  /hunav/agent_goals  PoseArray, WORLD frame (empty -> yaml goals)
              /hunav/agent_speed  Float32 m/s (<= 0 -> yaml speed)
  publishes   /hunav/current_goals PoseArray, WORLD: the goals the pedestrian
                  is going through (YOUR goals -- not the route corners)
              /hunav/current_speed Float32
The goal cycling (cyclic_goals of the yaml) is now done by this bridge, since
HuNav only ever gets one point at a time.

NEEDS the local HuNav patch (patch_hunav_agent_manager.py), as before.

Lives in my_robot_description/agent_control/, next to obstacles.yaml and
world_obstacles.py. Run exactly like the ctrl bridge (the launch file does it):
    python3 hunav_model_bridge_nav.py --pose-mode topic --ros-args -p use_sim_time:=true
Options:  --obstacles PATH   (default: obstacles.yaml next to this file)
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'hunav_codes'))

from world_obstacles import (CLEARANCE_M, MIN_CLEARANCE_M, OBSTACLE_RANGE_M,  # noqa: E402
                             RouteFollower, closest_points, load_obstacles,
                             min_distance)

DEFAULT_OBSTACLES = os.path.join(_HERE, 'obstacles.yaml')
RELOAD_CHECK_SEC = 1.0
MARKER_PERIOD_SEC = 0.5
SAME_POINT_M = 1e-3


def main():
    import argparse
    import rclpy
    from geometry_msgs.msg import Point, Pose, PoseArray
    from std_msgs.msg import Float32
    from visualization_msgs.msg import Marker, MarkerArray
    from hunav_config import ensure_single_instance, world_to_odom
    from hunav_model_bridge import HunavModelBridge
    from hunav_model_bridge_ctrl import GoalCommander

    class HunavModelBridgeNav(HunavModelBridge):
        def __init__(self, obstacles_path, **kw):
            super().__init__(**kw)
            self.obs_path = obstacles_path
            self.obs_mtime = None
            self.obstacles = []
            self.cmd = None             # GoalCommander: delivers the current corner to HuNav
            self.follower = None        # RouteFollower: user goals -> current corner
            self.user_goals = []        # world frame
            self.yaml_goals = []
            self.cyclic = True
            self.reach = 0.4
            self.sent_target = None
            self.goals_pub = self.create_publisher(PoseArray, '/hunav/current_goals', 10)
            self.speed_pub = self.create_publisher(Float32, '/hunav/current_speed', 10)
            self.marker_pub = self.create_publisher(MarkerArray, '/hunav/nav_markers', 10)
            self.create_subscription(PoseArray, '/hunav/agent_goals', self._goals_cb, 10)
            self.create_subscription(Float32, '/hunav/agent_speed', self._speed_cb, 10)
            self.create_timer(RELOAD_CHECK_SEC, self._reload_obstacles)
            self.create_timer(MARKER_PERIOD_SEC, self._draw)
            self._reload_obstacles()

        def _log(self, lvl, msg):
            # One call line PER LEVEL: rclpy remembers the level used at each
            # line of code and raises "Logger severity cannot be changed between
            # calls" if the same line logs at another level later (2026-10-06 crash).
            lg = self.get_logger()
            if lvl == 'error':
                lg.error(msg)
            elif lvl == 'warn':
                lg.warn(msg)
            elif lvl == 'debug':
                lg.debug(msg)
            else:
                lg.info(msg)

        # ---- obstacles -------------------------------------------------------
        def _reload_obstacles(self):
            try:
                mtime = os.path.getmtime(self.obs_path)
            except OSError:
                mtime = None
            if mtime == self.obs_mtime:
                return
            self.obs_mtime = mtime
            if mtime is None:
                self.obstacles = []
                self.get_logger().warn(f'No obstacle file at {self.obs_path}: the pedestrian sees NO obstacles.')
            else:
                try:
                    self.obstacles = load_obstacles(self.obs_path)
                except Exception as e:      # keep the old list on a typo, say why
                    self.get_logger().error(f'{self.obs_path} not loaded ({e}); keeping the previous obstacles.')
                    return
                self.get_logger().info(
                    f'{len(self.obstacles)} obstacle(s) from {self.obs_path} (world frame):'
                    + ''.join(f'\n    {o.describe()}' for o in self.obstacles))
            if self.follower is not None:
                self.follower.set_obstacles(self.obstacles)
                self._check_goals()

        # ---- goals and speed ----------------------------------------------------
        def initialize_agents(self):
            ok = super().initialize_agents()
            if ok and self.current_agents.agents:
                a = self.current_agents.agents[0]          # single agent, like the rest of the project
                self.yaml_goals = [(g.position.x, g.position.y) for g in a.goals]
                self.cyclic = bool(a.cyclic_goals)
                self.reach = float(a.goal_radius) + 0.1    # HuNav's own "goal reached" distance
                quiet = lambda lvl, m: self._log('error', m) if lvl == 'error' else self.get_logger().debug(m)
                self.cmd = GoalCommander(self.yaml_goals, a.desired_velocity, log=quiet)
                self._new_goals(self.yaml_goals, 'yaml goals')
                self.get_logger().info(
                    f'navigation ready: yaml goals {self.yaml_goals}, speed {a.desired_velocity:.2f} m/s, '
                    f'cyclic {self.cyclic}, agent radius {a.radius:.2f} m, goal reached within {self.reach:.2f} m, '
                    f'obstacle force factor {a.behavior.obstacle_force_factor:.1f}. Route keeps {CLEARANCE_M:.2f} m '
                    f'from obstacles ({MIN_CLEARANCE_M:.2f} m in narrow places).')
            return ok

        def _new_goals(self, goals, label):
            self.user_goals = list(goals)
            self.follower = RouteFollower(self.user_goals, self.obstacles, self.cyclic, self.reach, log=self._log)
            self.sent_target = None
            self.get_logger().info(f'new goal list ({len(goals)}) [{label}]: '
                                   + ', '.join(f'({x:.2f}, {y:.2f})' for x, y in goals))
            self._check_goals()

        def _check_goals(self):
            for i, (x, y) in enumerate(self.user_goals):
                d, o = min_distance(x, y, self.obstacles)
                if d < MIN_CLEARANCE_M:
                    self._log('warn', f'goal {i + 1} ({x:.2f}, {y:.2f}) is {"INSIDE" if d <= 0 else f"only {d:.2f} m from"} '
                                      f'obstacle {o.name}: the pedestrian cannot reach it and will stop in front of '
                                      f'the obstacle. Move the goal or the obstacle (obstacles.yaml).')

        def _goals_cb(self, msg):
            if self.cmd is None:
                return
            goals = [(p.position.x, p.position.y) for p in msg.poses]
            self._new_goals(goals if goals else self.yaml_goals, 'from /hunav/agent_goals' if goals else 'yaml goals')

        def _speed_cb(self, msg):
            if self.cmd is not None:
                self.cmd.command_speed(msg.data)
                self.get_logger().info(f'walking speed -> {self.cmd.effective_speed():.2f} m/s')

        # ---- every HuNav step ---------------------------------------------------
        def _tick(self):
            if self.cmd is not None and self.current_agents is not None and self.current_agents.agents \
                    and not self._compute_agents_pending:
                a = self.current_agents.agents[0]
                px, py = a.position.position.x, a.position.position.y
                target = self.follower.update(px, py) if self.follower else None
                if target is not None and (self.sent_target is None
                                           or math.dist(target, self.sent_target) > SAME_POINT_M):
                    self.cmd.command_goals([target])
                    self.sent_target = target
                goals = self.cmd.goals_for_request()
                if goals is not None:
                    a.goals = []
                    for x, y in goals:
                        p = Pose()
                        p.position.x, p.position.y = float(x), float(y)
                        p.orientation.w = 1.0
                        a.goals.append(p)
                a.desired_velocity = float(self.cmd.speed_for_request())
                a.closest_obs = [Point(x=float(x), y=float(y), z=0.0)
                                 for x, y in closest_points(px, py, self.obstacles, OBSTACLE_RANGE_M)]
            super()._tick()

        def _on_compute_agents_response(self, future):
            super()._on_compute_agents_response(future)
            if self.cmd is None or self.current_agents is None or not self.current_agents.agents:
                return
            a = self.current_agents.agents[0]
            self.cmd.on_response([(g.position.x, g.position.y) for g in a.goals])
            out = PoseArray()
            out.header.stamp = self.current_agents.header.stamp
            out.header.frame_id = 'world'
            for x, y in self.user_goals:
                p = Pose()
                p.position.x, p.position.y = float(x), float(y)
                p.orientation.w = 1.0
                out.poses.append(p)
            self.goals_pub.publish(out)
            self.speed_pub.publish(Float32(data=float(self.cmd.effective_speed())))

        # ---- RViz -----------------------------------------------------------------
        def _draw(self):
            arr = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            now = self.get_clock().now().to_msg()

            def mk(ns, mid, mtype, rgba, scale):
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = now
                m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = rgba
                m.scale.x = m.scale.y = m.scale.z = scale
                return m

            def odom_pt(x, y, z):
                ox, oy, _ = world_to_odom(x, y, 0.0)
                return Point(x=float(ox), y=float(oy), z=z)

            for i, o in enumerate(self.obstacles):
                line = mk('obstacles', i, Marker.LINE_STRIP, (1.0, 0.2, 0.2, 0.9), 0.05)
                outline = o.corners(0.0)
                line.points = [odom_pt(x, y, 0.05) for x, y in outline + outline[:1]]
                arr.markers.append(line)
            if self.follower is not None and self.current_agents is not None and self.current_agents.agents:
                a = self.current_agents.agents[0]
                route = self.follower.remaining_route()
                if route:
                    line = mk('route', 0, Marker.LINE_STRIP, (0.1, 0.8, 1.0, 0.9), 0.04)
                    line.points = [odom_pt(a.position.position.x, a.position.position.y, 0.08)] + \
                                  [odom_pt(x, y, 0.08) for x, y in route]
                    ball = mk('route', 1, Marker.SPHERE, (0.1, 0.8, 1.0, 0.9), 0.2)
                    ball.pose.position = odom_pt(route[0][0], route[0][1], 0.1)
                    arr.markers += [line, ball]
            self.marker_pub.publish(arr)

    ensure_single_instance('hunav_model_bridge')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pose-mode', choices=['set_pose', 'topic'], default='set_pose')
    parser.add_argument('--debug-raw', action='store_true')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES)
    args, _ = parser.parse_known_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = HunavModelBridgeNav(os.path.abspath(os.path.expanduser(args.obstacles)),
                               pose_mode=args.pose_mode, debug_raw=args.debug_raw)
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