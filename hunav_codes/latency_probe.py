#!/usr/bin/env python3
"""
latency_probe.py -- how old is each message by the time it arrives, and
which clock is it stamped with?

WHY THIS EXISTS
In RViz the depth point cloud (and so the occupancy grid) visibly trails
Gazebo -- the pedestrian is walking one way in Gazebo while its cloud is
still coming back the other way -- while the /people-based heading arrow
keeps up with Gazebo. That points to data of DIFFERENT ages being combined.
Before fixing anything, this measures, for every stage of the pipeline:

  age = (current time) - (the message's own header.stamp)

i.e. how old the moment the message describes is when it reaches a
subscriber. Rising age from one stage to the next shows where delay builds
up (e.g. a node that cannot keep up and works through a backlog).

It also reports WHICH CLOCK each topic is stamped with:
  SIM  = Gazebo simulation time (/clock, starts near 0)
  WALL = real computer time (seconds since 1970)
Data stamped on different clocks cannot be matched by timestamp at all.

And it prints the real-time factor (RTF = sim seconds per real second): the
simulation is slower than real time when RTF < 1, so a delay of X sim
seconds looks like X / RTF seconds on screen.

USAGE (with the simulation running)
    python3 latency_probe.py            # all depth/cloud/grid/people/pose topics
    python3 latency_probe.py <regex>    # only topics whose name matches
Prints a summary every 5 s. Stop with Ctrl+C (NOT Ctrl+Z).
"""
import re
import statistics
import sys
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

# Message types worth timing. Images only if the topic name says "depth"
# (colour images are large and not used by the pipeline).
WATCH_TYPES = {
    'sensor_msgs/msg/PointCloud2',
    'sensor_msgs/msg/Image',
    'nav_msgs/msg/OccupancyGrid',
    'people_msgs/msg/People',
    'geometry_msgs/msg/PoseStamped',
    'geometry_msgs/msg/PoseArray',
    'nav_msgs/msg/Odometry',
}
REPORT_PERIOD_SEC = 5.0
WALL_STAMP_THRESHOLD = 1e9   # stamps later than 2001 are wall-clock stamps


class TopicStats:
    def __init__(self):
        self.ages = []
        self.arrivals = []
        self.clock = None


class LatencyProbe(Node):
    def __init__(self, name_filter):
        super().__init__('latency_probe',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.filter = re.compile(name_filter) if name_filter else None
        self.stats = {}
        self.subs = []
        self.rtf_ref = None  # (sim_sec, wall_sec) at the previous report
        wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)  # timers keep running even if /clock stalls
        self.create_timer(2.0, self._discover, clock=wall_clock)
        self.create_timer(REPORT_PERIOD_SEC, self._report, clock=wall_clock)
        print('latency_probe: discovering topics...', flush=True)

    def _discover(self):
        for name, types in self.get_topic_names_and_types():
            if name in self.stats or not types:
                continue
            t = types[0]
            if t not in WATCH_TYPES:
                continue
            if t == 'sensor_msgs/msg/Image' and 'depth' not in name:
                continue
            if self.filter and not self.filter.search(name):
                continue
            try:
                cls = get_message(t)
            except (AttributeError, ModuleNotFoundError, ValueError):
                continue
            self.stats[name] = TopicStats()
            self.subs.append(self.create_subscription(
                cls, name, lambda msg, n=name: self._cb(n, msg), qos_profile_sensor_data))
            print(f'  watching {name}  [{t}]', flush=True)

    def _cb(self, name, msg):
        st = self.stats[name]
        wall = time.time()
        st.arrivals.append(wall)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            st.clock = 'ZERO'
            return
        if stamp > WALL_STAMP_THRESHOLD:
            st.clock = 'WALL'
            st.ages.append(wall - stamp)
        else:
            st.clock = 'SIM'
            now = self.get_clock().now().nanoseconds * 1e-9
            if now > 0.0:
                st.ages.append(now - stamp)

    def _report(self):
        wall = time.time()
        sim = self.get_clock().now().nanoseconds * 1e-9
        rtf_txt = 'n/a (no /clock yet)'
        if sim > 0.0:
            if self.rtf_ref is not None and wall > self.rtf_ref[1]:
                rtf = (sim - self.rtf_ref[0]) / (wall - self.rtf_ref[1])
                rtf_txt = f'{rtf:.2f}'
            self.rtf_ref = (sim, wall)
        print('\n' + '=' * 104)
        print(f'sim time {sim:8.2f} s   real-time factor {rtf_txt}   '
              f'(age units: seconds of the topic\'s own clock)')
        print(f'{"topic":52s} {"Hz":>6s} {"clock":>5s} {"age med":>8s} {"age max":>8s} {"age min":>8s}')
        for name in sorted(self.stats):
            st = self.stats[name]
            recent = [a for a in st.arrivals if wall - a <= REPORT_PERIOD_SEC]
            hz = len(recent) / REPORT_PERIOD_SEC
            if st.ages:
                med = statistics.median(st.ages)
                row = f'{med:8.3f} {max(st.ages):8.3f} {min(st.ages):8.3f}'
            else:
                row = f'{"-":>8s} {"-":>8s} {"-":>8s}'
            print(f'{name:52s} {hz:6.1f} {st.clock or "-":>5s} {row}')
            st.ages.clear()
            st.arrivals = recent
        print('=' * 104, flush=True)


def main():
    name_filter = sys.argv[1] if len(sys.argv) > 1 else None
    rclpy.init()
    node = LatencyProbe(name_filter)
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