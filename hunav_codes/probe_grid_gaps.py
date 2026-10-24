#!/usr/bin/env python3
"""
probe_grid_gaps.py -- where do the gaps in the occupancy grid stream come from? (2026-10-24)

WHY
Every recording shows "no grid near the sample time" drops, anchor gaps > 1 s
and history steps with no grid near their time. In R1 (robot still, open floor,
pedestrian always visible) that alone made 13% of the samples unusable. The
gaps happen at the same rate with a still robot, a moving robot and clutter,
so they are not caused by the scene. Somewhere in the chain
    Gazebo depth image -> depth_image_proc cloud -> fused cloud -> grid
messages go missing. This probe listens to every stage at once and tells, for
each gap in the grid, at which stage the gap first appears.

WHAT IT DOES (read-only, changes nothing)
Records the header stamp (simulation time) of every message on each stage's
topic. At Ctrl+C it prints, per stage: messages, rate in simulation time,
median / p90 / max spacing, and the number of gaps longer than GAP_SEC. Then,
for every grid gap, whether the earlier stages also had nothing in that window.
The first stage that has the gap is where to look.

Subscribing to the depth IMAGES costs some CPU (they are big); --no-images
skips them if the probe itself seems to slow things down.

USAGE (pipeline running, any scene; 3-5 minutes is enough)
    python3 probe_grid_gaps.py
    python3 probe_grid_gaps.py --no-images
    python3 probe_grid_gaps.py --topics /my/fused/cloud /occupancy_grid/base   (other names)
"""
import argparse
import threading
import time

import numpy as np

GAP_SEC = 0.30          # same as MAX_GRID_MISMATCH_SEC in dataset_recorder.py
GRID_TOPIC = '/occupancy_grid/base'
DEFAULT_STAGES = [
    ('1 depth image L', '/depth_cam/left/depth_image'),
    ('1 depth image R', '/depth_cam/right/depth_image'),
    ('2 cloud L', '/depth_cam/left/points_corrected'),
    ('2 cloud R', '/depth_cam/right/points_corrected'),
    ('3 fused cloud', None),            # found automatically: a PointCloud2 topic with "fuse" in its name
    ('4 grid', GRID_TOPIC),
]


def spacing_stats(t):
    t = np.sort(np.asarray(t))
    if len(t) < 3:
        return None
    d = np.diff(t)
    span = t[-1] - t[0]
    return {'n': len(t), 'rate': (len(t) - 1) / span if span > 0 else float('nan'),
            'median': float(np.median(d)), 'p90': float(np.percentile(d, 90)), 'max': float(d.max()),
            'gaps': int((d > GAP_SEC).sum())}


def gaps_of(t, gap=GAP_SEC):
    t = np.sort(np.asarray(t))
    d = np.diff(t)
    return [(t[i], t[i + 1]) for i in np.nonzero(d > gap)[0]]


def has_gap_over(t, a, b, margin=0.05):
    """True if stream t has NO message strictly inside (a + margin, b - margin)."""
    t = np.asarray(t)
    return not np.any((t > a + margin) & (t < b - margin))


def attribute(streams, grid_name):
    """For each grid gap: the list of earlier stages that also had nothing in that window."""
    out = []
    for a, b in gaps_of(streams[grid_name]):
        also = [name for name, t in streams.items()
                if name != grid_name and len(t) > 2 and min(t) <= a and max(t) >= b and has_gap_over(t, a, b)]
        out.append((a, b, also))
    return out


def report(streams, grid_name, wall_sec, sim_span):
    print('\n' + '=' * 90)
    rtf = sim_span / wall_sec if wall_sec > 0 else float('nan')
    print(f'PIPELINE GAP PROBE   {wall_sec:.0f} s wall, {sim_span:.0f} s simulation (real-time factor {rtf:.2f})')
    print(f'  gap = no message for more than {GAP_SEC:.2f} s of simulation time')
    print(f'  {"stage":18s} {"msgs":>6s} {"rate/sim s":>10s} {"median":>8s} {"p90":>7s} {"max":>7s} {"gaps":>6s}')
    for name, t in streams.items():
        s = spacing_stats(t)
        if s is None:
            print(f'  {name:18s} {len(t):6d}   (too few messages -- wrong topic name?)')
            continue
        print(f'  {name:18s} {s["n"]:6d} {s["rate"]:10.2f} {s["median"]:8.3f} {s["p90"]:7.3f} {s["max"]:7.2f} {s["gaps"]:6d}')
    if grid_name not in streams or spacing_stats(streams[grid_name]) is None:
        print('No grid messages: is the grid builder running?')
        return
    att = attribute(streams, grid_name)
    if not att:
        print('VERDICT: no grid gaps longer than the limit in this run.')
        return
    first = {}
    for _, _, also in att:
        key = min(also) if also else 'only in the grid'
        first[key] = first.get(key, 0) + 1
    print(f'\n  {len(att)} grid gaps. Earliest stage that was ALSO silent during the gap:')
    for k, v in sorted(first.items()):
        print(f'    {k:22s} {v:4d}  = {100.0 * v / len(att):.0f}%')
    print('  first gaps (simulation time, length, also silent):')
    for a, b, also in att[:10]:
        print(f'    {a:9.2f} .. {b:9.2f}  ({b - a:.2f} s)  {", ".join(also) if also else "-"}')
    top = max(first, key=first.get)
    where = {'only in the grid': 'the grid builder itself (its input kept coming): look at build_occupancy_grid_dynamic.py',
             '1 depth image L': 'Gazebo already sends no depth images: the simulation/rendering side (sensor update rate, CPU/GPU load)',
             '1 depth image R': 'Gazebo already sends no depth images: the simulation/rendering side (sensor update rate, CPU/GPU load)',
             '2 cloud L': 'depth_image_proc (the images came, the clouds did not)',
             '2 cloud R': 'depth_image_proc (the images came, the clouds did not)',
             '3 fused cloud': 'fuse_depth_clouds.py (both clouds came, the fused one did not)'}.get(top, top)
    print(f'VERDICT: most grid gaps start at "{top}" -> {where}.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-images', action='store_true', help='do not subscribe to the depth images')
    ap.add_argument('--topics', nargs='*', default=None,
                    help='use these topics instead (in pipeline order, the LAST one is the grid)')
    ap.add_argument('--duration', type=float, default=None, help='stop after this many wall seconds')
    args, _ = ap.parse_known_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message

    rclpy.init()
    node = Node('probe_grid_gaps')
    time.sleep(2.0)                                  # let discovery fill the topic list
    types = dict((n, ts[0]) for n, ts in node.get_topic_names_and_types())
    if args.topics:
        stages = [(f'{i + 1} {t}', t) for i, t in enumerate(args.topics)]
    else:
        stages = []
        for name, topic in DEFAULT_STAGES:
            if topic is None:
                cands = [n for n, ty in types.items() if 'fuse' in n.lower() and ty.endswith('PointCloud2')]
                if not cands:
                    print('  (no fused-cloud topic found automatically; use --topics to give it)')
                    continue
                topic = sorted(cands)[0]
                name = f'3 fused cloud'
                print(f'  fused cloud topic: {topic}')
            if args.no_images and 'image' in topic:
                continue
            stages.append((name, topic))
    grid_name = stages[-1][0]
    streams = {name: [] for name, _ in stages}
    lock = threading.Lock()
    for name, topic in stages:
        if topic not in types:
            print(f'  WARNING: {topic} is not published right now (stage "{name}")')
            continue
        msg_type = get_message(types[topic])

        def cb(msg, name=name):
            st = msg.header.stamp
            with lock:
                streams[name].append(st.sec + st.nanosec * 1e-9)
        node.create_subscription(msg_type, topic, cb, qos_profile_sensor_data)
        print(f'  listening: {name:18s} {topic}')
    print('Ctrl+C for the report (3-5 minutes is enough).')
    t0 = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if args.duration and time.monotonic() - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    wall = time.monotonic() - t0
    with lock:
        data = {k: list(v) for k, v in streams.items()}
    allt = [x for v in data.values() for x in v]
    report(data, grid_name, wall, (max(allt) - min(allt)) if allt else 0.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()