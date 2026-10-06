#!/usr/bin/env python3
"""
spawn_obstacles.py -- put the obstacles of obstacles.yaml into the running
Gazebo world, exactly where the pedestrian thinks they are (2026-10-05).

WHY
The pedestrian avoids what obstacles.yaml lists; the robot's depth camera sees
what is in Gazebo. If the two disagree (a box moved by hand in the GUI, a typo
in the yaml) the pedestrian dodges a box that is not there and walks through
one that is -- and the recording is wrong without any error. Spawning the
Gazebo boxes FROM the yaml makes them agree by construction, and makes every
scenario repeatable (same file -> same scene).

WHAT IT DOES
Removes the models it spawned before (all named obs_<name>) and spawns one
static model per obstacle: box (size_x, size_y, height) or cylinder (radius,
height); 'height' is optional in the yaml (default 1.0 m). Run it again after
every edit of obstacles.yaml (the bridge re-reads the yaml by itself).

USAGE (simulation running)
    python3 spawn_obstacles.py                 # obstacles.yaml next to this file
    python3 spawn_obstacles.py my_scene.yaml   # another file (give the bridge the same one)
    python3 spawn_obstacles.py --remove        # only remove what it spawned

Uses Gazebo Fortress's own command line tool (ign service), world 'empty'
(WORLD_NAME in hunav_config.py).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'hunav_codes'))
from world_obstacles import load_obstacles  # noqa: E402

try:
    from hunav_config import WORLD_NAME
except Exception:
    WORLD_NAME = 'empty'

PREFIX = 'obs_'
DEFAULT_HEIGHT = 1.0
STATE_FILE = HERE / '.spawned_obstacles'      # names spawned last time, for --remove


def ign(service, reqtype, reptype, req):
    cmd = ['ign', 'service', '-s', service, '--reqtype', reqtype, '--reptype', reptype,
           '--timeout', '3000', '--req', req]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and 'true' in r.stdout, (r.stdout + r.stderr).strip()


def heights(path):
    import yaml
    data = yaml.safe_load(Path(path).read_text()) or {}
    return {str(e.get('name', f'obstacle{i + 1}')): float(e.get('height', DEFAULT_HEIGHT))
            for i, e in enumerate(data.get('obstacles') or [])}


def model_sdf(o, h):
    if o.kind == 'box':
        geom = f"<box><size>{2 * o.hx} {2 * o.hy} {h}</size></box>"
    else:
        geom = f"<cylinder><radius>{o.r}</radius><length>{h}</length></cylinder>"
    # single quotes only: the whole thing goes inside a double-quoted protobuf string
    return ("<?xml version='1.0'?><sdf version='1.7'>"
            f"<model name='{PREFIX}{o.name}'><static>true</static>"
            f"<pose>{o.x} {o.y} {h / 2} 0 0 {o.yaw}</pose><link name='link'>"
            f"<collision name='c'><geometry>{geom}</geometry></collision>"
            f"<visual name='v'><geometry>{geom}</geometry><material>"
            "<ambient>0.6 0.4 0.2 1</ambient><diffuse>0.6 0.4 0.2 1</diffuse></material></visual>"
            "</link></model></sdf>")


def remove(names):
    for n in names:
        ok, out = ign(f'/world/{WORLD_NAME}/remove', 'ignition.msgs.Entity', 'ignition.msgs.Boolean',
                      f'name: "{n}" type: MODEL')
        print(f'  removed {n}' if ok else f'  {n}: not removed (already gone?)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='?', default=str(HERE / 'obstacles.yaml'))
    ap.add_argument('--remove', action='store_true')
    args = ap.parse_args()

    old = STATE_FILE.read_text().split() if STATE_FILE.exists() else []
    if old:
        print(f'Removing {len(old)} previously spawned obstacle model(s)...')
        remove(old)
    STATE_FILE.write_text('')
    if args.remove:
        return

    obstacles = load_obstacles(args.file)
    if not obstacles:
        sys.exit(f'No obstacles in {args.file}.')
    h = heights(args.file)
    spawned = []
    for o in obstacles:
        sdf = model_sdf(o, h.get(o.name, DEFAULT_HEIGHT))
        ok, out = ign(f'/world/{WORLD_NAME}/create', 'ignition.msgs.EntityFactory', 'ignition.msgs.Boolean',
                      f'sdf: "{sdf}"')
        if ok:
            spawned.append(PREFIX + o.name)
            print(f'  spawned {PREFIX}{o.name}: {o.describe()}')
        else:
            print(f'  FAILED {o.name}: {out or "no answer (is the simulation running? world name ok?)"}')
    STATE_FILE.write_text('\n'.join(spawned))
    print(f'{len(spawned)}/{len(obstacles)} obstacle(s) in Gazebo, world frame, from {os.path.abspath(args.file)}')


if __name__ == '__main__':
    main()