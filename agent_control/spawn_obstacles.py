#!/usr/bin/env python3
"""
spawn_obstacles.py -- load a SAVED SCENE (a yaml of obstacles) into the
running Gazebo world (2026-10-05, updated 2026-10-06).

Since 2026-10-06 Gazebo itself is the truth: the bridge reads whatever is in
the world, so you can also just add boxes from Gazebo's toolbar. This script is
for REPEATING a scene exactly: save_scene.py writes what Gazebo has now into
scenes/<name>.yaml, this script puts it back.

WHAT IT DOES
Removes the models it spawned before (all named obs_<name>) and spawns one
static model per obstacle: box (size_x, size_y, height) or cylinder (radius,
height); 'height' is optional in the yaml (default 1.0 m). The bridge then
sees them in Gazebo like any other model.

USAGE (simulation running)
    python3 spawn_obstacles.py scenes/box_middle.yaml   # load a saved scene
    python3 spawn_obstacles.py --remove                 # remove what it spawned before
Models you added by hand in the GUI are not touched (delete those in the GUI).

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
    ap.add_argument('file', nargs='?', default=None, help='scene yaml, e.g. scenes/box_middle.yaml')
    ap.add_argument('--remove', action='store_true')
    args = ap.parse_args()

    old = STATE_FILE.read_text().split() if STATE_FILE.exists() else []
    if old:
        print(f'Removing {len(old)} previously spawned obstacle model(s)...')
        remove(old)
    STATE_FILE.write_text('')
    if args.remove:
        return
    if not args.file:
        sys.exit('Give a scene file, e.g.  python3 spawn_obstacles.py scenes/box_middle.yaml')

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