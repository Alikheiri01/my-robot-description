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

CHANGED (2026-11-05, long runs): the whole scene is created with ONE call of Gazebo's
/world/<world>/create_multiple service (a list of models) instead of one create call per
obstacle. Gazebo's create/remove handlers only queue the request, but they share a lock with the
step that carries the queue out, so a burst of 15-20 calls made later calls wait and time out
(the rehearsal had 1-4 of ~20 calls fail per scene, and the retries made it worse). One call
means nothing to queue behind. --one-by-one gives the old behaviour (last retry of the runner).
Removal is still one call per model (Gazebo has no multiple remove), but it now stops asking
after 3 failures in a row (Gazebo is not answering; the rest stays in the state file) instead of
waiting 20 s for each of the remaining models.
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
# CHANGED (2026-11-03): was 3000 ms. Right after a scene change Gazebo can need longer than 3 s to answer;
# the model was then often created anyway, the script reported FAILED and did not remember the name.
IGN_TIMEOUT_MS = 20000
MULTI_TIMEOUT_MS = 60000      # NEW (2026-11-05): one call creates the whole scene
REMOVE_TIMEOUT_MS = 8000      # NEW: a remove either answers at once or Gazebo is busy
REMOVE_GIVE_UP_AFTER = 3      # NEW: failures in a row before the rest is left for the next try


def ign(service, reqtype, reptype, req, timeout_ms=IGN_TIMEOUT_MS):
    cmd = ['ign', 'service', '-s', service, '--reqtype', reqtype, '--reptype', reptype,
           '--timeout', str(timeout_ms), '--req', req]
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
    """Removes the models by name. Returns the names NOT confirmed removed (to try again later)."""
    left = []
    fails = 0
    for i, n in enumerate(names):
        if fails >= REMOVE_GIVE_UP_AFTER:
            left.append(n)
            continue
        ok, out = ign(f'/world/{WORLD_NAME}/remove', 'ignition.msgs.Entity', 'ignition.msgs.Boolean',
                      f'name: "{n}" type: MODEL', timeout_ms=REMOVE_TIMEOUT_MS)
        if ok:
            fails = 0
            print(f'  removed {n}')
        else:
            fails += 1
            left.append(n)
            print(f'  {n}: not removed (Gazebo busy, or already gone)')
    if len(left) > 0 and fails >= REMOVE_GIVE_UP_AFTER:
        print(f'  Gazebo did not answer {REMOVE_GIVE_UP_AFTER} removals in a row: {len(left)} name(s) kept for the next try')
    return left


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='?', default=None, help='scene yaml, e.g. scenes/box_middle.yaml')
    ap.add_argument('--remove', action='store_true')
    ap.add_argument('--one-by-one', action='store_true',
                    help='one create call per obstacle (the old way) instead of one call for the whole scene')
    args = ap.parse_args()

    old = STATE_FILE.read_text().split() if STATE_FILE.exists() else []
    kept = []
    if old:
        print(f'Removing {len(old)} previously spawned obstacle model(s)...')
        kept = remove(old)
    STATE_FILE.write_text('\n'.join(kept))
    if args.remove:
        return
    if not args.file:
        sys.exit('Give a scene file, e.g.  python3 spawn_obstacles.py scenes/box_middle.yaml')

    obstacles = load_obstacles(args.file)
    if not obstacles:
        sys.exit(f'No obstacles in {args.file}.')
    h = heights(args.file)
    spawned = []
    maybe = []     # CHANGED (2026-11-03): calls that got no answer -- the model may exist anyway, so --remove tries them too
    if args.one_by_one:
        for o in obstacles:
            sdf = model_sdf(o, h.get(o.name, DEFAULT_HEIGHT))
            ok, out = ign(f'/world/{WORLD_NAME}/create', 'ignition.msgs.EntityFactory', 'ignition.msgs.Boolean',
                          f'sdf: "{sdf}"')
            if ok:
                spawned.append(PREFIX + o.name)
                print(f'  spawned {PREFIX}{o.name}: {o.describe()}')
            else:
                maybe.append(PREFIX + o.name)
                print(f'  FAILED {o.name}: {out or "no answer (is the simulation running? world name ok?)"}')
    else:
        # NEW (2026-11-05): the whole scene in ONE call (EntityFactory_V: repeated field 'data')
        req = ' '.join('data { sdf: "%s" }' % model_sdf(o, h.get(o.name, DEFAULT_HEIGHT)) for o in obstacles)
        ok, out = ign(f'/world/{WORLD_NAME}/create_multiple', 'ignition.msgs.EntityFactory_V',
                      'ignition.msgs.Boolean', req, timeout_ms=MULTI_TIMEOUT_MS)
        names = [PREFIX + o.name for o in obstacles]
        if ok:
            spawned = names
            for o in obstacles:
                print(f'  spawned {PREFIX}{o.name}: {o.describe()}')
        else:
            maybe = names          # the request may have been queued anyway: --remove must try all of them
            print(f'  FAILED create_multiple ({len(names)} obstacles): '
                  f'{out or "no answer (is the simulation running? world name ok?)"}')
    STATE_FILE.write_text('\n'.join(spawned + maybe + kept))
    print(f'{len(spawned)}/{len(obstacles)} obstacle(s) in Gazebo, world frame, from {os.path.abspath(args.file)}')


if __name__ == '__main__':
    main()