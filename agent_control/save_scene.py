#!/usr/bin/env python3
"""
save_scene.py -- save the obstacles that are in Gazebo NOW as a scene file
(2026-10-06), so the same scene can be loaded again later with
spawn_obstacles.py.

USAGE (simulation running)
    python3 save_scene.py box_middle          # -> scenes/box_middle.yaml
    python3 save_scene.py                     # just print what Gazebo has now
Reads the same way the bridge does (gazebo_scene.py), so what is saved is
exactly what the pedestrian avoids.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gazebo_scene import read_obstacles, to_yaml  # noqa: E402


def main():
    obs = read_obstacles()
    print(f'{len(obs)} obstacle(s) in Gazebo now:')
    for o in obs:
        print('   ', o.describe())
    if len(sys.argv) < 2:
        print('(not saved: give a name, e.g.  python3 save_scene.py box_middle)')
        return
    name = sys.argv[1][:-5] if sys.argv[1].endswith('.yaml') else sys.argv[1]
    out = HERE / 'scenes' / f'{name}.yaml'
    if out.exists() and '--force' not in sys.argv:
        sys.exit(f'{out} exists. Use another name, or add --force to overwrite it.')
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_yaml(obs))
    print(f'saved -> {out}\nload it again with:  python3 spawn_obstacles.py scenes/{name}.yaml')


if __name__ == '__main__':
    main()