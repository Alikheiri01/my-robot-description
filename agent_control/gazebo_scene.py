#!/usr/bin/env python3
"""
gazebo_scene.py -- read the obstacles that are in the running Gazebo world
RIGHT NOW (2026-10-06). Gazebo is the truth: whatever you add, move or delete
in Gazebo's GUI is what the pedestrian avoids.

HOW (Gazebo Fortress has no Python API for this, so its command line tool is
used, the same commands you ran by hand):
  * SHAPES: `ign service -s /world/<world>/scene/info` -> every model with its
    links and visuals and their geometry. Its poses are NOT updated when a
    model moves (the robot showed its spawn height there, 2.0 m, while it
    stood on the floor), so it is used for the shapes only.
  * POSES:  one message of `ign topic -e -t /world/<world>/pose/info` -> the
    current pose of every model (matched by entity id).
Each visual with a simple shape (box, cylinder, sphere) of every model that is
not ignored becomes one obstacle: its footprint on the floor, turned by its yaw.
Mesh shapes cannot be measured from here and are skipped with a warning.

Ignored models: the floor, the robot, the pedestrian (IGNORE_MODELS), anything
lying flat on the floor (top below MIN_TOP_Z) or hanging above head height
(bottom above MAX_BOTTOM_Z).

Usable alone:  python3 gazebo_scene.py   -> prints the obstacles Gazebo has now
"""
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'hunav_codes'))
from world_obstacles import Obstacle  # noqa: E402

try:
    from hunav_config import WORLD_NAME, MODEL_NAME
except Exception:
    WORLD_NAME, MODEL_NAME = 'empty', 'hunav_actor'

IGNORE_MODELS = {'ground_plane', 'my_robot', MODEL_NAME, 'hunav_actor', 'pedestrian_standin', 'sun'}
MIN_TOP_Z = 0.10        # m: lower than this = flat on the floor, a person steps over/on it
MAX_BOTTOM_Z = 1.80     # m: higher than this = above head height
MAX_TILT_DEG = 10.0     # footprint is exact only for upright shapes
TIMEOUT_MS = 3000


# ---------------------------------------------------------------------------
# protobuf text format -> nested dicts  (field -> list of values)
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r'\s*(?:([A-Za-z_][\w.]*)|("(?:[^"\\]|\\.)*")|([-+]?[\d.]+(?:[eE][-+]?\d+)?)|([{}:]))')


def parse_text_proto(text):
    pos, n = 0, len(text)
    tokens = []
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == '':
                break
            raise ValueError(f'cannot parse near: {text[pos:pos + 40]!r}')
        pos = m.end()
        ident, string, number, punct = m.groups()
        if ident is not None:
            tokens.append(('id', ident))
        elif string is not None:
            tokens.append(('val', bytes(string[1:-1], 'utf-8').decode('unicode_escape')))
        elif number is not None:
            tokens.append(('val', float(number)))
        else:
            tokens.append(('p', punct))
    i = 0

    def block():
        nonlocal i
        out = {}
        while i < len(tokens):
            kind, v = tokens[i]
            if kind == 'p' and v == '}':
                i += 1
                return out
            if kind != 'id':
                raise ValueError(f'unexpected token {v!r}')
            name = v
            i += 1
            if tokens[i] == ('p', ':'):
                i += 1
                kind2, v2 = tokens[i]
                i += 1
                if kind2 == 'p' and v2 == '{':          # "name: { ... }" is allowed too
                    out.setdefault(name, []).append(block())
                else:
                    out.setdefault(name, []).append(v2)  # enum identifiers stay strings
            elif tokens[i] == ('p', '{'):
                i += 1
                out.setdefault(name, []).append(block())
            else:
                raise ValueError(f'expected : or {{ after {name}')
        return out

    return block()


def first(d, key, default=None):
    v = d.get(key)
    return v[0] if v else default


# ---------------------------------------------------------------------------
# poses
# ---------------------------------------------------------------------------
def pose_of(msg):
    """Pose message dict -> ((x, y, z), (qx, qy, qz, qw))."""
    p = first(msg, 'position', {}) or {}
    o = first(msg, 'orientation', {}) or {}
    return ((first(p, 'x', 0.0), first(p, 'y', 0.0), first(p, 'z', 0.0)),
            (first(o, 'x', 0.0), first(o, 'y', 0.0), first(o, 'z', 0.0), first(o, 'w', 1.0)))


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qrot(q, v):
    x, y, z, w = q
    qv = (v[0], v[1], v[2], 0.0)
    r = qmul(qmul(q, qv), (-x, -y, -z, w))
    return r[:3]


def compose(a, b):
    """world<-a, a<-b  =>  world<-b"""
    (pa, qa), (pb, qb) = a, b
    r = qrot(qa, pb)
    return ((pa[0] + r[0], pa[1] + r[1], pa[2] + r[2]), qmul(qa, qb))


def yaw_tilt(q):
    x, y, z, w = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    up = qrot(q, (0.0, 0.0, 1.0))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
    return yaw, tilt


# ---------------------------------------------------------------------------
# scene -> obstacles
# ---------------------------------------------------------------------------
def obstacles_from(scene, model_poses, warn=print):
    """scene: parsed Scene; model_poses: {entity id: pose} from pose/info
    (missing -> the scene's own, possibly stale, pose). Returns [Obstacle]."""
    out = []
    for model in scene.get('model', []):
        name = first(model, 'name', '?')
        if name in IGNORE_MODELS:
            continue
        mid = first(model, 'id')
        mpose = model_poses.get(mid, pose_of(first(model, 'pose', {}) or {}))
        k = 0
        for link in model.get('link', []):
            lpose = compose(mpose, pose_of(first(link, 'pose', {}) or {}))
            for vis in link.get('visual', []):
                vpose = compose(lpose, pose_of(first(vis, 'pose', {}) or {}))
                geo = first(vis, 'geometry', {}) or {}
                (x, y, z), q = vpose
                yaw, tilt = yaw_tilt(q)
                if 'box' in geo:
                    s = first(first(geo, 'box'), 'size', {}) or {}
                    sx, sy, sz = first(s, 'x', 0.0), first(s, 'y', 0.0), first(s, 'z', 0.0)
                    kind, half_h = 'box', sz / 2
                elif 'cylinder' in geo:
                    c = first(geo, 'cylinder')
                    r, length = first(c, 'radius', 0.0), first(c, 'length', 0.0)
                    kind, half_h = 'cylinder', length / 2
                elif 'sphere' in geo:
                    r = first(first(geo, 'sphere'), 'radius', 0.0)
                    kind, half_h = 'cylinder', r
                elif 'plane' in geo:
                    continue
                else:
                    warn(f'model "{name}": shape {first(geo, "type", "?")} cannot be measured, NOT avoided '
                         f'(use box, cylinder or sphere)')
                    continue
                if z + half_h < MIN_TOP_Z or z - half_h > MAX_BOTTOM_Z:
                    continue
                if tilt > MAX_TILT_DEG:
                    warn(f'model "{name}" is tilted {tilt:.0f} deg: its footprint is approximate')
                k += 1
                oname = name if k == 1 else f'{name}_{k}'
                if kind == 'box':
                    o = Obstacle(oname, 'box', x, y, yaw, sx, sy)
                else:
                    o = Obstacle(oname, 'cylinder', x, y, radius=r)
                o.height = 2 * half_h               # kept for saved scenes
                out.append(o)
    return out


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_MS / 1000 + 2)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError((r.stderr or r.stdout or 'no output').strip()[:200])
    return r.stdout


def fetch_scene(world=WORLD_NAME):
    return parse_text_proto(_run(['ign', 'service', '-s', f'/world/{world}/scene/info',
                                  '--reqtype', 'ignition.msgs.Empty', '--reptype', 'ignition.msgs.Scene',
                                  '--timeout', str(TIMEOUT_MS), '--req', '']))


def fetch_poses(world=WORLD_NAME):
    msg = parse_text_proto(_run(['timeout', str(TIMEOUT_MS / 1000), 'ign', 'topic', '-e',
                                 '-t', f'/world/{world}/pose/info', '-n', '1']))
    return {first(p, 'id'): pose_of(p) for p in msg.get('pose', [])}


def read_obstacles(world=WORLD_NAME, warn=print):
    """The obstacles in Gazebo now. Raises RuntimeError if Gazebo does not answer."""
    scene = fetch_scene(world)
    try:
        poses = fetch_poses(world)
    except Exception as e:                     # shapes are still useful with spawn poses
        warn(f'no live poses from /world/{world}/pose/info ({e}); using the scene poses')
        poses = {}
    return obstacles_from(scene, poses, warn)


def to_yaml(obstacles, strip_prefix='obs_', header='saved from Gazebo by save_scene.py'):
    """Text for an obstacles.yaml with these obstacles (spawn_obstacles.py can load it)."""
    lines = [f'# {header} (world frame)', 'obstacles:']
    for o in obstacles:
        name = o.name[len(strip_prefix):] if strip_prefix and o.name.startswith(strip_prefix) else o.name
        lines += [f'  - name: {name}', f'    type: {o.kind}', f'    x: {o.x:.3f}', f'    y: {o.y:.3f}']
        if o.kind == 'box':
            lines += [f'    yaw: {o.yaw:.4f}', f'    size_x: {2 * o.hx:.3f}', f'    size_y: {2 * o.hy:.3f}']
        else:
            lines += [f'    radius: {o.r:.3f}']
        if getattr(o, 'height', None):
            lines += [f'    height: {o.height:.3f}']
    if not obstacles:
        lines[-1] = 'obstacles: []'
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    obs = read_obstacles()
    print(f'{len(obs)} obstacle(s) in Gazebo world "{WORLD_NAME}" now:')
    for o in obs:
        print('   ', o.describe())