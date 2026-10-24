#!/usr/bin/env python3
"""
measure_walk_stride.py -- how far does the walk animation travel per second of
animation? That decides whether the feet slide.

WHY (2026-10-05)
HuNavActorDriver moves the actor along the ground itself and advances the walk
animation by  distance_walked * <animation_seconds_per_meter>  (default 2.0,
set in hunav_actor.sdf). The feet only stay planted when that matches the
animation's own stride: seconds_per_meter = cycle_duration / distance the
animation's root travels in one cycle. If the value is too small the legs move
too slowly for the ground speed and the body glides ("sliding"); too large and
the legs churn.

This script reads walk.dae (from Gazebo's Fuel cache, or a path you give),
finds the root bone's translation over the animation, and prints the value to
put into hunav_actor.sdf. It changes nothing.

USAGE
    python3 measure_walk_stride.py                 # searches ~/.ignition and ~/.gz for walk.dae
    python3 measure_walk_stride.py /path/to/walk.dae
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def find_dae():
    for base in (Path.home() / '.ignition' / 'fuel', Path.home() / '.gz' / 'fuel'):
        if base.exists():
            hits = sorted(p for p in base.rglob('walk.dae') if 'actor' in str(p).lower())
            if hits:
                return hits[-1]
    return None


def strip(tag):
    return tag.split('}', 1)[-1]


def load_root_translation(path):
    root = ET.parse(path).getroot()
    unit = 1.0
    for el in root.iter():
        if strip(el.tag) == 'unit' and el.get('meter'):
            unit = float(el.get('meter'))
            break
    arrays = {}
    for el in root.iter():
        if strip(el.tag) == 'float_array' and el.get('id') and el.text:
            arrays[el.get('id')] = np.array([float(v) for v in el.text.split()])
    results = []
    for anim in root.iter():
        if strip(anim.tag) != 'animation':
            continue
        sources = {}
        for src in anim:
            if strip(src.tag) == 'source':
                fa = next((c for c in src if strip(c.tag) == 'float_array'), None)
                if fa is not None and fa.get('id') in arrays:
                    sources['#' + src.get('id')] = arrays[fa.get('id')]
        samplers = {}
        for smp in anim:
            if strip(smp.tag) == 'sampler':
                ins = {i.get('semantic'): i.get('source') for i in smp if strip(i.tag) == 'input'}
                samplers['#' + smp.get('id')] = ins
        for ch in anim:
            if strip(ch.tag) != 'channel':
                continue
            ins = samplers.get(ch.get('source'), {})
            t, out = sources.get(ins.get('INPUT')), sources.get(ins.get('OUTPUT'))
            if t is None or out is None or len(t) < 2 or len(out) != 16 * len(t):
                continue                                  # only full 4x4 matrix tracks
            m = out.reshape(len(t), 4, 4)
            trans = m[:, :3, 3] * unit
            results.append((ch.get('target'), t, trans))
    return results


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_dae()
    if path is None or not Path(path).exists():
        sys.exit('walk.dae not found. Give its path: python3 measure_walk_stride.py /path/to/walk.dae\n'
                 '(Gazebo caches it under ~/.ignition/fuel/fuel.gazebosim.org/mingfei/models/actor/...)')
    tracks = load_root_translation(path)
    if not tracks:
        sys.exit(f'No matrix animation tracks found in {path}.')
    # the root is the bone whose translation moves the most from start to end of the clip
    best = max(tracks, key=lambda tr: np.max(np.abs(tr[2][-1] - tr[2][0])))
    target, t, trans = best
    net = trans[-1] - trans[0]
    axis = int(np.argmax(np.abs(net)))
    dist = abs(net[axis])
    dur = float(t[-1] - t[0])
    print(f'file      : {path}')
    print(f'root track: {target}   ({len(t)} keys, {dur:.3f} s)')
    print(f'root moves {dist:.3f} m along axis {"xyz"[axis]} over the clip (other axes: '
          + ', '.join(f'{"xyz"[i]} {net[i]:+.3f}' for i in range(3) if i != axis) + ')')
    if dist < 0.05:
        print('The root hardly moves: this clip is an in-place animation, the stride cannot be read from the root.\n'
              'Send me this output and I will measure it from the foot bones instead.')
        return
    spm = dur / dist
    print(f'\nanimation speed: {dist / dur:.2f} m/s of root motion')
    print(f'=> <animation_seconds_per_meter>{spm:.2f}</animation_seconds_per_meter>   (currently 2.0 if not set)')


if __name__ == '__main__':
    main()