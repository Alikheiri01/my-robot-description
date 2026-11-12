#!/usr/bin/env python3
"""tests/test_video.py -- make_video.py renders a video with stand-in predictors (no torch).  Must end with VIDEO CHECK PASSED."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_video as MV  # noqa: E402
from make_demo import episode_frames, pick_episodes  # noqa: E402
from motion.baselines import cvm  # noqa: E402
from motion.data import load_dataset  # noqa: E402
from test_all import build  # noqa: E402


class Fake:
    seeds = [0]

    def __init__(self, noise):
        self.noise = noise

    def predict(self, h, c, f, robot=None):
        return (cvm(h, 4) + self.noise)[None].astype(np.float32)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'ds'
    build(root, 'determined', n_env=3)
    for ef in root.glob('*/*/environment.json'):                 # give the fake worlds some walls and an event
        e = json.loads(ef.read_text())
        e['obstacles'] = [{'name': 'w', 'role': 'wall', 'type': 'box', 'x': 10, 'y': 2.5, 'yaw': 0.0, 'size_x': 40, 'size_y': 0.2},
                          {'name': 'f', 'role': 'furniture', 'type': 'box', 'x': 8, 'y': -1.5, 'yaw': 0.4, 'size_x': 0.6, 'size_y': 0.5}]
        for ep in e['episodes']:
            ep['event'] = {'type': 'corner', 'position': [12.0, 0.0]}
        ef.write_text(json.dumps(e))
    ds = load_dataset(root, cache=False, verbose=False)
    ep = pick_episodes(ds)[0]
    idx = episode_frames(ds, ep)
    for ext in ('gif', 'mp4'):
        res = MV.render_video(ds, root, ep, idx, Fake(0.0), Fake(0.1), Path(tmp) / f'v.{ext}', fps=4)
        out = Path(tmp) / f'v.{ext}'
        if not out.exists():
            out = out.with_suffix('.gif')
        assert out.stat().st_size > 20000 and res['map'] > 0, out
print('VIDEO CHECK PASSED')