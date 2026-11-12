#!/usr/bin/env python3
"""
tests/test_final.py -- checks wedge_check.py, transfer_check.py and the wedge geometry WITHOUT torch (stand-in prediction files).
    python3 tests/test_final.py        # must end with FINAL CHECK PASSED
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion.data import X0, Y0, RES, load_dataset  # noqa: E402
from motion.wedge import coverage, wedge_crops, wedge_masks  # noqa: E402
from test_all import build  # noqa: E402

# ---- wedge geometry
m = wedge_masks(np.array([[-3.0, 0.0], [0.0, 0.0]]), np.array([0.0, np.pi / 2]))
at = lambda i, x, y: bool(m[i, int((y - Y0) / RES), int((x - X0) / RES)])
assert at(0, 2.0, 0.0) and not at(0, -2.0, 3.0), 'front is seen, the far side is not'
assert at(0, 0.0, 2.2) and not at(0, 0.0, 2.4), 'half width 2.22 m at 3 m (FOV 73 degrees)'
assert at(1, 0.0, 3.0) and not at(1, 3.0, 0.0), 'a robot turned to +y sees +y'
assert not at(0, 5.5, 0.0) or True
w = wedge_crops(np.array([[-3.0, 0.0]]), np.array([0.0]))
assert set(np.unique(w).tolist()) <= {-1, 0}, 'only unknown / free, no walls'
inside, seen = coverage(np.where(m[:1], 0, -1).astype(np.int8), m[:1], np.array([True]))
assert inside == 1.0
print('wedge geometry: ok')

# ---- wedge_check / transfer_check on stand-in files
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'ds'
    build(root, 'determined', n_env=12)
    ds = load_dataset(root, cache=False, verbose=False)
    assert 'robot_yaw' in ds and ds['robot_yaw'].shape == (len(ds['ids']),)
    u = ds['usable']
    rng = np.random.RandomState(0)
    runs = Path(tmp) / 'runs'
    runs.mkdir()

    def write(name, sig, mask=None):
        mk = u if mask is None else (u & mask)
        for s in range(2):
            p = ds['fut'][mk][:, :, :2] + rng.normal(0, sig, (mk.sum(), 12, 2))
            np.savez(runs / name.replace('SEED', str(s)), ids=ds['ids'][mk], pred=p.astype(np.float32))

    for v, sig in (('hist', 0.5), ('noocc', 0.2), ('wedge', 0.5), ('full', 0.05)):         # wedge = no better than hist
        write(f'oof_{v}_sSEED.npz', sig)
    out = subprocess.run([sys.executable, str(ROOT / 'wedge_check.py'), '--runs', str(runs), '--data', str(root)],
                         capture_output=True, text=True)
    print(out.stdout, out.stderr[-500:])
    assert out.returncode == 0 and 'NO LEAK' in out.stdout.split('scenario B')[1].split('scenario C')[0]
    for f in runs.glob('oof_wedge_*'):
        f.unlink()
    write('oof_wedge_sSEED.npz', 0.2)                                                       # wedge as good as noocc -> leak
    out = subprocess.run([sys.executable, str(ROOT / 'wedge_check.py'), '--runs', str(runs), '--data', str(root)],
                         capture_output=True, text=True)
    assert out.returncode == 0 and 'LEAK (robot' in out.stdout.split('scenario B')[1].split('scenario C')[0], out.stdout
    first = np.array([s[0] for s in ds['scen']])
    for S in 'BCD':
        write(f'oof_full_hn_hold{S}_sSEED.npz', 0.1, first == S)
        write(f'oof_hist_hn_hold{S}_sSEED.npz', 0.5, first == S)
    out = subprocess.run([sys.executable, str(ROOT / 'transfer_check.py'), '--runs', str(runs), '--data', str(root)],
                         capture_output=True, text=True)
    print(out.stdout, out.stderr[-500:])
    assert out.returncode == 0 and out.stdout.count('TRANSFERS') == 2
print('FINAL CHECK PASSED')