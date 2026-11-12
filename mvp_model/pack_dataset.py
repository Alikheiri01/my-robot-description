#!/usr/bin/env python3
"""
pack_dataset.py -- turns a dataset folder (thousands of small .npz files) into ONE file for Kaggle / Colab.
numpy only. Run from mvp_model/ on your laptop:

    python3 pack_dataset.py                                   # ../mvp_collection/datasets/mvp_v1 -> packed/mvp_v1_packed.npz
    python3 pack_dataset.py PATH_TO_DATASET  OUTPUT_FILE.npz

The file holds everything train.py and evaluate.py need (history, label, crop, flags, ids, the 5 folds), so on Kaggle you
upload ONE file of a few MB instead of the whole mvp_collection. It is checked after writing: it is loaded again and
compared with the original.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion.data import load_dataset  # noqa: E402
from motion.wedge import coverage, wedge_masks  # noqa: E402


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else '../mvp_collection/datasets/mvp_v1'
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('packed') / f'{Path(src).name}_packed.npz'
    ds = load_dataset(src, cache=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **ds)
    back = load_dataset(dst)
    for k, v in ds.items():
        if not np.array_equal(v, back[k], equal_nan=v.dtype.kind == 'f'):
            sys.exit(f'CHECK FAILED: field {k} differs after packing')
    wm = wedge_masks(ds['robot'], ds['robot_yaw'])
    inside, seen = coverage(ds['crop'], wm, ds['usable'])
    print(f'[pack] camera-cone check (wedge control): {inside * 100:.1f} % of the known crop cells lie inside the modelled camera cone; '
          f'{seen * 100:.1f} % of the cone cells are known in the crop')
    if inside < 0.9:
        print('[pack] NOTE: below 90 % -> the real crop contains more than one instant camera view (or the pose/FOV model is off). '
              'The wedge control is then only a weaker test; send me this line.')
    mb = dst.stat().st_size / 1e6
    print(f'[pack] wrote {dst}  ({mb:.1f} MB, {len(ds["ids"])} samples, {int(ds["usable"].sum())} usable) -- check passed')


if __name__ == '__main__':
    main()