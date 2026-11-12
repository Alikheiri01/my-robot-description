#!/usr/bin/env python3
"""
speed_test.py -- M6a / H9 (information): how long does one prediction take on THIS laptop (CPU)? Run from mvp_model/ :
    python3 speed_test.py                          # final model if found (full_hn_final_s0.pt), else any full_hn file under runs/
    python3 speed_test.py --variant full_hn --models runs --threads 1
Prints parameters, PyTorch CPU time per prediction (batch 1 and batch 32), and ONNX Runtime time if `onnx` and
`onnxruntime` are installed (pip install onnx onnxruntime; optional). Writes reports/speed.txt.
The 5 ms / 1 M parameter targets of H9 are for the real project's Jetson Orin Nano; the laptop number is only a first
indication, NOT the Jetson number. (Written without torch in the sandbox: not run there.)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def bench(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return t.mean(), np.median(t), np.percentile(t, 95)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='full_hn')
    ap.add_argument('--models', default='runs')
    ap.add_argument('--data', default=None)
    ap.add_argument('--threads', type=int, default=0, help='CPU threads (0 = PyTorch default)')
    args = ap.parse_args()
    import torch
    from motion.data import load_dataset
    from motion.models import VARIANTS, count_params
    if args.threads:
        torch.set_num_threads(args.threads)
    files = sorted(Path(args.models).rglob(f'{args.variant}_final_s0.pt')) or sorted(Path(args.models).rglob(f'{args.variant}_s0_f0.pt'))
    if not files:
        sys.exit(f'no {args.variant}_final_s0.pt or {args.variant}_s0_f0.pt under {args.models}')
    model = VARIANTS[args.variant]()
    model.load_state_dict(torch.load(files[0], map_location='cpu'))
    model.eval()
    here = Path(__file__).resolve().parent
    data = args.data or ('packed/mvp_v1_packed.npz' if (here / 'packed' / 'mvp_v1_packed.npz').exists()
                         else '../mvp_collection/datasets/mvp_v1')
    ds = load_dataset(data, verbose=False)
    u = np.nonzero(ds['usable'])[0][:64]
    h, c = torch.tensor(ds['hist'][u]), torch.tensor(ds['crop'][u])
    lines = [f'model {args.variant}  file {files[0]}', f'parameters {count_params(model)}',
             f'CPU threads {torch.get_num_threads()}  torch {torch.__version__}']
    with torch.no_grad():
        m, med, p95 = bench(lambda: model(h[:1], c[:1]))
        lines.append(f'PyTorch, batch 1 : mean {m:.2f} ms   median {med:.2f} ms   95th percentile {p95:.2f} ms')
        m, med, p95 = bench(lambda: model(h[:32], c[:32]), n=50, warm=5)
        lines.append(f'PyTorch, batch 32: mean {m / 32:.2f} ms per prediction')
    try:
        import onnxruntime as ort
        onnx_path = Path('reports') / f'{args.variant}.onnx'
        onnx_path.parent.mkdir(exist_ok=True)
        torch.onnx.export(model, (h[:1], c[:1]), str(onnx_path), input_names=['hist', 'crop'], output_names=['pred'],
                          dynamic_axes={'hist': {0: 'b'}, 'crop': {0: 'b'}, 'pred': {0: 'b'}}, opset_version=17)
        sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        feed = {'hist': h[:1].numpy(), 'crop': c[:1].numpy()}
        ref = model(h[:1], c[:1]).detach().numpy()
        diff = float(np.abs(sess.run(None, feed)[0] - ref).max())
        m, med, p95 = bench(lambda: sess.run(None, feed))
        lines.append(f'ONNX Runtime, batch 1: mean {m:.2f} ms   median {med:.2f} ms   95th percentile {p95:.2f} ms   '
                     f'(max difference to PyTorch {diff:.1e} m)')
    except Exception as e:  # noqa: BLE001
        lines.append(f'ONNX skipped ({type(e).__name__}: {str(e)[:120]}) - pip install onnx onnxruntime to include it')
    text = '\n'.join(lines)
    print(text)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/speed.txt').write_text(text + '\n')


if __name__ == '__main__':
    main()