#!/usr/bin/env python3
"""
check_dataset.py -- one command that (A) checks a whole collected dataset and (B) describes it (2026-11-07).

It reads, for every environment, the NEWEST PASSED run (report says OVERALL: PASS) and all its samples, and writes
ONE text report (default: <dataset>/dataset_report.txt). Nothing is changed in the dataset.

  A  INTEGRITY  (PASS / WARN / FAIL per line): coverage per scenario, sample fields/shapes/dtypes, NaN, the
                pedestrian frame (last history row = 0), time grid (0.5 s), episode membership and the kept time
                window, duplicates, physically possible label speeds, and the ground-truth label never lying in an
                occupied crop cell. The last check runs 4 crop orientations: the stated one must be clean and the
                three mirrored ones are the control (if a mirrored one is just as clean, the check proves nothing).
  B  DESCRIPTION: samples / usable per scenario, robot mode, pedestrian speed, time inside the episode, how many
                samples have the event in their 6 s future, a constant-velocity baseline (CVM) with and without the
                event in the future, crop content (occupied/unknown share, empty and repeated crops), how much the
                future bends or stops, balance between environments, templates.

  --sheets   also writes PNG contact sheets for the eye check into <dataset>/eyecheck/:
             sheet_<scenario>.png  12 samples: 6 usable with the event in the future, 3 usable without, 3 with the
                                   largest CVM error. Crop (black = occupied, grey = unknown, white = free),
                                   blue = history, red = future (the label), green = pedestrian, cyan = robot.
             strip_<scenario>.png  ONE episode, all its samples in time order (is the crop consistent from sample
                                   to sample, does the label match what the crop shows).
             index.txt             which tile is which sample file (open it in the viewer).

USAGE (from mvp_collection/)
    python3 check_scripts/check_dataset.py datasets/mvp_v1 --sheets
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mvp_config as C  # noqa: E402

REQUIRED = ['trajectory_history', 'future_target', 'crop', 'ped_anchor_odom', 'robot_pose_odom', 'history_times',
            'future_times', 'anchor_time_sim', 'ped_seen_anchor', 'ped_seen_history', 'ped_seen_future', 'gated',
            'schema', 'episode_id']
DT = 0.5
N_HIST, N_FUT = 8, 12
CROP_RES = 0.1
X0, Y0 = -2.0, -4.0            # crop origin: columns = x from -2 m, rows = y from -4 m (0.1 m cells, 80 x 80)
LINES = []


def out(s=''):
    print(s)
    LINES.append(s)


def verdict(level, name, text):
    out(f'  {level:5s}  {name:34s} {text}')
    return level


def run_passed(run_dir):
    try:
        return bool(re.search(r'OVERALL: PASS', (Path(run_dir) / 'report.txt').read_text()))
    except OSError:
        return False


def pct(a, b):
    return 100.0 * a / b if b else 0.0


# ---------------------------------------------------------------------------------------------------------- loading
def load_dataset(root):
    scenarios = {}
    for sc in sorted(p for p in root.iterdir() if p.is_dir()):
        envs = sorted(p for p in sc.iterdir() if (p / 'environment.json').exists())
        if envs:
            scenarios[sc.name] = envs
    if not scenarios:
        sys.exit(f'No scenario folders with environments found in {root}')
    return scenarios


def flatten(d, prefix=''):
    """Numeric entries of a nested dict -> {'a.b': value}; booleans and lists are skipped."""
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flatten(v, prefix + k + '.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            o[prefix + k] = float(v)
    return o


def load_env(env_dir):
    """-> (info dict, list of sample dicts). info['status'] explains an environment without passed run."""
    env = json.loads((env_dir / 'environment.json').read_text())
    info = {'name': env_dir.name, 'template': env.get('template', '?'), 'n_obst': len(env.get('obstacles', [])),
            'eps': {e['episode_id']: e for e in env['episodes']}, 'status': 'ok', 'run': None, 'meas': {}}
    info['params'] = flatten(env.get('params', {}))
    runs = [r for r in sorted((env_dir / 'runs').glob('*')) if r.is_dir() and run_passed(r)] \
        if (env_dir / 'runs').is_dir() else []
    n_failed = len(list((env_dir / 'runs_failed').glob('*'))) if (env_dir / 'runs_failed').is_dir() else 0
    info['n_failed'] = n_failed
    info['failed_eps'] = []                      # episodes of the newest failed run: (id, outcome, t_end)
    if n_failed:
        fr = sorted(p for p in (env_dir / 'runs_failed').glob('*') if p.is_dir())[-1]
        for f in sorted((fr / 'episodes').glob('*.json')):
            try:
                m = json.loads(f.read_text()).get('measured', {})
                info['failed_eps'].append((f.stem, m.get('outcome', '?'), m.get('t_end_after_walk_start_sec')))
            except (OSError, ValueError):
                pass
    if not runs:
        info['status'] = f'no passed run ({n_failed} failed run(s) in runs_failed/)'
        return info, []
    run = runs[-1]
    info['run'] = run
    info['n_passed_runs'] = len(runs)
    for f in sorted((run / 'episodes').glob('*.json')):
        try:
            info['meas'][f.stem] = json.loads(f.read_text()).get('measured', {})
        except (OSError, ValueError):
            pass
    samples = []
    for f in sorted((run / 'samples').glob('*.npz')):
        try:
            d = np.load(f, allow_pickle=True)
            s = {'file': f, 'keys': set(d.files)}
            for k in REQUIRED:
                if k in d.files:
                    s[k] = d[k]
        except Exception as e:  # noqa: BLE001
            s = {'file': f, 'keys': set(), 'error': str(e)}
        samples.append(s)
    return info, samples


# -------------------------------------------------------------------------------------------------- crop / geometry
def crop_hits(crop, xy, flip_x, flip_y):
    """(inside, occupied, known) counts of the points xy (ped frame) in the crop for one orientation."""
    n_in = n_occ = n_known = 0
    for x, y in xy:
        c = int(np.floor((x - X0) / CROP_RES))
        r = int(np.floor((y - Y0) / CROP_RES))
        if flip_x:
            c = 79 - c
        if flip_y:
            r = 79 - r
        if 0 <= c < 80 and 0 <= r < 80:
            n_in += 1
            v = crop[r, c]
            n_known += v != -1
            n_occ += v == 100
    return n_in, n_occ, n_known


def robot_in_ped_frame(ped, rob):
    ax, ay, ayaw = [float(v) for v in ped[:3]]
    dx, dy = float(rob[0]) - ax, float(rob[1]) - ay
    c, s = np.cos(ayaw), np.sin(ayaw)
    return c * dx + s * dy, -s * dx + c * dy


def cvm(hist, k_back):
    """Constant-velocity prediction in the pedestrian frame from the last k_back history steps -> (12, 2)."""
    v = (hist[-1, :2] - hist[-1 - k_back, :2]) / (DT * k_back)
    return np.array([v * DT * (i + 1) for i in range(N_FUT)])


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='dataset folder, e.g. datasets/mvp_v1')
    ap.add_argument('--out', default=None, help='report file (default: <dataset>/dataset_report.txt)')
    ap.add_argument('--sheets', action='store_true', help='also write the PNG contact sheets (eye check)')
    ap.add_argument('--seed', type=int, default=0, help='random seed for the sheet selection')
    args = ap.parse_args()
    root = Path(args.dataset).resolve()
    scenarios = load_dataset(root)

    out('=' * 110)
    out(f'DATASET REPORT  {root}')
    out(f'usable rule: ped seen at the anchor and >= {C.USABLE_MIN_HISTORY_SEEN} of {N_HIST} history steps seen')
    out('=' * 110)

    # ---------------------------------------------------------------------------------------------- load everything
    rows = []                                  # one dict per sample (compact)
    env_infos = {}                             # scenario -> list of info
    problems = defaultdict(list)               # check name -> list of strings
    crops = []
    for sc, envs in scenarios.items():
        env_infos[sc] = []
        for ed in envs:
            info, samples = load_env(ed)
            env_infos[sc].append(info)
            for s in samples:
                if 'error' in s or not set(REQUIRED) <= s['keys']:
                    miss = sorted(set(REQUIRED) - s['keys']) if 'error' not in s else [s['error']]
                    problems['fields'].append(f'{s["file"].name}: {miss[:4]}')
                    continue
                h, f = np.asarray(s['trajectory_history'], float), np.asarray(s['future_target'], float)
                crop = np.asarray(s['crop'])
                if h.shape != (N_HIST, 3) or f.shape != (N_FUT, 3) or crop.shape != (80, 80):
                    problems['shapes'].append(f'{s["file"].name}: {h.shape} {f.shape} {crop.shape}')
                    continue
                if crop.dtype != np.int8:
                    problems['dtype'].append(f'{s["file"].name}: crop {crop.dtype}')
                eid = str(s['episode_id'])
                ep = info['eps'].get(eid)
                meas = info['meas'].get(eid, {})
                t0 = meas.get('t_walk_start_sim')
                t_rel = float(s['anchor_time_sim']) - t0 if t0 is not None else float('nan')
                hist_seen = np.asarray(s['ped_seen_history']).astype(int).ravel()
                anchor_seen = int(s['ped_seen_anchor'])
                usable = anchor_seen == 1 and int((hist_seen == 1).sum()) >= C.USABLE_MIN_HISTORY_SEEN
                t_ev, t_end = meas.get('t_event_after_walk_start_sec'), meas.get('t_end_after_walk_start_sec')
                ev_in_future = (t_ev is not None and t_rel < t_ev <= t_rel + N_FUT * DT)
                rob = robot_in_ped_frame(s['ped_anchor_odom'], s['robot_pose_odom'])
                rows.append(dict(scen=sc, env=info['name'], ep=eid, file=s['file'], run=info['run'],
                                 idx=len(crops), t=t_rel, t_ev=t_ev, t_end=t_end, usable=usable,
                                 anchor_seen=anchor_seen, ev_future=ev_in_future, hist=h, fut=f,
                                 anchor_time=float(s['anchor_time_sim']), schema=str(s['schema']),
                                 hist_t=np.asarray(s['history_times'], float), fut_t=np.asarray(s['future_times'], float),
                                 gated=bool(s['gated']), robot=rob, mode=(ep or {}).get('robot', {}).get('mode', '?'),
                                 known_ep=ep is not None, speed_plan=(ep or {}).get('speed')))
                crops.append(crop)
    if not rows:
        sys.exit('No samples loaded.')
    crops = np.stack(crops)
    n_total = len(rows)

    # ================================================================================== A  INTEGRITY
    out()
    out('A  INTEGRITY')
    levels = []

    # A1 coverage
    out('  -- coverage: environments with a passed run')
    for sc, infos in env_infos.items():
        ok = [i for i in infos if i['run'] is not None]
        bad = [f'{i["name"]} ({i["status"]})' for i in infos if i['run'] is None]
        multi = [i['name'] for i in ok if i.get('n_passed_runs', 1) > 1]
        txt = f'{sc}: {len(ok)} of {len(infos)} environments have a passed run'
        if bad:
            txt += '; missing: ' + ', '.join(bad)
        if multi:
            txt += f'; several passed runs (newest used): {", ".join(multi)}'
        levels.append(verdict('PASS' if not bad else 'WARN', 'coverage', txt))

    # A2 fields / shapes
    for key, label in (('fields', 'sample fields'), ('shapes', 'shapes'), ('dtype', 'crop dtype')):
        if problems[key]:
            levels.append(verdict('FAIL', label, f'{len(problems[key])} sample(s), e.g. {problems[key][0]}'))
        else:
            levels.append(verdict('PASS', label, f'all {n_total} samples have all {len(REQUIRED)} fields' if key == 'fields'
                                  else 'all as expected'))
    schemas = Counter(r['schema'] for r in rows)
    levels.append(verdict('PASS' if list(schemas) == ['time_aligned_v2'] else 'FAIL', 'schema', str(dict(schemas))))

    # A3 finite
    bad = [r['file'].name for r in rows if not (np.isfinite(r['hist']).all() and np.isfinite(r['fut']).all())]
    levels.append(verdict('PASS' if not bad else 'FAIL', 'NaN / inf', 'none' if not bad else f'{len(bad)} sample(s), e.g. {bad[0]}'))
    vals = np.unique(crops)
    levels.append(verdict('PASS' if set(vals.tolist()) <= {-1, 0, 100} else 'FAIL', 'crop values', f'{vals.tolist()}'))

    # A4 pedestrian frame
    dev = np.array([np.abs(r['hist'][-1]).max() for r in rows])
    levels.append(verdict('PASS' if dev.max() < 1e-3 else 'FAIL', 'pedestrian frame (last hist row 0)', f'max |deviation| {dev.max():.5f}'))

    # A5 time grid
    dh = np.array([np.abs(np.diff(r['hist_t']) - DT).max() for r in rows])
    df = np.array([np.abs(r['fut_t'] - r['anchor_time'] - DT * np.arange(1, N_FUT + 1)).max() for r in rows])
    dl = np.array([abs(r['hist_t'][-1] - r['anchor_time']) for r in rows])
    ok = max(dh.max(), df.max()) <= 0.06
    levels.append(verdict('PASS' if ok else 'WARN', 'time grid (0.5 s steps)',
                          f'history step error max {dh.max():.3f} s, future step error max {df.max():.3f} s, '
                          f'last history time vs anchor max {dl.max():.3f} s'))

    # A6 episode membership + kept window
    unk = [r for r in rows if not r['known_ep']]
    levels.append(verdict('PASS' if not unk else 'FAIL', 'samples inside episodes',
                          'every sample belongs to an episode of its environment' if not unk
                          else f'{len(unk)} sample(s) with an unknown episode id, e.g. {unk[0]["file"].name} ({unk[0]["ep"]})'))
    early = [r for r in rows if r['t'] < C.START_MARGIN_SEC - 0.75]
    late = [r for r in rows if r['t_end'] is not None and r['t'] + N_FUT * DT > r['t_end'] + 0.75]
    levels.append(verdict('PASS' if not early and not late else 'WARN', 'kept time window',
                          f'before walk start + {C.START_MARGIN_SEC:.1f} s: {len(early)}; future reaching beyond the '
                          f'episode end: {len(late)}'))

    # A7 duplicates
    c = Counter((r['env'], round(r['anchor_time'], 2)) for r in rows)
    dup = sum(1 for v in c.values() if v > 1)
    levels.append(verdict('PASS' if not dup else 'FAIL', 'duplicate samples', f'{dup} anchor time(s) used twice in one environment'))

    # A8 label speeds
    sp_f = np.array([np.linalg.norm(np.diff(np.vstack([[0, 0], r['fut'][:, :2]]), axis=0), axis=1).max() / DT for r in rows])
    sp_h = np.array([np.linalg.norm(np.diff(r['hist'][:, :2], axis=0), axis=1).max() / DT for r in rows])
    worst = max(sp_f.max(), sp_h.max())
    n_hi = int(((sp_f > 2.0) | (sp_h > 2.0)).sum())
    lvl = 'PASS' if worst <= 2.0 else ('WARN' if worst <= 4.0 and pct(n_hi, n_total) < 1 else 'FAIL')
    levels.append(verdict(lvl, 'label / history speed', f'max step speed {worst:.2f} m/s; samples above 2.0 m/s: {n_hi} '
                                                      f'(walking speeds are planned at 0.8 .. 1.5 m/s)'))

    # A9 label never in an occupied cell (+ orientation control)
    combos = {(False, False): 'as stated (x right, y up)', (False, True): 'y mirrored', (True, False): 'x mirrored',
              (True, True): 'both mirrored'}
    res = {}
    for combo in combos:
        tot_in = tot_occ = 0
        for r in rows:
            n_in, n_occ, _ = crop_hits(crops[r['idx']], r['fut'][:, :2], *combo)
            tot_in += n_in
            tot_occ += n_occ
        res[combo] = (tot_in, tot_occ)
    rate = {k: pct(v[1], v[0]) for k, v in res.items()}
    stated = rate[(False, False)]
    controls = {combos[k]: rate[k] for k in rate if k != (False, False)}
    best_other = min(res[k][1] for k in res if k != (False, False))          # fewest hits among the mirrored crops
    n_stated = res[(False, False)][1]
    # PASS when the stated crop is clean (<= 1 %) AND every mirrored crop has clearly more label points in obstacles
    # (at least 20 points and 10 times as many): then the check can tell the orientation and the stated one is right.
    separated = best_other >= max(20, 10 * n_stated)
    lvl = 'PASS' if stated <= 1.0 and separated else ('WARN' if stated <= 1.0 else 'FAIL')
    levels.append(verdict(lvl, 'label never in an obstacle',
                          f'{n_stated} of {res[(False, False)][0]} future points in the crop are in an occupied cell '
                          f'= {stated:.2f} % (limit 1 %). Mirrored crops (control): ' +
                          ', '.join(f'{combos[k]} {res[k][1]} points = {rate[k]:.2f} %' for k in rate if k != (False, False))))
    if lvl == 'WARN':
        out('         (a mirrored crop is nearly as clean, so this check cannot tell the orientation; not a data problem)')
    flip = min(rate, key=rate.get)           # the orientation the plots use (the cleanest one)
    if flip != (False, False):
        out(f'         NOTE: the cleanest orientation is "{combos[flip]}", not the stated one -- plots use it; tell Claude.')

    # A10 episode outcomes
    outc = Counter()
    for infos in env_infos.values():
        for i in infos:
            for eid, m in i['meas'].items():
                outc[m.get('outcome', '?')] += 1
    bad_o = {k: v for k, v in outc.items() if k not in ('ok', 'ok_late_end')}
    levels.append(verdict('PASS' if not bad_o else 'FAIL', 'episode outcomes', str(dict(outc))))

    # A11 samples per episode / environment
    per_ep = Counter((r['env'], r['ep']) for r in rows)
    expected = sum(len(i['eps']) for infos in env_infos.values() for i in infos if i['run'] is not None)
    empty = expected - len(per_ep)
    levels.append(verdict('PASS' if empty == 0 else 'WARN', 'samples per episode',
                          f'{len(per_ep)} of {expected} episodes have samples; min {min(per_ep.values())}, '
                          f'median {int(np.median(list(per_ep.values())))}, max {max(per_ep.values())}'))

    # A12 usable share
    n_use = sum(r['usable'] for r in rows)
    share = pct(n_use, n_total)
    levels.append(verdict('PASS' if share >= 60 else 'WARN', 'usable share', f'{n_use} of {n_total} samples = {share:.1f} % (target >= 60 %)'))
    gated = pct(sum(r['gated'] for r in rows), n_total)
    levels.append(verdict('PASS' if gated >= 99 else 'WARN', 'start gate flag', f'{gated:.1f} % of the samples recorded with the gate open'))

    out()
    out(f'  A summary: {levels.count("PASS")} PASS, {levels.count("WARN")} WARN, {levels.count("FAIL")} FAIL'
        f'   ->  {"DATA OK" if "FAIL" not in levels else "NOT OK -- see the FAIL lines"}')

    # ================================================================================== B  DESCRIPTION
    out()
    out('B  DESCRIPTION')

    def by_scen():
        d = defaultdict(list)
        for r in rows:
            d[r['scen']].append(r)
        return d

    S = by_scen()
    out('  -- B1 size per scenario')
    out(f'  {"scenario":10s} {"envs":>5s} {"episodes":>8s} {"samples":>8s} {"usable":>7s} {"usable%":>8s} {"anchor seen%":>13s} '
        f'{"event in future%":>17s} {"samples/ep":>11s}')
    tot = [0, 0, 0, 0]
    for sc, rs in S.items():
        envs_ok = len({r['env'] for r in rs})
        eps = len({(r['env'], r['ep']) for r in rs})
        u = sum(r['usable'] for r in rs)
        ev = sum(r['ev_future'] for r in rs)
        out(f'  {sc:10s} {envs_ok:5d} {eps:8d} {len(rs):8d} {u:7d} {pct(u, len(rs)):7.1f}% {pct(sum(r["anchor_seen"] == 1 for r in rs), len(rs)):12.1f}% '
            f'{pct(ev, len(rs)):16.1f}% {len(rs) / eps:11.1f}')
        for i, v in zip(range(4), (envs_ok, eps, len(rs), u)):
            tot[i] += v
    out(f'  {"TOTAL":10s} {tot[0]:5d} {tot[1]:8d} {tot[2]:8d} {tot[3]:7d} {pct(tot[3], tot[2]):7.1f}%')

    out()
    out('  -- B2 robot mode (watch = parked, follow = drives behind the pedestrian)')
    out(f'  {"scenario":10s} {"mode":8s} {"samples":>8s} {"usable%":>8s} {"robot dist to ped: median (p10 .. p90) m":>42s}')
    for sc, rs in S.items():
        for mode in sorted({r['mode'] for r in rs}):
            m = [r for r in rs if r['mode'] == mode]
            dist = np.array([np.hypot(*r['robot']) for r in m])
            out(f'  {sc:10s} {mode:8s} {len(m):8d} {pct(sum(r["usable"] for r in m), len(m)):7.1f}% '
                f'{np.median(dist):12.2f} ({np.percentile(dist, 10):.2f} .. {np.percentile(dist, 90):.2f})')

    out()
    out('  -- B3 pedestrian speed at the anchor (from the last history step), share of samples')
    bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.1), (1.1, 1.5), (1.5, 9)]
    out(f'  {"scenario":10s} ' + ' '.join(f'{f"{a}..{b}":>10s}' for a, b in bins) + f' {"median m/s":>11s}')
    for sc, rs in S.items():
        v = np.array([np.linalg.norm(r['hist'][-1, :2] - r['hist'][-2, :2]) / DT for r in rs])
        out(f'  {sc:10s} ' + ' '.join(f'{pct(((v >= a) & (v < b)).sum(), len(v)):9.1f}%' for a, b in bins) + f' {np.median(v):11.2f}')

    out()
    out('  -- B4 where in the episode the anchors are (seconds after walk start), number of samples; '
        'ev = event inside the 6 s future')
    edges = [0, 6, 9, 12, 15, 18, 40]
    out(f'  {"scenario":10s} ' + ' '.join(f'{f"{a}-{b}s":>9s}' for a, b in zip(edges[:-1], edges[1:])) + f' {"ev share by bin":>20s}')
    for sc, rs in S.items():
        t = np.array([r['t'] for r in rs])
        ev = np.array([r['ev_future'] for r in rs])
        cnt = [int(((t >= a) & (t < b)).sum()) for a, b in zip(edges[:-1], edges[1:])]
        evs = [pct(int((ev & (t >= a) & (t < b)).sum()), c) if c else 0 for (a, b), c in zip(zip(edges[:-1], edges[1:]), cnt)]
        out(f'  {sc:10s} ' + ' '.join(f'{c:9d}' for c in cnt) + '   ' + ' '.join(f'{e:3.0f}' for e in evs))

    out()
    out('  -- B5 constant-velocity baseline (CVM) on USABLE samples, metres. ADE = mean error over the 12 future steps, '
        'FDE = error at 6 s')
    out('     CVM-2 uses the last 2 history steps (0.5 s), CVM-4 the last 4 (2 s). "event" = the scenario event lies in the '
        'sample\'s 6 s future.')
    out(f'  {"scenario":10s} {"group":10s} {"n":>6s} {"CVM-2 ADE":>10s} {"CVM-2 FDE":>10s} {"CVM-4 ADE":>10s} {"CVM-4 FDE":>10s}')
    cvm_err = {}
    for sc, rs in S.items():
        for r in rs:
            g = r['fut'][:, :2]
            for k in (1, 3):
                e = np.linalg.norm(cvm(r['hist'], k) - g, axis=1)
                r[f'ade{k}'], r[f'fde{k}'] = float(e.mean()), float(e[-1])
        groups = [('all', [r for r in rs if r['usable']]),
                  ('event', [r for r in rs if r['usable'] and r['ev_future']]),
                  ('no event', [r for r in rs if r['usable'] and not r['ev_future']])]
        for name, m in groups:
            if not m:
                out(f'  {sc:10s} {name:10s} {0:6d}')
                continue
            out(f'  {sc:10s} {name:10s} {len(m):6d} {np.mean([r["ade1"] for r in m]):10.2f} {np.mean([r["fde1"] for r in m]):10.2f} '
                f'{np.mean([r["ade3"] for r in m]):10.2f} {np.mean([r["fde3"] for r in m]):10.2f}')
    out('     Reading: if "event" is not clearly worse than "no event", the scenario does not create situations that '
        'constant velocity gets wrong.')

    out()
    out('  -- B6 how much the future bends or stops (usable samples; ped frame: +x = walking direction, y = left)')
    out(f'  {"scenario":10s} {"|y| at 6 s: median":>19s} {"p90":>6s} {"max":>6s} {"6 s distance: median":>21s} '
        f'{"standing at 6 s*":>17s} {"future turn >45deg*":>20s}')
    for sc, rs in S.items():
        u = [r for r in rs if r['usable']]
        if not u:
            continue
        y = np.array([abs(r['fut'][-1, 1]) for r in u])
        d = np.array([np.hypot(*r['fut'][-1, :2]) for r in u])
        stand = np.mean([np.linalg.norm(r['fut'][-1, :2] - r['fut'][-3, :2]) / (2 * DT) < 0.3 for r in u]) * 100
        turn = np.mean([abs(np.degrees(np.arctan2(np.sin(r['fut'][-1, 2]), np.cos(r['fut'][-1, 2])))) > 45 for r in u]) * 100
        out(f'  {sc:10s} {np.median(y):19.2f} {np.percentile(y, 90):6.2f} {y.max():6.2f} {np.median(d):21.2f} {stand:16.1f}% {turn:19.1f}%')
    out('     * standing at 6 s = speed over the last 1 s of the label below 0.3 m/s; turn = heading at 6 s differs from the '
        'heading now by more than 45 deg.')

    out()
    out('  -- B7 crop content (all samples)')
    out(f'  {"scenario":10s} {"occupied%":>10s} {"unknown%":>9s} {"free%":>7s} {"no occupied cell":>17s} {"same as previous sample":>24s}')
    for sc, rs in S.items():
        idx = np.array([r['idx'] for r in rs])
        c = crops[idx]
        occ, unk_, fre = (c == 100).mean() * 100, (c == -1).mean() * 100, (c == 0).mean() * 100
        none = pct(int(((c == 100).sum(axis=(1, 2)) == 0).sum()), len(c))
        seq = defaultdict(list)
        for r in rs:
            seq[(r['env'], r['ep'])].append(r)
        same = n_pairs = 0
        for k, v in seq.items():
            v.sort(key=lambda r: r['anchor_time'])
            for a, b in zip(v[:-1], v[1:]):
                n_pairs += 1
                same += bool(np.array_equal(crops[a['idx']], crops[b['idx']]))
        out(f'  {sc:10s} {occ:9.1f}% {unk_:8.1f}% {fre:6.1f}% {none:16.1f}% {pct(same, n_pairs):23.1f}%')
    out('     "same as previous sample" = identical crop in the next sample of the same episode (robot parked and the '
        'pedestrian\'s frame moved little).')

    out()
    out('  -- B8 balance between environments (samples per environment) and templates')
    for sc, rs in S.items():
        cnt = Counter(r['env'] for r in rs)
        v = np.array(list(cnt.values()))
        low = sorted(cnt, key=cnt.get)[:2]
        tpl = Counter(i['template'] for i in env_infos[sc] if i['run'] is not None)
        out(f'  {sc:10s} min {v.min():3d} ({low[0]}), median {int(np.median(v)):3d}, max {v.max():3d};  templates: {dict(tpl)};  '
            f'obstacles per environment: mean {np.mean([i["n_obst"] for i in env_infos[sc] if i["run"] is not None]):.1f}')

    out()
    out('  -- B9 pedestrian speed planned per episode (m/s): the speed range the model will see')
    for sc, rs in S.items():
        sp = np.array(sorted({(r['env'], r['ep']): r['speed_plan'] for r in rs if r['speed_plan'] is not None}.values()))
        if len(sp):
            out(f'  {sc:10s} min {sp.min():.2f}  p25 {np.percentile(sp, 25):.2f}  median {np.median(sp):.2f}  p75 {np.percentile(sp, 75):.2f}  max {sp.max():.2f}')

    out()
    out('  -- B6b bend direction of the label (usable samples with |y| at 6 s > 1 m): left / right share')
    for sc, rs in S.items():
        u = [r for r in rs if r['usable'] and abs(r['fut'][-1, 1]) > 1.0]
        if u:
            left = sum(r['fut'][-1, 1] > 0 for r in u)
            out(f'  {sc:10s} n={len(u):5d}   left {pct(left, len(u)):5.1f} %   right {pct(len(u) - left, len(u)):5.1f} %'
                f'   (a mirror augmentation is only needed when one side is rare)')

    out()
    out('  -- B11 can the crop SEE where the event happens? (usable samples with the event in the 6 s future)')
    out('     event point = where the label is at the event time (interpolated). known = cell is not unknown (grey).')
    out(f'  {"scenario":10s} {"n":>6s} {"event point in crop":>20s} {"on a known cell":>16s} {"1 m disk >= 50% known":>22s} '
        f'{"occupied within 1 m":>20s}   disk known by time to event: 0-2 s / 2-4 s / 4-6 s')
    for sc, rs in S.items():
        ev_rows = [r for r in rs if r['usable'] and r['ev_future'] and r['t_ev'] is not None]
        if not ev_rows:
            continue
        n_in = n_known = n_disk = n_occ = 0
        by_tau = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
        for r in ev_rows:
            tau = min(max(r['t_ev'] - r['t'], 0.0), N_FUT * DT)
            pts = np.vstack([[0, 0], r['fut'][:, :2]])
            k = min(int(tau // DT), N_FUT - 1)
            frac = (tau - k * DT) / DT
            p = pts[k] + frac * (pts[k + 1] - pts[k])
            cx, cy = int(np.floor((p[0] - X0) / CROP_RES)), int(np.floor((p[1] - Y0) / CROP_RES))
            if flip[0]:
                cx = 79 - cx
            if flip[1]:
                cy = 79 - cy
            b = min(int(tau // 2), 2)
            by_tau[b][1] += 1
            if not (0 <= cx < 80 and 0 <= cy < 80):
                continue
            n_in += 1
            c = crops[r['idx']]
            n_known += c[cy, cx] != -1
            y0_, y1_, x0_, x1_ = max(cy - 10, 0), min(cy + 11, 80), max(cx - 10, 0), min(cx + 11, 80)
            win = c[y0_:y1_, x0_:x1_]
            yy, xx = np.mgrid[y0_:y1_, x0_:x1_]
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= 100
            good = (win[disk] != -1).mean() >= 0.5
            n_disk += good
            n_occ += bool((win[disk] == 100).any())
            by_tau[b][0] += good
        n = len(ev_rows)
        out(f'  {sc:10s} {n:6d} {pct(n_in, n):19.1f}% {pct(n_known, n):15.1f}% {pct(n_disk, n):21.1f}% {pct(n_occ, n):19.1f}%   '
            + ' / '.join(f'{pct(*by_tau[b]):4.0f} %' for b in range(3)))
    out('     Reading: if the cause of the behaviour (corner, wall, door) is mostly on unknown cells, the crop cannot explain it '
        'and a null result says nothing about the idea. In hallways walls are always within 1 m, so "occupied within 1 m" '
        'proves little; the known-cell columns are the ones that matter.')

    out()
    out('  -- B12 environments without a passed run: what do they share?')
    any_failed = False
    for sc, infos in env_infos.items():
        passed = [i for i in infos if i['run'] is not None]
        for i in infos:
            if i['run'] is not None:
                continue
            any_failed = True
            out(f'  {sc} {i["name"]}: {i["n_failed"]} failed run(s); newest failed run:')
            for eid, o, t_end in i['failed_eps']:
                ep = i['eps'].get(eid, {})
                out(f'      {eid[-3:]:4s} {o:8s} speed {ep.get("speed", 0):.2f} m/s, robot {ep.get("robot", {}).get("mode", "?"):7s}'
                    f' ended at {t_end} s (planned {ep.get("planned", {}).get("t_end_sec", "?")} s)')
            odd = []
            for k, v in sorted(i['params'].items()):
                vals = np.array([q['params'][k] for q in passed if k in q['params']])
                if len(vals) < 5:
                    continue
                rank = pct(int((vals < v).sum()) + 0.5 * int((vals == v).sum()), len(vals))
                if rank <= 10 or rank >= 90:
                    odd.append(f'{k} = {v:.2f} (passed envs: {vals.min():.2f} .. {vals.max():.2f}, this one is at the {rank:.0f}th percentile)')
            out('      unusual parameters: ' + ('; '.join(odd) if odd else 'none (all inside the 10 .. 90 percentile of the passed environments)'))
    if not any_failed:
        out('  none')
    out('     Survivorship: whatever makes an environment fail (slow pedestrian + tight passage?) is missing from the dataset.')

    A_envs = [i for i in env_infos.get('A_detour', []) if i['run'] is not None and 'event_obstacle.across' in i['params']]
    if A_envs:
        out()
        out('  -- B13 scenario A: how much of the hallway does the event obstacle block, and how far does the label bend?')
        ratio, bend = [], []
        for i in A_envs:
            ys = [abs(r['fut'][-1, 1]) for r in S['A_detour'] if r['env'] == i['name'] and r['usable'] and r['ev_future']]
            if ys:
                ratio.append(i['params']['event_obstacle.across'] / i['params']['hall_width'])
                bend.append(np.median(ys))
        ratio, bend = np.array(ratio), np.array(bend)
        if len(ratio) >= 6:
            q = np.percentile(ratio, [33, 66])
            groups = [('small', ratio <= q[0]), ('medium', (ratio > q[0]) & (ratio <= q[1])), ('large', ratio > q[1])]
            out(f'     blocked share of the hallway width (obstacle across / hall width): median {np.median(ratio):.2f}, '
                f'min {ratio.min():.2f}, max {ratio.max():.2f}')
            for name, m in groups:
                out(f'     {name:7s} obstacle ({ratio[m].min():.2f} .. {ratio[m].max():.2f} of the width): {int(m.sum())} environments, '
                    f'median |y| of the label at 6 s = {np.median(bend[m]):.2f} m')
            out('     Reading: a small obstacle in a wide hallway needs almost no detour, so CVM is hardly wrong.')

    out()
    out('  -- B10 suggested split by ENVIRONMENT (never by sample: neighbouring samples are almost the same scene)')
    out('     per scenario the environments in the order of the folder: first 70 % train, next 15 % validation, last 15 % test')
    for sc, infos in env_infos.items():
        names = [i['name'] for i in infos if i['run'] is not None]
        n = len(names)
        n_tr, n_va = int(round(0.7 * n)), int(round(0.15 * n))
        out(f'  {sc:10s} {n} environments -> train {n_tr}, validation {n_va}, test {n - n_tr - n_va}')

    # ================================================================================== eye-check sheets
    if args.sheets:
        make_sheets(root, S, rows, crops, flip, args.seed)

    rep = Path(args.out) if args.out else root / 'dataset_report.txt'
    rep.write_text('\n'.join(LINES) + '\n')
    print(f'\nreport written to {rep}')


# ------------------------------------------------------------------------------------------------------ sheets
def draw_tile(ax, r, crop, flip, title):
    """One sample: crop + history + label + pedestrian + robot, in the pedestrian frame."""
    img = np.where(crop == 100, 0.0, np.where(crop == -1, 0.62, 1.0))
    if flip[1]:
        img = img[::-1, :]
    if flip[0]:
        img = img[:, ::-1]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=[X0, X0 + 8, Y0, Y0 + 8], interpolation='nearest')
    h, f = r['hist'], r['fut']
    ax.plot(h[:, 0], h[:, 1], '-o', color='tab:blue', ms=2.5, lw=1.2)
    ax.plot(np.r_[0, f[:, 0]], np.r_[0, f[:, 1]], '-o', color='tab:red', ms=2.5, lw=1.2)
    ax.plot([0], [0], 'o', color='limegreen', ms=6, mec='k')
    ax.annotate('', xy=(0.7, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='limegreen', lw=1.5))
    rx, ry = r['robot']
    if X0 <= rx <= X0 + 8 and Y0 <= ry <= Y0 + 8:
        ax.plot([rx], [ry], '^', color='cyan', ms=7, mec='k')
    ax.set_xlim(X0, X0 + 8)
    ax.set_ylim(Y0, Y0 + 8)
    ax.set_xticks([-2, 0, 2, 4, 6])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=6.5)


def make_sheets(root, S, rows, crops, flip, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    d = root / 'eyecheck'
    d.mkdir(exist_ok=True)
    index = []
    for sc, rs in S.items():
        u = [r for r in rs if r['usable']]
        if not u:
            continue
        ev = [r for r in u if r['ev_future']]
        noev = [r for r in u if not r['ev_future']]

        def pick(pool, n, used):
            pool = [r for r in pool if id(r) not in used]
            if not pool:
                return []
            sel = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            return [pool[i] for i in sorted(sel)]

        used = set()
        tiles = []
        for r in pick(ev, 6, used):
            used.add(id(r))
            tiles.append(('event in future', r))
        for r in pick(noev, 3, used):
            used.add(id(r))
            tiles.append(('no event', r))
        for r in sorted(u, key=lambda r: -r.get('fde3', 0))[:12]:
            if len([1 for t in tiles if t[0] == 'worst CVM']) >= 3:
                break
            if id(r) not in used:
                used.add(id(r))
                tiles.append(('worst CVM', r))
        for r in pick(u, 12 - len(tiles), used):
            used.add(id(r))
            tiles.append(('random', r))
        fig, axs = plt.subplots(3, 4, figsize=(13, 10.5))
        for ax, (kind, r) in zip(axs.ravel(), tiles):
            draw_tile(ax, r, crops[r['idx']], flip, f'{r["env"][4:]} {r["ep"][-3:]} t={r["t"]:.1f}s {r["mode"]}  [{kind}]\n'
                                                   f'FDE(CVM-4)={r.get("fde3", 0):.1f} m  ev={"yes" if r["ev_future"] else "no"}')
            index.append(f'{sc}\tsheet\t{kind}\t{r["env"]}\t{r["ep"]}\tt={r["t"]:.1f}s\t{r["file"]}')
        for ax in axs.ravel()[len(tiles):]:
            ax.axis('off')
        fig.suptitle(f'{sc}: crop (black occupied, grey unknown, white free), blue = history, red = label (future), '
                     f'green = pedestrian (arrow = walking direction), cyan = robot', fontsize=9)
        fig.tight_layout()
        fig.savefig(d / f'sheet_{sc}.png', dpi=80)
        plt.close(fig)

        # strip: one episode, all samples in time order (an episode with the event and a median number of samples)
        eps = defaultdict(list)
        for r in rs:
            eps[(r['env'], r['ep'])].append(r)
        cands = [k for k, v in eps.items() if any(x['ev_future'] for x in v)] or list(eps)
        med = np.median([len(eps[k]) for k in cands])
        key = sorted(cands, key=lambda k: (abs(len(eps[k]) - med), k))[0]
        seq = sorted(eps[key], key=lambda r: r['anchor_time'])[:20]
        ncol = 5
        nrow = int(np.ceil(len(seq) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(13, 2.7 * nrow + 0.8), squeeze=False)
        for ax, r in zip(axs.ravel(), seq):
            draw_tile(ax, r, crops[r['idx']], flip, f't={r["t"]:.1f}s usable={"y" if r["usable"] else "n"} ev={"y" if r["ev_future"] else "n"}')
            index.append(f'{sc}\tstrip\t{key[1]}\t{r["env"]}\t{r["ep"]}\tt={r["t"]:.1f}s\t{r["file"]}')
        for ax in axs.ravel()[len(seq):]:
            ax.axis('off')
        fig.suptitle(f'{sc}: episode {key[1]} in time order (event at {seq[0]["t_ev"]} s after walk start)', fontsize=9)
        fig.tight_layout()
        fig.savefig(d / f'strip_{sc}.png', dpi=80)
        plt.close(fig)
    (d / 'index.txt').write_text('scenario\tkind\tgroup\tenvironment\tepisode\ttime\tfile\n' + '\n'.join(index) + '\n')
    out()
    out(f'  eye-check sheets written to {d}  (sheet_<scenario>.png, strip_<scenario>.png, index.txt)')


if __name__ == '__main__':
    main()