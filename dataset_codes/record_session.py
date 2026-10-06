#!/usr/bin/env python3
"""
record_session.py -- one command for a whole recording: record, validate, report.

Start it once my_robot_hunav_full.launch.py is up and the crop viewer shows the
pedestrian walking. It runs, side by side:
    dataset_recorder.py          -> samples into runs/<time>[_name]/samples
    validate_time_alignment.py   -> timing / exclusion / heading measurement
Stop with Ctrl+C (or give --duration). It then stops both cleanly and runs
    inspect_dataset_samples.py   (structure, time checks, anchor spacing)
    analyze_crop_leftover.py     (pedestrian cells left in the crops)
and prints ONE summary with PASS / FAIL per check. Everything (all logs and
the full outputs) is saved in the run folder, report.txt first.

USAGE (from anywhere)
    python3 record_session.py                      # Ctrl+C to stop
    python3 record_session.py --duration 180       # stop after 180 s (wall clock)
    python3 record_session.py --name crossing_3m   # label the run folder
    python3 record_session.py --no-validate        # recorder only (still reports)

Each run gets its own folder, so runs never mix. This script changes nothing in
the other scripts; it only starts them as you would by hand.

PASS criteria (the ones we agreed on 2026-10-04):
    validator  position PASS (p90 <= 0.15 m, residual offset within +-0.1 s),
               exclusion adequate, heading PASS
    inspector  0 samples with issues
    leftover   at most LEFTOVER_LIMIT_PCT % of moving samples with pedestrian cells
    smoothness the walk does not zig-zag (analyze_walk_smoothness.py, if present)
    obstacles  the pedestrian never walks into an obstacle (analyze_obstacle_clearance.py, if present)
    scene      the obstacles did not change while recording

OBSTACLES (2026-10-06): the bridge writes the obstacles it uses to
agent_control/current_obstacles.yaml. This script copies it into the run
folder as obstacles.yaml, so the checks know where the obstacles were. Do not
add/move/delete obstacles while recording: the report flags it.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent                 # dataset_codes
HUNAV_DIR = THIS_DIR.parent / 'hunav_codes'
RUNS_DIR = THIS_DIR / 'runs'

RECORDER = THIS_DIR / 'dataset_recorder.py'
INSPECTOR = THIS_DIR / 'inspect_dataset_samples.py'
LEFTOVER = THIS_DIR / 'analyze_crop_leftover.py'
SMOOTH = THIS_DIR / 'analyze_walk_smoothness.py'
CLEAR = THIS_DIR / 'analyze_obstacle_clearance.py'
VALIDATOR = HUNAV_DIR / 'validate_time_alignment.py'
CURRENT_OBSTACLES = THIS_DIR.parent / 'agent_control' / 'current_obstacles.yaml'


def read_obstacles_text():
    """current_obstacles.yaml without its comment lines (they hold a timestamp), or None."""
    try:
        return '\n'.join(l for l in CURRENT_OBSTACLES.read_text().splitlines() if not l.startswith('#'))
    except OSError:
        return None

LEFTOVER_LIMIT_PCT = 10.0
STARTUP_CHECK_SEC = 6.0        # a child that dies this early failed to start
STOP_TIMEOUT_SEC = 20.0        # time given to each child to write its report after Ctrl+C
PROGRESS_EVERY_SEC = 10.0


def start(cmd, cwd, log_path):
    log = open(log_path, 'w')
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    # own session: the terminal's Ctrl+C reaches only THIS script, which then
    # stops the children itself, in order, and waits for their reports
    # restore normal Ctrl+C handling in the child (it is inherited as 'ignored'
    # when this script itself runs in the background, and then the child could
    # never be asked to stop and print its report)
    p = subprocess.Popen([sys.executable] + [str(c) for c in cmd], cwd=str(cwd), stdout=log,
                         stderr=subprocess.STDOUT, env=env, start_new_session=True,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    return p, log


def stop(proc, name):
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, STOP_TIMEOUT_SEC), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            print(f'  {name} did not stop after {sig.name}, escalating...', flush=True)


def tail(path, n=15):
    try:
        return '\n'.join(Path(path).read_text(errors='replace').splitlines()[-n:])
    except OSError:
        return ''


def last_match(text, pattern):
    m = None
    for m in re.finditer(pattern, text):
        pass
    return m


def run_tool(cmd, cwd):
    r = subprocess.run([sys.executable] + [str(c) for c in cmd], cwd=str(cwd), capture_output=True, text=True)
    return (r.stdout + ('\n' + r.stderr if r.stderr.strip() else '')).strip()


def config_snapshot():
    sys.path.insert(0, str(HUNAV_DIR))
    try:
        import hunav_config as c
        return {k: getattr(c, k, None) for k in (
            'POSE_LOOKUP_OFFSET_SEC', 'PEDESTRIAN_EXCLUSION_RADIUS', 'PAUSE_PHASE_ENABLED',
            'TRAJECTORY_HISTORY_LENGTH', 'FUTURE_HORIZON_LENGTH', 'DATASET_SAMPLE_PERIOD_SEC')}
    except Exception as e:                       # report, never block a recording
        return {'error': f'could not read hunav_config.py: {e}'}


# --- judging the outputs (pure text parsing) ----------------------------------
def judge_validator(text):
    """-> list of (check, status, detail). status: PASS / FAIL / NO DATA"""
    if not text.strip():
        return [('validator', 'NO DATA', 'no report (see validator.log)')]
    if 'Too few usable clouds' in text:
        return [('validator', 'NO DATA', 'too few usable clouds -- record longer, keep the pedestrian in view')]
    out = []
    p90 = last_match(text, r"lookup \(B\) is within ([\d.]+) m|B's 90th-percentile error is ([\d.]+) m")
    off = last_match(text, r'Residual timing offset[^:]*: ([+-][\d.]+) s|constant timing offset ([+-][\d.]+) s')
    detail = []
    if p90:
        detail.append(f'p90 {p90.group(1) or p90.group(2)} m')
    if off:
        detail.append(f'timing offset {off.group(1) or off.group(2)} s')
    out.append(('timing / position', 'PASS' if re.search(r'^\s*PASS:', text, re.M) else 'FAIL', ', '.join(detail)))
    ex = last_match(text, r'EXCLUSION: (.*)')
    out.append(('exclusion', 'PASS' if ex and ex.group(1).startswith('only') else 'FAIL', ex.group(1).strip() if ex else 'no verdict line'))
    hd = last_match(text, r'HEADING: (.*)')
    if hd:
        h = hd.group(1).strip()
        # "too few walking cloud pairs" = nothing to judge (e.g. obstacles near the
        # pedestrian make the validator skip clouds), not a failure
        out.append(('heading', 'PASS' if h.startswith('PASS') else ('NO DATA' if h.startswith('too few') else 'FAIL'), h))
    else:
        out.append(('heading', 'NO DATA', 'too few walking cloud pairs'))
    return out


def judge_inspector(text):
    clean = last_match(text, r'clean: (\d+)')
    bad = last_match(text, r'with issues: (\d+)')
    if not clean or not bad:
        return [('samples (inspector)', 'NO DATA', text.splitlines()[0] if text else 'no output')]
    n_clean, n_bad = int(clean.group(1)), int(bad.group(1))
    sp = last_match(text, r'anchor spacing (median [\d.]+ s, min [\d.]+ s, max [\d.]+ s); breaks \(> 1 s\): (\d+)')
    detail = f'{n_clean} clean, {n_bad} with issues'
    if sp:
        detail += f'; spacing {sp.group(1)}, gaps > 1 s: {sp.group(2)}'
    status = 'NO DATA' if n_clean + n_bad == 0 else ('PASS' if n_bad == 0 else 'FAIL')
    return [('samples (inspector)', status, detail)]


def judge_leftover(text):
    if 'No leftover pedestrian cells' in text:
        return [('leftover pedestrian cells', 'PASS', '0% of moving samples')]
    m = last_match(text, r'leftover present: (\d+)% of moving samples')
    if not m:
        return [('leftover pedestrian cells', 'NO DATA', 'no result (no samples?)')]
    pct = int(m.group(1))
    return [('leftover pedestrian cells', 'PASS' if pct <= LEFTOVER_LIMIT_PCT else 'FAIL',
             f'{pct}% of moving samples (limit {LEFTOVER_LIMIT_PCT:.0f}%)')]


def judge_smoothness(text):
    m = last_match(text, r'WALK SMOOTHNESS: (PASS|FAIL)')
    if not m:
        return [('walk smoothness', 'NO DATA', (text.splitlines() or ['no output'])[-1])]
    f = last_match(text, r'left/right flips\s+: (\d+%)')
    c = last_match(text, r'crop tilt\s+: median ([\d.]+ deg)')
    t = last_match(text, r'turn per 0.5 s step : median [\d.]+ deg, p90 ([\d.]+ deg)')
    detail = ', '.join(x for x in ((f'flips {f.group(1)}' if f else ''), (f'tilt median {c.group(1)}' if c else ''),
                                    (f'turn p90 {t.group(1)}' if t else '')) if x)
    return [('walk smoothness', m.group(1), detail)]


def judge_clearance(text):
    m = last_match(text, r'OBSTACLE CLEARANCE: (PASS|FAIL)')
    if not m:
        return [('obstacle clearance', 'NO DATA', (text.splitlines() or ['no output'])[-1])]
    c = last_match(text, r'crop check\s+: (.*)')
    g = last_match(text, r'ground truth : (.*)')
    detail = ' | '.join(x.group(1).strip() for x in (c, g) if x)
    return [('obstacle clearance', m.group(1), detail)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration', type=float, default=None, help='stop after this many seconds (wall clock)')
    ap.add_argument('--name', default='', help='label added to the run folder name')
    ap.add_argument('--no-validate', action='store_true', help='do not run validate_time_alignment.py')
    args = ap.parse_args()

    for f in [RECORDER, INSPECTOR, LEFTOVER] + ([] if args.no_validate else [VALIDATOR]):
        if not f.exists():
            sys.exit(f'Missing {f}. Put record_session.py in dataset_codes/ next to dataset_recorder.py.')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS_DIR / (stamp + (f'_{args.name}' if args.name else ''))
    samples_dir = run_dir / 'samples'
    samples_dir.mkdir(parents=True)
    cfg = config_snapshot()
    info = {'started': datetime.now().isoformat(timespec='seconds'), 'args': vars(args), 'config': cfg}

    print(f'Run folder: {run_dir}')
    if cfg.get('PAUSE_PHASE_ENABLED'):
        print('  WARNING: PAUSE_PHASE_ENABLED is True in hunav_config.py (debug duty cycle, not for real data).')
    scene_start = read_obstacles_text()
    scene_changes = 0
    if scene_start is None:
        print(f'  NOTE: no {CURRENT_OBSTACLES.name} (is the bridge hunav_model_bridge_nav.py?): '
              f'obstacle checks will not know the obstacles.')
    else:
        (run_dir / 'obstacles_start.yaml').write_text(CURRENT_OBSTACLES.read_text())
    procs = [('recorder', *start([RECORDER, samples_dir], THIS_DIR, run_dir / 'recorder.log'))]
    if not args.no_validate:
        procs.append(('validator', *start([VALIDATOR], HUNAV_DIR, run_dir / 'validator.log')))

    stop_now = {'flag': False, 'count': 0}

    def on_sigint(signum, frame):
        stop_now['count'] += 1
        stop_now['flag'] = True
        if stop_now['count'] > 1:
            print('\n(second Ctrl+C: stopping without waiting for reports)', flush=True)
            for name, p, _ in procs:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    t0 = time.monotonic()
    next_progress = t0 + PROGRESS_EVERY_SEC
    print('Recording' + (f' for {args.duration:.0f} s' if args.duration else '') + ' ... Ctrl+C to stop and get the report.',
          flush=True)
    failed_start = False
    while not stop_now['flag']:
        time.sleep(0.5)
        now = time.monotonic()
        for name, p, _ in procs:
            if p.poll() is not None:
                early = now - t0 < STARTUP_CHECK_SEC
                print(f'\n{name} exited {"right at start-up" if early else "unexpectedly"} (code {p.returncode}). '
                      f'Last lines of {name}.log:\n{tail(run_dir / (name + ".log"))}', flush=True)
                failed_start = failed_start or early
                stop_now['flag'] = True
        if args.duration and now - t0 >= args.duration:
            stop_now['flag'] = True
        cur = read_obstacles_text()
        if cur != scene_start and cur is not None and scene_start is not None:
            scene_changes += 1
            scene_start = cur
            print('  WARNING: the obstacles changed during the recording.', flush=True)
        if now >= next_progress and not stop_now['flag']:
            next_progress += PROGRESS_EVERY_SEC
            n = len(list(samples_dir.glob('*.npz')))
            st = last_match(tail(run_dir / 'recorder.log', 40), r'(\d+ samples saved.*)')
            print(f'  {now - t0:5.0f} s  {n} samples' + (f'  | {st.group(1)[:150]}' if st else ''), flush=True)

    print('\nStopping (waiting for the validator to write its report)...', flush=True)
    for name, p, log in procs:
        stop(p, name)
        log.close()
    info['stopped'] = datetime.now().isoformat(timespec='seconds')
    info['wall_seconds'] = round(time.monotonic() - t0, 1)
    if failed_start:
        print('A process failed to start (see above). Nothing to report.')
        (run_dir / 'run_info.json').write_text(json.dumps(info, indent=2, default=str))
        sys.exit(1)

    if CURRENT_OBSTACLES.exists():
        (run_dir / 'obstacles.yaml').write_text(CURRENT_OBSTACLES.read_text())
    print('Running the inspector and the leftover analysis...', flush=True)
    rec_log = (run_dir / 'recorder.log').read_text(errors='replace')
    val_log = '' if args.no_validate else (run_dir / 'validator.log').read_text(errors='replace')
    val_report = val_log[val_log.find('TIME-ALIGNMENT VALIDATION'):] if 'TIME-ALIGNMENT VALIDATION' in val_log else ''
    ins_out = run_tool([INSPECTOR, samples_dir], THIS_DIR)
    lo_out = run_tool([LEFTOVER, samples_dir], THIS_DIR)
    sm_out = run_tool([SMOOTH, samples_dir], THIS_DIR) if SMOOTH.exists() else ''
    cl_out = run_tool([CLEAR, samples_dir], THIS_DIR) if CLEAR.exists() else ''

    checks = ([] if args.no_validate else judge_validator(val_report)) + judge_inspector(ins_out) + judge_leftover(lo_out) + (judge_smoothness(sm_out) if sm_out else []) + (judge_clearance(cl_out) if cl_out else [])
    if (run_dir / 'obstacles.yaml').exists():
        checks.append(('scene unchanged', 'PASS' if scene_changes == 0 else 'FAIL',
                       'obstacles constant during the recording' if scene_changes == 0 else
                       f'obstacles changed {scene_changes}x while recording: obstacle checks use the final scene'))
    rec_status = last_match(rec_log, r'(Stopped\..*|\d+ samples saved, .*)')
    if all(s == 'PASS' for _, s, _ in checks):
        overall = 'PASS'
    elif any(s == 'FAIL' for _, s, _ in checks):
        overall = 'NOT YET'
    else:
        overall = 'PASS, but ' + ', '.join(n for n, s, _ in checks if s != 'PASS') + ' could not be checked (NO DATA)'

    lines = ['=' * 100, f'RECORDING SESSION REPORT   {run_dir.name}   ({info["wall_seconds"]:.0f} s wall clock)',
             f'config: offset {cfg.get("POSE_LOOKUP_OFFSET_SEC")} s, exclusion radius {cfg.get("PEDESTRIAN_EXCLUSION_RADIUS")} m, '
             f'pause phase {cfg.get("PAUSE_PHASE_ENABLED")}', '']
    for name, status, detail in checks:
        lines.append(f'  {status:8s} {name:28s} {detail}')
    lines += ['', f'  recorder: {rec_status.group(1) if rec_status else "no status line (see recorder.log)"}',
              '', f'OVERALL: {overall}', '=' * 100]
    summary = '\n'.join(lines)
    full = '\n\n'.join([summary,
                        '---- validate_time_alignment.py ----\n' + (val_report or '(not run / no report)'),
                        '---- inspect_dataset_samples.py ----\n' + ins_out,
                        '---- analyze_crop_leftover.py ----\n' + lo_out,
                        '---- analyze_walk_smoothness.py ----\n' + (sm_out or '(not found)'),
                        '---- analyze_obstacle_clearance.py ----\n' + (cl_out or '(not found)')])
    (run_dir / 'report.txt').write_text(full + '\n')
    info['overall'] = overall
    info['checks'] = checks
    (run_dir / 'run_info.json').write_text(json.dumps(info, indent=2, default=str))
    print(full)
    print(f'\nSaved: {run_dir / "report.txt"}  (samples in {samples_dir})')
    sys.exit(0 if overall.startswith('PASS') else 2)


if __name__ == '__main__':
    main()