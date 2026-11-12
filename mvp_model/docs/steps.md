# Steps for the model part (M1-M6)

Status is binary: "done" / "not done". Commit only when a step is truly done (short message, no co-author lines).
Results so far: `docs/results.md`.

## Status
| step | what | status |
|---|---|---|
| M1 | data layer, evaluation, baselines | done |
| M2 | networks run (torch check, smoke runs) | done |
| M3 | main experiment: full vs hist, residual and direct | done (H1-H4, H6 PASS) |
| M4 | ablations: noocc, concat | done (H5 PASS, H7 transformer slightly better) |
| M5 | look at it + robustness: pictures, damaged maps, history noise, counterfactual edits (H8), seed check, wedge-only leak check (H11), damage-trained model | done (M5a-e; `full_hn` PASSED = final model; seed check, H11 and H10 reported) |
| LAST-RUN | one Kaggle notebook: wedge (H11), final models on all data, transfer (H10) | done (H11 no leak, H10 partial transfer) |
| M6 | speed on the laptop (H9), replay demo + plots, ideas for the real project, advisor presentation | scripts ready (speed_test.py, make_demo.py), not run; ideas document and presentation not written |

## Where things run
Laptop: tests, packing, evaluation (`make_reports.py`), pictures, counterfactual and damaged-map tests (they only run the trained models, no training; CPU is enough). Kaggle: anything that trains. Guide: `kaggle-guide.md`.
Trained models of run 1: `runs/run_1/runs/models/<variant>_s<seed>_f<fold>.pt` (keep `results_main.zip` and `results_ablations.zip` as backup).

## Folder layout (`my_robot_description/mvp_model/`)
```
mvp_model/
  motion/            data.py  baselines.py  metrics.py  stats.py  models.py  damage.py
  train.py           training with cross-validation (Kaggle)
  evaluate.py        one evaluation, hypothesis table
  make_reports.py    builds all reports from every run under runs/
  inspect_model.py   M5: pictures, damaged inputs, counterfactual edits (laptop, CPU)
  seed_check.py      is an edge (e.g. full vs concat) stable over seeds and folds? (laptop)
  pack_dataset.py    dataset folder -> one file for Kaggle
  kaggle/            kaggle_train.ipynb
  tests/             test_all.py  test_torch.py  test_inspect.py
  docs/              model-spec.md  evaluation-plan.md  steps.md  kaggle-guide.md  results.md  architecture.png
  runs/  packed/     generated, not committed          reports/  generated, committed
```

---
## Scope decision (2026-11-10): this is the MVP. Everything stays on the laptop and in simulation.
No Jetson, no real robot, no camera work. The real-robot test belongs to the real project. The only open decision is whether the MVP also contains an MPC (closed loop) or only the prediction mechanism (see M6e).

## M5  Look at it and try to break it (about 1 to 1.5 days, laptop + one Kaggle run)
Why now: the model relies strongly on the map (a wrong map is worse than no map). Before we call the MVP done we must know how it behaves with a damaged map and a noisy tracker, and whether it reacts to geometry itself.

**M5a Run `inspect_model.py` on the current `full` models** (my script, tested with stand-in predictors, not yet with your weights):
    cd ~/ros2_ws/src/my_robot_description/mvp_model
    python3 tests/test_inspect.py                   # must print INSPECT CHECK PASSED (no torch needed)
    python3 inspect_model.py --seeds 0 --what pics  # quick look, about 1 minute
    python3 inspect_model.py                        # everything, 10-20 minutes on CPU
It writes `reports/inspect/`: `pics_B.png`, `pics_C.png` (best / typical / worst cases), `damage.png`, `inspect_report.txt`.
First line to check in the report: `self-check ... ok` (it recomputes the stored predictions; if it says TOO BIG the wrong models or data were loaded).
Expect: the cases make sense (turn at the corner, stop before the wall); the damage table is the **"before" table**.
Pass: the files exist and the self-check is ok.

**What the damage table contains** (ADE@3s on B near, C near, A, E; same samples in every row): clean; no map; map shifted 0.2 / 0.5 m; 1 % / 5 % wrong cells; map rotated 5 / 10 degrees; history noise 0.05 / 0.1 / 0.2 m (positions get noise, heading and velocity are recomputed from the noisy points); reference rows cvm4 and hist.

**M5b Counterfactual edits (H8)**, part of the same run. Rules are fixed inside the script docstring (H8a: remove the obstacle at the stop point; H8b: put a wall in the way; both with a control edit). H8b is exploratory.

**M5c Seed check** (`python3 seed_check.py`): the transformer edge over concat is 1.5 cm, about the size of the seed noise. We only keep "full is better than concat" if the sign is the same for every seed and most folds; otherwise the thesis says "not distinguishable".

**M5d Train with damage (UNCONDITIONAL, one Kaggle run, about 25 minutes).** The levels are fixed in `motion/damage_torch.py` (chosen from the M5a table: shift up to 0.3 m, rotation up to 5 degrees, 0-3 % wrong cells, 15 % samples without any map, tracker noise 0 / 0.05 / 0.1 m). Two versions are trained, `full_dmg` and `hist_dmg` (the same networks as `full` and `hist`; only the training inputs are damaged).
In the notebook change cell 1 only: `RUN_NAME = 'dmg'`, `VARIANTS = ['full_dmg', 'hist_dmg']`. Then on the laptop: unzip into `runs/run_3`, `python3 make_reports.py` (it now also builds `reports/eval_full_dmg.txt`), and `python3 inspect_model.py --models runs --variant full_dmg --out reports/inspect_dmg` (`--models runs` on purpose: it finds the model files by name and the clean-trained `hist` predictions of run 1 for the reference rows; under tracker noise the opponent is `hist_dmg`).
**Pass rule (fixed now, before the run):** (a) on clean input, `full_dmg` B near and C near ADE@3s are at most 5 % worse than `full` (0.19 / 0.19); (b) with no map, `full_dmg` is not more than 10 % worse than `hist` (0.51 / 0.45); (c) in every map-damage row of its table (shift, wrong cells, rotation), `full_dmg` is better than `hist` on B near and C near. History noise rows are reported, no verdict.

**M5d result (2026-11-11): NOT PASSED** (clean B 0.31 vs 0.19; no-map C 0.73 vs 0.45). Noise robustness improved a lot, map robustness did not (details in `results.md`). Final model stays `full`.

**M5d2 result (2026-11-11): PASSED.** `full_hn` (tracker noise only): clean B 0.20 / C 0.19 (rule: at most 0.20), under noise 0.1 m better than constant velocity AND `hist_hn` in all four groups, under 0.2 m far better than constant velocity in A and E. **`full_hn` is the final model.** Details in `results.md`. The remaining weakness: with NO map it is worse than before (B 0.66, C 0.97, E 0.21 against constant velocity 0.13), so the real system needs a map-confidence signal and a fallback to `hist_hn` or constant velocity.

**M5e Wedge-only leak check (H11): part of the LAST-RUN notebook (below).** The robot camera is known now (73 degree horizontal FOV, 8 m range, from `depth_camera.xacro`). `pack_dataset.py` also prints how well this camera model fits the real crops ("camera-cone check").

## LAST-RUN: everything that is left on Kaggle, in ONE notebook (about 1 hour, run once)
Preparation on the laptop (about 10 minutes):
1. Replace/add the files from the chat (`motion/data.py`, `motion/wedge.py`, `motion/models.py`, `train.py`, `pack_dataset.py`, `wedge_check.py`, `transfer_check.py`, `make_reports.py`, `tests/test_final.py`, `kaggle/kaggle_final.ipynb`, docs).
2. `python3 tests/test_final.py`, `python3 tests/test_all.py` and `python3 tests/test_torch.py` must pass.
3. `python3 pack_dataset.py` writes the NEW packed file. **Read the line "camera-cone check"** and send it to me with your next message (it says whether the wedge control is a fair test; below 90 % it is weaker).
4. Make the code zip as before and upload code + the new packed file as a new version of your Kaggle dataset.
On Kaggle: import `kaggle/kaggle_final.ipynb`, add the dataset, GPU T4, **Save Version -> Save & Run All**. Close the page. It trains (1) `wedge` (H11), (2) `full_hn` and `hist_hn` on ALL 95 environments, (3) `full_hn` and `hist_hn` with scenario B, C or D held out (H10). Download `results_final.zip`, unzip into `runs/run_5`.
On the laptop:
    python3 wedge_check.py        # H11 verdict, needs the clean-trained runs 1 and 2 in runs/ too
    python3 transfer_check.py     # H10 table (exploratory)
**H11 rule** (fixed in evaluation-plan.md): LEAK if wedge minus hist on B near has its upper bound below 0 and the gain is at least 5 %. **H10:** report only.
The final models are `runs/run_5/runs/models/full_hn_final_s<seed>.pt` and `hist_hn_final_s<seed>.pt` (keep them; the speed test and the real project use them).

## M6  Wrap-up and what the MVP delivers (about 2 days, no more Kaggle runs)
**M6a Speed on the laptop (H9, information), `python3 speed_test.py`:** milliseconds per prediction and parameter count (282 k) on CPU, PyTorch and ONNX, for the final `full_hn`. The Jetson number is NOT part of the MVP.
**M6b Replay demo, `python3 make_demo.py` (GIFs in `reports/demo/`; `python3 tests/test_demo.py` first):** short videos/GIFs of held-out episodes (each sample predicted by the fold model that never saw its environment): the map, the history, the truth, and the predictions of constant velocity, `hist_hn` and `full_hn`, frame by frame. Plus a plot sheet for the advisor: ADE vs time to event, per scenario against constant velocity, the ablation bars, the wrong-map test, the damage chart, the noise chart.
**M6c Ideas document for the real project** (`docs/real-project-ideas.md`): built from `mvp-to-final-checklist`, the reading notes and the lessons of this MVP (more diverse environments, causes visible earlier, bigger obstacles in A, robot viewpoint coverage, a map-confidence signal with fallback, real tracker noise and filtering, map registration accuracy, Jetson speed, MPC integration), with details per item.
**M6d Decision: MPC in the MVP or prediction only?** Recommendation: prediction only (the data has the robot parked or 3-5 m behind; in closed loop it approaches, crosses and stops near pedestrians, which the model has never seen).
**M6e Advisor presentation** (Slides): problem, idea, data, model, hypotheses and verdicts, what works (replay demo, plots, comparison with constant velocity), honest limits (simulator-defined behaviour, needs an accurate map, transformer edge small, no transfer if H10 says so), what the real project will do.
**M6f Write-up:** results and decision into the project and memory; commit code, docs and reports.

## Time plan
LAST-RUN: half a day (10 minutes of preparation, about 1 hour of Kaggle in the background). M6: 2 days (the ideas document and the presentation are the bulk).
