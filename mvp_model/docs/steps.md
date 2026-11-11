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
| M5 | look at it + robustness: pictures, damaged maps, history noise, counterfactual edits (H8), seed check, wedge-only leak check (H11), damage-trained model | M5a/M5b done (report received), M5c/M5d/M5e not done |
| M6 | speed on the laptop (H9), final model, replay demo + plots, ideas for the real project, advisor presentation | not done |

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
In the notebook change cell 1 only: `RUN_NAME = 'dmg'`, `VARIANTS = ['full_dmg', 'hist_dmg']`. Then on the laptop: unzip into `runs/run_3`, `python3 make_reports.py` (it now also builds `reports/eval_full_dmg.txt`), and `python3 inspect_model.py --models runs/run_3 --variant full_dmg --out reports/inspect_dmg`.
**Pass rule (fixed now, before the run):** (a) on clean input, `full_dmg` B near and C near ADE@3s are at most 5 % worse than `full` (0.19 / 0.19); (b) with no map, `full_dmg` is not more than 10 % worse than `hist` (0.51 / 0.45); (c) in every map-damage row of its table (shift, wrong cells, rotation), `full_dmg` is better than `hist` on B near and C near. History noise rows are reported, no verdict.

**M5e Wedge-only leak check (H11, one Kaggle run)**: in B, `noocc` already gets most of the gain, so part of it may come from WHERE the robot stands (the grey/known pattern), not from geometry. Control: a model whose crop is only the robot's visibility wedge in an empty world (known inside the camera cone, unknown outside, no walls), built from the recorded robot pose. If it beats `hist` clearly, robot placement leaks the answer. I need one fact first: what `robot_pose_odom` contains (see "what I need from you" in the chat) and the camera field of view.

## M6  Wrap-up and what the MVP delivers (about 2 days)
**M6a Speed on the laptop (H9, information):** milliseconds per prediction and parameter count (282 k) on CPU, PyTorch and ONNX. The Jetson number is NOT part of the MVP (moved to the real project).
**M6b Final model:** one Kaggle run training the chosen version (`full` or `full_dmg`, decided by M5d) on ALL 95 environments. This is the model used in the demo.
**M6c Replay demo (my script `make_demo.py`):** short videos/GIFs of held-out episodes: the map, the history, the truth, and the predictions of CVM, `hist` and the final model, frame by frame. Plus a plot sheet for the advisor: ADE vs time to event, per scenario against CVM, the ablation bars, the wrong-map test, the damage chart.
**M6d Ideas document for the real project** (`docs/real-project-ideas.md`): built from `mvp-to-final-checklist`, the reading notes and the lessons of this MVP (more diverse environments, causes visible earlier, bigger obstacles in A, robot viewpoint coverage, map-confidence fallback, real tracker noise, Jetson speed, MPC integration), with details per item.
**M6e Decision: MPC in the MVP or prediction only?** My recommendation: prediction only. Reason: the data has the robot parked or following 3-5 m behind; in closed loop it approaches, crosses and stops near pedestrians, which the model has never seen, so a closed-loop result now would mostly measure that gap. The closed-loop test goes into the real project with the viewpoint coverage designed into the data. If you want an MPC in the MVP anyway, it needs its own steps (an MPC with CVM prediction as baseline, then the same MPC with this predictor).
**M6f Advisor presentation** (Slides): the story is problem, idea, data, model, hypotheses and verdicts, what works (replay demo, plots, comparison with CVM), honest limits (simulator-defined behaviour, damaged-map risk, transformer edge small), and what the real project will do.
**M6g Write-up:** results and decision into the project and memory; commit code, docs and reports.
Optional, only if time: H10 transfer (train without one scenario, test on it).

## Time plan
M5: 1 to 1.5 days (the two Kaggle runs M5d and M5e are about 20-30 minutes each). M6: 2 days (the ideas document and the presentation are the bulk).
