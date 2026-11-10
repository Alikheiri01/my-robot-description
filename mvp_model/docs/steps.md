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
| M5 | look at it + robustness: pictures, damaged maps, counterfactual edits (H8) | not done |
| M6 | speed (H9), final model on all data, wrap-up and decision | not done |

## Where things run
Laptop: tests, packing, evaluation (`make_reports.py`), pictures, counterfactual and damaged-map tests (they only run the trained models, no training; CPU is enough). Kaggle: anything that trains. Guide: `kaggle-guide.md`.
Trained models of run 1: `runs/run_1/runs/models/<variant>_s<seed>_f<fold>.pt` (keep `results_main.zip` and `results_ablations.zip` as backup).

## Folder layout (`my_robot_description/mvp_model/`)
```
mvp_model/
  motion/            data.py  baselines.py  metrics.py  stats.py  models.py
  train.py           training with cross-validation (Kaggle)
  evaluate.py        one evaluation, hypothesis table
  make_reports.py    builds all reports from every run under runs/
  pack_dataset.py    dataset folder -> one file for Kaggle
  kaggle/            kaggle_train.ipynb
  tests/             test_all.py  test_torch.py
  docs/              model-spec.md  evaluation-plan.md  steps.md  kaggle-guide.md  results.md  architecture.png
  runs/  packed/     generated, not committed          reports/  generated, committed
```

---
## M5  Look at it and test robustness (about 1 day, laptop only)
Why now: the results show the model relies strongly on the map (a wrong map is worse than no map). Before any robot use we must know how it behaves with imperfect maps, and whether it reacts to geometry itself.
I write one script (`inspect_model.py`) that loads the trained `full` models (each sample is predicted by the fold model that never saw its environment) and makes:

**M5a Pictures.** For B and C near events: the crop, history, truth, CVM, hist and full predictions. Best, typical and worst cases. A folder of PNGs like the dataset eye-check sheets.
Expect: good cases turn at the corner / stop before the wall; bad cases show where it fails (probably far events or hidden causes).
Pass: you looked and the cases make sense.

**M5b Damaged maps.** The same test samples with the map: (1) all unknown (no map), (2) shifted by 0.2 / 0.5 m, (3) with random wrong cells (noise 1 % / 5 %), (4) rotated by 5 / 10 degrees.
Expect: a table of ADE@3s per damage type. Guess: "no map" falls to about hist level or worse; small shifts hurt little; big shifts hurt a lot.
Pass: the table exists. Decision: if "no map" or small damage is much worse than `hist`, M5c is needed.

**M5c (only if M5b says so) Train with damaged maps.** One Kaggle run (`full` with random map damage during training), then repeat M5b. About 20 minutes on Kaggle.

**M5d Counterfactual edits (H8).** Change only the map, keep the history: remove the wall in front of a C stop, open/close the corner in B (only on known cells, moving the grey shadow with the wall). Count how often the prediction changes the way physics says (no wall -> no stop, corner closed -> no turn).
Pass: H8 table exists; the rule (share of correct reactions) is fixed in the script before it is run.

## M6  Wrap-up (about 1 day)
**M6a Speed (H9).** Export `full` to ONNX and time it: laptop CPU first, then the Jetson Orin Nano (target at most 5 ms per prediction). I write the export + timing script.
**M6b Final model.** One last Kaggle run: `full` trained on ALL 95 environments (no test fold; the fold results above are the evidence) = the model the robot will use.
**M6c Write-up.** Results and decision into the project and memory; commit code, docs and reports.
**M6d Decision.** Next: closed-loop test in the simulator (MPC with this predictor vs MPC with CVM), or first the final data collection design (lessons in memory: more diversity, causes visible earlier, bigger obstacles in A).

Optional, only if time: H10 transfer (train without one scenario, test on it). Exploratory, one Kaggle run.

## Time plan
M5: 1 day (M5c adds half a day if needed). M6: 1 day.
