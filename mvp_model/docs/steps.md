# Steps for the model part, from now to the end (M1-M6) -- about 3-4 days

Status is binary: "done" / "not done". A step is done only when its pass line is true. Commit only when a step is truly done (short message, no co-author lines, e.g. "added model code and tests").
For every step: **Do** (the commands), **Expect** (what you will see), **Pass**, **If not**. My guesses about results are marked "guess": they are expectations, not promises.

## Where things run
Your laptop: M1 (tests, baselines), packing the data, and all evaluation (numpy, light). **Kaggle (GPU): M2 smoke runs and all training** (M3, M4). Full guide: `kaggle-guide.md`. The training code is device-agnostic (`--device auto` = GPU if there is one, else CPU).

## Folder layout (`my_robot_description/mvp_model/`)
```
mvp_model/
  motion/            code library (imported by the scripts)
    data.py            loads a dataset, folds, mirror
    baselines.py       stand, CVM, history-only ridge
    metrics.py         ADE/FDE @3s/@6s, occupied share
    stats.py           95 % intervals (bootstrap over environments)
    models.py          the transformer and the simple "concat" model, residual and direct styles (PyTorch)
  train.py           trains one version with cross-validation (PyTorch)
  evaluate.py        compares everything, prints the hypothesis table (numpy only)
  tests/
    test_all.py        data layer + evaluation + planted effects + mirror (numpy only)
    test_torch.py      the networks run, mirror in torch (needs torch)
  pack_dataset.py    packs a dataset folder into ONE file for Kaggle/Colab
  kaggle/            kaggle_train.ipynb (notebook for Kaggle and Colab)
  docs/              model-spec.md  evaluation-plan.md  steps.md  kaggle-guide.md  architecture.png
  runs/              generated: predictions, trained weights, logs, cache   (not committed)
  packed/            generated: the packed data file   (not committed)
  reports/           generated: evaluation reports
```
The versions: `full`, `hist`, `noocc`, `concat` (residual style = correction on top of CVM) and the same four with `_direct` (the network outputs the path itself). Residual is the primary family, direct the secondary comparison.

---
## M1  Data layer, evaluation and baselines work (day 1, ~30 min)
**Do** (from `mvp_model/`):
```
python3 tests/test_all.py
python3 evaluate.py
```
**Expect**
- First command, about 10-30 s: a line "mirror: twice = identity ... ok", then `[determined] H1 B: PASS ... H4 B: PASS`, `[noise] H1 B: FAIL ...`, a hypothesis table, and `ALL TESTS PASSED`.
- Second command, about 1 min the first time (reads all samples, makes a cache in `runs/`): a report with tables E1-E6 and the columns `stand`, `cvm2`, `cvm4`, `ridge_hist`. No hypothesis lines yet (no model).
- E2 (ADE@6s), `cvm4`, close to what you have seen: event vs no event A 0.31/0.32, B 1.67/0.89, C 1.66/0.22, D 0.60/0.33, E 0.30 (within about 0.05).
- E1 (ADE@3s) is smaller than E2 everywhere; near events in B and C are clearly worse than the no-event rows.
- E4: the label's occupied share is about 0.000; the predictors' values are small.
- Guess: `ridge_hist` is equal to or slightly better than `cvm4`. If it already removes 30 % or more of the error on B/C near events, the history explains a lot and the map has less to add; that is useful to know.

**Pass:** `ALL TESTS PASSED` and `reports/eval_report.txt` exists with the numbers above.
**If not:** paste the error or the report to me.
**Send me:** `reports/eval_report.txt`.

## M2  The networks run (day 1, ~1 h)
**Do:**
```
python3 tests/test_torch.py
python3 train.py --variant full --seeds 0 --epochs 2 --fold 0
python3 train.py --variant hist --seeds 0 --epochs 2 --fold 0
python3 train.py --variant full_direct --seeds 0 --epochs 2 --fold 0
python3 train.py --variant noocc --seeds 0 --epochs 2 --fold 0
python3 train.py --variant concat --seeds 0 --epochs 2 --fold 0
```
**Expect**
- `test_torch.py`: one line per version (10 lines) with the parameter count (about 0.3 M for the transformer versions, similar for `concat`), "equals CVM at init: ok" for residual versions, "finite, not CVM-based: ok" for direct ones, then the mirror and token checks and `TORCH CHECK PASSED`.
- Each smoke run: `[train] variant ... device ...`, one line `seed 0 fold 0: train ~5000 val ~600 test ~1700 ... test ADE@6s <number>`, `wrote runs/oof_...`. A 2-epoch result is poor: the number only has to be finite and in the range of 0.5-3 m. Not a result.
- Write down the time of one smoke run: it sets the plan for M3 (a full run = 15 trainings per version, each up to 60 epochs, usually stopping earlier).

**Pass:** `TORCH CHECK PASSED` and five smoke runs finish with a finite ADE.
**If not:** paste the error. This is the step where untested code usually needs a small fix; that is normal.
**Decision here:** if one smoke epoch takes much more than about 20 s on your machine, tell me and we cut seeds from 3 to 2 or use `--patience 6`.

## M3  Main experiment: with the map vs without (day 2)
**Do:**
```
python3 train.py --variant full
python3 train.py --variant hist
python3 evaluate.py --test full --ref hist \
  --preds full=runs/oof_full_s0.npz,runs/oof_full_s1.npz,runs/oof_full_s2.npz \
  --preds hist=runs/oof_hist_s0.npz,runs/oof_hist_s1.npz,runs/oof_hist_s2.npz \
  --preds full_shufscn=runs/oof_full_s0_shufscn.npz,runs/oof_full_s1_shufscn.npz,runs/oof_full_s2_shufscn.npz \
  --shuf-scn full_shufscn
```
Then the direct pair (M3b, can run overnight): `--variant full_direct` and `--variant hist_direct`, and the same evaluate command with `full_direct`/`hist_direct` names.
**Expect**
- Training prints 15 lines per version (3 seeds x 5 folds) with the validation and test ADE per fold. Fold numbers differ from each other (a few hundredths of a metre to 0.1 m): that is normal, it is the reason for the intervals.
- The evaluate output: the tables E1-E6 with the extra columns `full`, `hist`, `full_shufscn`, then the HYPOTHESES block with H1, H2, H3, H4 lines, each PASS or FAIL with a number and an interval.
- Guess (not a promise): H2 passes (the model does not hurt walking), H1 C is the most likely PASS because the stop rule is easy, H1 B is the open question. A FAIL in H1 B is a valid, publishable result about this data and map, not a bug.
- Direct models: guess they are somewhat worse than residual overall (they must learn straight walking themselves), possibly equal on events. If `full_direct` is clearly worse on the controls, that is the reason for the CVM correction.

**Pass:** both evaluate reports are produced and saved in `reports/` (whatever the verdicts).
**Stop rule:** if H1 B is FAIL, send me the report before M4 and we read it together first.
**Send me:** the two reports.

## M4  Ablations (day 3)
**Do:** `python3 train.py --variant noocc` and `python3 train.py --variant concat` (start with `--seeds 0` if time is short, then add 1,2). Then run the evaluate command again with `--preds noocc=... --noocc noocc` and `--preds concat=... --concat concat`.
**Expect:** two more lines, H5 (B and C) and H7 (B and C) plus H6 (occupied share).
- H5 PASS = the obstacles matter. H5 FAIL while H1 B passes = worrying: the gain may come from the unknown-area pattern. I will tell you how to follow up (a change in the collection design, not the model).
- H7: the text says `transformer better`, `TIE` or `simple fusion better`. TIE is an honest result, not a failure.

**Pass:** H5, H6, H7 lines exist in the report.

## M5  Look with your eyes (day 3-4)
**Do:** I write two small scripts after M3 (I need the real output format): pictures of predicted vs true paths on the crop for B and C near events (good cases, bad cases, average), and the counterfactual-edit test (H8: change only the map and see whether the prediction changes the way physics says).
**Expect:** a folder of PNGs like the dataset eye-check sheets, with the crop, history, truth, CVM and model prediction; and one H8 table (share of edited samples where the prediction moves the right way).
**Pass:** you looked at the pictures and agree that good and bad cases make sense; H8 was run.

## M6  Wrap up (day 4)
**Do:** I write the results and the verdict table into the project and memory; you commit the code and the reports (`runs/` is ignored); we make one decision:
- go on to the closed loop in the simulator (T8: MPC using this predictor vs MPC using CVM), or
- go back to change the final data collection design first (more diversity, events visible at 4 s, bigger obstacles in A), if H1 B or H5 say the map is not used as hoped.
**Pass:** reports committed, verdict written, decision made.

---
## Time plan
Day 1: M1 + M2. Day 2: M3 (training runs). Day 3: M4 + start M5. Day 4: M5 + M6. The spare half-day absorbs reruns and fixes after M2.

## After M6 (outside these 4 days)
T8 closed loop in simulation with the MPC; T9 real test in the faculty; the final data collection with the lessons in memory; thesis text. These come after the decision at M6.
