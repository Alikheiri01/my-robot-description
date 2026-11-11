# Evaluation plan (v1, 2026-11-08, written BEFORE any model is trained)

Everything marked PASS/FAIL below is fixed now, before we see any result. If a rule is changed later, the change and the reason are written here with a date, and the old result stays in the report.

## 1. What we want to find out
One question: **does the model that sees the map (crop) predict the pedestrian better than the same model without the map, in the situations where the map explains the behaviour (a corner, a scene-explained stop), and without getting worse elsewhere?** Everything else (the other tests) checks that a "yes" is real and not an accident.

## 2. Words used in this file
- **Pedestrian frame**: coordinates with the pedestrian at (0,0) looking along +x, as at the anchor time. All errors are in metres in this frame.
- **Anchor**: the moment of the sample; history is before it, the label (truth) is the 6 s after it.
- **Event**: the thing the scenario is about: the bend at the corner (B), the stop in front of the wall (C), the doorway (D), the small obstacle detour (A). E has no event.
- **Near event**: a sample whose event happens within 4 s after the anchor. Our main stratum. We use it because the crop only sees the cause well when the event is close (known cells 85-97 % within 2 s, only 5-17 % at 4-6 s), and because the crop only reaches 6 m ahead, which a pedestrian passes after about 4-5 s.
- **Visible event**: the point where the event happens lies inside the crop and at least half of the 1 m disk around it is known (not grey).
- **Usable sample**: pedestrian seen at the anchor and in at least 6 of the 8 history steps. Only usable samples are trained on and evaluated.
- **Environment**: one generated scene (walls, furniture, a walk). 95 of them have data. Samples from one environment are almost the same picture, so **95 environments, not 8000 samples, is our real sample size**.
- **Fold / cross-validation**: we cut the 95 environments into 5 groups ("folds"). To predict fold 1 we train on folds 2-5 and so on. Every sample is predicted by a model that never saw its environment. A scenario's environments are spread over all 5 folds.
- **Seed**: a random start of the training. We train 3 times and average, so one lucky or unlucky run does not decide.
- **Reference ("hist")**: the same network without the crop. It is the fair "no map" opponent. Until it is trained, `ridge_hist` (a linear model on the history) is used.
- **Control scenarios**: A and E (and the no-event samples of B, C, D). There the map should not matter; the model must simply not get worse.

## 3. The numbers we compute (metrics)
All in metres (the occupied share is a fraction). Lower is better for all of them.
| metric | meaning | why |
|---|---|---|
| **ADE@3s** (PRIMARY) | average distance between predicted and true position over the first 3 s (6 points) | the crop can still act here; it is closer to what an MPC needs |
| **ADE@6s** | same over all 6 s (12 points) | secondary: the cause is often invisible at 4-6 s, so this dilutes the effect |
| **FDE@3s / FDE@6s** | distance at the last point of that horizon | shows the worst point of the path |
| **Occupied share** | fraction of the 12 predicted points that lie inside an occupied crop cell | the true path is almost never in an occupied cell (0.00 % of label points in mvp_v1). A model that averages two possibilities may predict THROUGH an obstacle with a decent ADE; an MPC would inherit that. ADE cannot show it, this does |

**Interval ("95 % interval")**: after each mean we print `[low, high]`. It comes from a bootstrap over environments: draw 95 environments with replacement (inside each scenario), recompute, repeat 2000 times, take the 2.5 % and 97.5 % values. If the interval of a DIFFERENCE does not contain 0, the difference is not luck of which environments we happened to have.

## 4. What this data can and cannot show
- The behaviour is defined by the simulator and each scenario uses its own scene template. A good result means "the model can learn cause and effect from the crop in simulation". Real-world generalisation is only tested in the faculty test (T9).
- A (detour) is weak by construction (the path bends only about 0.3 m) and E has no event. They are controls, not evidence.
- **B (corner) is the main test**, because the turn timing is hard. C (scene-explained stop) is expected to be easy, because the stop is almost a rule (0.5-0.9 m before the wall), so a pass in C alone would not prove much. Every test that matters is therefore read for B and for C **separately**, never pooled.
- D (doorway) is marginal: reported, no verdict.

## 5. The hypotheses (H1-H7 are computed by `evaluate.py`)
For each: the question, how it is measured, the pass rule, and what a FAIL means. "Test" = `full`, "reference" = `hist`.

**H1  Does the map help where it should? (main result)**
- Question: on near events, is the full model better than the model without the map?
- Measured: ADE@3s of `full` minus ADE@3s of `hist`, near events, for B and for C separately.
- PASS: the upper end of the 95 % interval is below 0 AND the gain is at least 5 % of the reference's ADE. **B is the primary** (H1 B); C (H1 C) is also reported. H1 as a whole counts as passed only if B passes.
- FAIL means: with this data and model, the map does not help where we expected it. Look at H3 to see whether the map could see the cause at all.

**H2  Does it do no harm elsewhere?**
- Question: on the control samples (all of A and E; the samples without event in B, C, D), is `full` worse than `hist`?
- Measured: relative change of ADE@3s = (full - hist) / hist. Relative, because in C the standing samples have an error of only about 0.05 m, where a fixed slack would be too big.
- PASS: the upper end of the interval is below +3 % (at most 3 % worse).
- FAIL means: the map confuses the model on normal walking.

**H3  Does the gain follow visibility? (exploratory)**
- Question: is the gain bigger when the event point is visible in the crop than when it is not?
- Measured: (diff on visible near events) minus (diff on not-visible near events), B, C, D together.
- PASS: upper end of the interval below 0.
- FAIL means: either the gain comes from something other than seeing the cause, or visibility is too rare to tell. If H1 fails and H3 fails, the problem may be the data (the robot does not see the cause), not the model.

**H4  Does it use the geometry of THIS environment?**
- Question: if we give the trained `full` model the crop of ANOTHER environment of the same scenario (same kind of scene, different walls), does it get worse?
- Measured: ADE@3s with swapped crops (`shufscn`) minus normal, near events, B and C separately.
- PASS: lower end of the interval above 0.
- FAIL means: the model does not read where the corner or wall really is; it only recognises "this kind of scene". (The swap across ALL scenarios, `shufall`, is printed for information: if it hurts a lot but `shufscn` does not, the model only learned the scenario type.)

**H5  Is it the obstacles, or only the grey pattern?**
- Question: if all occupied cells are shown as free (`noocc`), is the model worse than `full`?
- Measured: ADE@3s of `noocc` minus `full`, near events, B and C.
- PASS: lower end above 0.
- FAIL means (the case to worry about most): the gain comes from the shape of the unknown (grey) area, which our robot placement creates, not from obstacle geometry. Then the result may not transfer to a real robot.

**H6  Does it avoid predicting through walls?**
- Measured: occupied share of `full` minus `hist`, scenarios A, B, D.
- PASS: upper end of the interval below +0.005 (not worse than the reference).

**H7  Is the transformer needed? (information only)**
- Measured: ADE@3s of `full` minus `concat`, near events, B and C.
- Reading: `full` better = attention helps; TIE = a simple fusion does the same, then our contribution is the scene-aware data and evaluation, not the architecture. Better to know it now than from a committee member.

**Later tests (separate scripts, written after we have a trained model):**
- **H8  Counterfactual edits ("what if the wall were different?")**: take a real test sample and change ONLY the crop (for example remove the wall at the end of the corridor, or open the corner), keep the history identical, and look at the predicted path. If the model truly reads the scene, the prediction must change the way physics says: no wall = no stop, a corner = a turn. This is the strongest single test because it asks whether the model reacts to geometry itself. Rules for the edits: change cells only where the cell is already known, and move the grey shadow together with a moved wall; otherwise the edited crop looks like nothing seen in training and we would measure that instead. The exact edit list and threshold are fixed when this script is written, before it is run.
- **H9  Speed (information, MVP)**: at most 5 ms per prediction and at most 1 M parameters. In the MVP this is measured on the laptop CPU only; the Jetson Orin Nano measurement belongs to the real project (change dated 2026-11-10, see the change log).
- **H10  Transfer (exploratory)**: train on all scenarios but one, test on the missing one. Report only.
- **H11  Wedge-only leak check (added 2026-11-10, before it is run)**: a model whose crop is only the robot's visibility wedge in an empty world (known inside the camera cone, unknown outside, no walls). Measured like H1 on near events of B: ADE@3s of wedge minus `hist`. LEAK if the upper end of the 95 % interval is below 0 and the gain is at least 5 % of `hist`'s ADE (the H1 rule): robot placement alone then predicts the corner. Also report the share (hist - wedge) / (hist - noocc): how much of the noocc gain the wedge explains. No leak = the shadow geometry is real information a real sensor would also produce.
- **R1/R2  Robustness tables (information, added 2026-11-10)**: ADE@3s of the final model with damaged input (map none / shifted / wrong cells / rotated, tracker noise on the history). No pass/fail; the "before" table belongs to the model trained on clean data, the "after" table to the damage-trained model.

### Residual and direct models
Both styles are trained (see model-spec.md). **The residual family is primary**: its hypotheses H1-H7 are the verdicts. The direct family gets the same table (`--test full_direct --ref hist_direct`) but is a secondary comparison. We also look at `full_direct` against `full` (ADE@3s on near events of B and C, and on the controls) to answer "does the network need the constant-velocity guess?". This choice is written before any result.

## 6. How to read the outcomes
| result | meaning | next step |
|---|---|---|
| H1 B, H2, H4 B, H5 B all PASS | the map carries real, geometry-based information in simulation | H8, H9, then closed loop |
| H1 PASS, H4 FAIL | the gain is not from this environment's geometry (scenario type or something else) | look at `shufall`, check for leakage |
| H1 PASS, H4 PASS, H5 FAIL | gain comes from the unknown/known pattern, not obstacles | the final data collection must make the robot see more (placement), re-test |
| H1 FAIL and H3 FAIL | not enough visible cause, or the idea fails in this data | inspect examples, check collection design (events visible at 4 s) before blaming the model |
| H1 FAIL but H3 PASS | works only where the cause is visible | collection design problem, not a model problem |
| H2 FAIL | model hurts normal walking | smaller model, stronger regularisation |
| H7 TIE | attention not needed | say so honestly in the thesis; use the simpler model |
Never tune on the test folds. A change after seeing results is a new, dated experiment.

## 7. Tests of the tests (already built, `tests/test_all.py`)
We check the evaluation itself before trusting it. Two synthetic datasets with the real folder format:
- **determined**: the crop contains a marker that decides what the pedestrian does. The evaluation must say H1 PASS and H4 PASS.
- **noise**: the same markers placed at random. The evaluation must say H1 FAIL.
Also tested: environments are never split over folds, the mirror augmentation (flips y, heading, label and the crop rows together; CVM of a mirrored history = mirrored CVM; the side of the marker in the crop flips with the label), the cache, and the baselines.

## 8. Commands (from `mvp_model/`; dataset default `../mvp_collection/datasets/mvp_v1`)
    python3 tests/test_all.py                    # must print ALL TESTS PASSED
    python3 evaluate.py                          # baselines only; writes reports/eval_report.txt
    python3 tests/test_torch.py                  # must print TORCH CHECK PASSED
    python3 train.py --variant full --seeds 0 --epochs 2 --fold 0      # smoke run
    python3 train.py --variant full              # then hist, noocc, concat
    python3 evaluate.py --test full --ref hist \
        --preds full=runs/oof_full_s0.npz,runs/oof_full_s1.npz,runs/oof_full_s2.npz \
        --preds hist=runs/oof_hist_s0.npz,runs/oof_hist_s1.npz,runs/oof_hist_s2.npz \
        --preds full_shufscn=runs/oof_full_s0_shufscn.npz,runs/oof_full_s1_shufscn.npz,runs/oof_full_s2_shufscn.npz \
        --shuf-scn full_shufscn
    The same for the direct family: train `--variant full_direct` and `--variant hist_direct`, then evaluate with `--test full_direct --ref hist_direct` and the `*_direct` files.
    (H5 and H7 add:  --preds noocc=runs/oof_noocc_s0.npz,... --noocc noocc   and   --preds concat=... --concat concat)

## 9. Change log (every change to a rule or a number, with date and reason)
- **2026-11-10, H8 rules fixed in `inspect_model.py` before the first run.** H8a (remove the obstacle at the stop point, control = the same removal 2.5 m aside, PASS = at least 60 % react and at least 30 points above the control) and H8b (a wall in the way, control = the wall beside the path, PASS = at least 50 % no longer cross and change at least 3x the control). The first draft measured the reaction over 3 s; a stand-in test showed that misses stops that come after 3 s, so the horizon is 6 s (changed before any real run).
- **2026-11-10, numbering.** The numbering of this file (H1-H10 as printed above and in `evaluate.py`) is the numbering of the results: H5 = noocc (obstacles or only the grey pattern), H6 = occupied share, H7 = transformer vs concat, H8 = counterfactual edits, H9 = speed, H10 = transfer. In the first draft the order was different (H5 counterfactual edits, H7 transfer); the draft was replaced on 2026-11-09, before any model was trained, so no verdict was ever given under the old numbers.
- **2026-11-10, H9 scope.** The Jetson measurement is moved out of the MVP (the MVP runs only on the laptop and in simulation). The 5 ms / 1 M parameter limits stay as targets for the real project.
- **2026-11-10, H11 added** (wedge-only leak check) and **R1/R2** (robustness tables), after the review of the M3/M4 results; both defined before they are run.
- **2026-11-10, H7 reading.** The transformer edge over concat (about 1.5 cm) is only claimed if the sign holds for every seed and most folds (`seed_check.py`). The bootstrap interval averages the seeds first and does not contain seed noise.
- **2026-11-09, H2** threshold changed from an absolute +0.02 m to a relative +3 % (before training; in C the standing samples have an error of only 0.05 m).
- **2026-11-11, M5 results and training damage.** H8a PASS, H8b FAIL (see results.md). The pass rule for the damage-trained model (a/b/c in steps.md, M5d) and the damage levels (`motion/damage_torch.py`) are fixed before that run. Fair opponents (constant velocity and `hist` under the same history noise) added to the damage table.
