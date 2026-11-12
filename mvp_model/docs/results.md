# Results: M3 + M4 (mvp_v1, 2026-11-10)

Setup: 95 environments, 8303 usable samples, 5 folds grouped by environment, 3 seeds per version, Kaggle T4.
Main number: **ADE@3s on near events** (event within 4 s), metres, lower is better. `[a, b]` = 95 % interval (bootstrap over environments).

## Headline numbers
| predictor | B near (corner) | C near (stop) | all samples | what it is |
|---|---|---|---|---|
| cvm4 | 0.74 | 0.85 | 0.30 | keep walking at the last speed |
| hist | 0.51 | 0.45 | 0.17 | network, no map |
| noocc | 0.26 | 0.39 | 0.15 | network, map without the obstacles |
| concat | 0.20 | 0.20 | 0.11 | simple fusion (GRU + CNN), map |
| **full** | **0.19** | **0.19** | **0.11** | transformer, map (final model) |
| full_direct | 0.19 | 0.18 | 0.10 | the same, outputs the path directly |
| full, other env's map (same scenario) | 0.61 | 0.70 | 0.31 | the trained full model given a wrong map |

## Hypotheses (rules fixed before the runs), residual family: full vs hist
| | verdict | number | meaning |
|---|---|---|---|
| H1 B (primary) | **PASS** | -0.32 [-0.36, -0.29] m, gain 63 % | the map helps at corners |
| H1 C | **PASS** | -0.27 [-0.30, -0.24] m, gain 59 % | the map helps for scene-explained stops |
| (D, descriptive) | | -0.15 [-0.19, -0.11] m | doorways: smaller gain |
| H2 no harm | **PASS** | -14 % [-16, -11] | on controls the map makes it better, not worse |
| H3 visibility | **PASS** | -0.24 [-0.28, -0.19] m | the gain is where the cause is visible (0.32 m visible vs 0.09 m not) |
| H4 B / C | **PASS / PASS** | +0.42 / +0.52 m | it reads THIS environment's geometry, not only the scenario type |
| H5 B / C | **PASS / PASS** | +0.07 [0.04, 0.10] / +0.21 [0.18, 0.23] m | obstacles matter (but see below) |
| H6 | **PASS** | -0.0007 | it does not predict through walls (hist does a little in C) |
| H7 B / C | transformer better | -0.014 [-0.026, -0.001] / -0.016 [-0.028, -0.005] m | real but SMALL edge over simple fusion |

Direct family (full_direct vs hist_direct): the same verdicts (H1 B 62 %, H1 C 62 %, H2, H3, H4, H6 PASS).
Seeds and folds agree closely (per-fold test ADE within about 0.01-0.02 between seeds).

## What it means
1. **The idea works in simulation: learnable cause and effect from the map.** Seeing the map cuts the near-event error at corners and stops, with no harm elsewhere, and the model uses the actual geometry of the scene (H4). Do NOT quote the 60 % as a headline: the pedestrians come from a route planner plus a social-force model, so their behaviour is almost a deterministic function of the map and the model is largely learning to invert that planner. Real people are less predictable, so the real gain will be much smaller. The percentage is supporting evidence only.
2. **The gain is short-range.** By time to the event (ADE@6s, B/C/D): 0-2 s: 0.42 vs 1.47 (hist); 2-4 s: 0.63 vs 1.24; 4-6 s: 0.34 vs 0.40. The map helps when the cause is close and visible, as the dataset analysis predicted. A (small detour) shows no gain, as expected for a control.
3. **Obstacles vs free space (H5).** In C, the occupied cells carry most of the gain (noocc 0.39 vs full 0.19). In B, most of the gain survives without them (noocc 0.26 vs full 0.19): the shape of the free/unknown area already shows where the corridor turns. That is real geometry, not necessarily a leak, but this test cannot fully separate "free-space geometry" from "robot-visibility pattern". The counterfactual test (H8) is the next check.
4. **Architecture (H7).** The transformer beats simple fusion by only about 1.5 cm (7 %), about the size of the seed-to-seed noise (0.01-0.02 m), and the bootstrap interval does not include seed noise. So: a design choice (pre-registered, attention maps can be inspected), not a finding, until `seed_check.py` shows the same sign for every seed and fold. The thesis weight goes to the scene-aware predictor and the evaluation protocol. `concat` is a strong, cheaper alternative (229k params, about 3x faster to train).
5. **Residual vs direct.** No difference: the network does not need the constant-velocity starting guess on this data.

## Decision
**Final model: `full` (residual).** It was the primary choice before the results, it ties with direct, it is slightly better than concat, and with a missing map it falls back toward constant velocity.

## Warnings (to carry into the next steps)
- **A wrong map is worse than no map:** with another environment's map, B 0.61 / C 0.70 vs no map 0.51 / 0.45; even straight walking (E) goes from 0.05 to 0.32 with a map from another scenario. On a real robot, maps are imperfect, so this is the main sim-to-real risk. M5 tests empty, shifted and noisy maps.
- Behaviour is simulator-defined, in few scene templates; this shows learnable cause and effect from the map, not real-world transfer.
- B's gain may be partly carried by the free/unknown pattern (point 3), i.e. by where the robot stood. H8 cannot separate this; the wedge-only control (H11, M5e) can.
- The data has the robot parked or following 3-5 m behind. A robot that approaches or stops near pedestrians is outside this data (matters for any closed-loop test).
- The history is noise-free simulator truth; a real tracker is not (M5 tests 0.05-0.2 m).

# M5 results (2026-11-11): what happens when the input is imperfect
Test: the final `full` model, the same samples, ADE@3s on near events [m]. "Damage" = we deliberately make one input worse (a real map and a real tracker are never perfect) and see how much the prediction gets worse. Reference: `hist` (no map) B 0.51 / C 0.45, constant velocity B 0.74 / C 0.85.
| input | B near | C near | meaning |
|---|---|---|---|
| clean | 0.19 | 0.19 | |
| no map at all | 0.57 | 0.75 | worse than `hist`; C even close to constant velocity |
| map shifted 0.2 m / 0.5 m | 0.37 / 0.44 | 0.21 / 0.25 | the corner position must be accurate: a 20 cm error removes most of the B gain |
| 1 % / 5 % wrong cells | 0.20 / 0.28 | 0.19 / 0.20 | small random errors are fine |
| map rotated 5 / 10 degrees | 0.29 / 0.37 | 0.19 / 0.21 | B is sensitive to rotation too |
| tracker noise 0.05 / 0.1 / 0.2 m | 0.29 / 0.48 / 0.80 | 0.31 / 0.54 / 0.91 | everything degrades, the map model least (see below) |
Fair opponents under the SAME noisy history (ADE@3s, B near / C near / A / E): at 0.1 m the map model is 0.48 / 0.54 / 0.34 / 0.29, constant velocity 0.84 / 0.92 / 0.33 / 0.30, `hist` 0.84 / 1.17 / 0.46 / 0.45. At 0.2 m: map model 0.80 / 0.91 / 0.63 / 0.59, constant velocity 1.00 / 1.06 / 0.56 / 0.54, `hist` 1.05 / 1.38 / 0.71 / 0.70. Reading: noise hurts every predictor, not only the map model. The map model stays clearly the best on B and C (the map still explains the behaviour); on the walking controls (A, E) it falls to constant-velocity level at 0.1 m and about 10 % worse at 0.2 m. The networks (with and without map) are worse than plain constant velocity on noisy data, because they use velocities and heading computed from consecutive noisy points. So the real fix is on the input side: smooth the tracker output (filter) and/or train with noise (M5d), not a better map model.
H8a (remove the obstacle at the stop point, C): 86 % of predictions move toward constant velocity, control (an edit 2.5 m aside) 29 % -> PASS. The model reacts to the obstacle that explains the stop.
H8b (put a wall in the way in A/E walking): 0 % stop before the wall, change 0.25 m vs control 0.14 m -> FAIL (exploratory). Reading: the model learned the situations of the training scenes (a wall at the end of a corridor), not "walls stop people" in general. A wall in an open room is a situation it never saw.
What this means: the model is accurate but brittle; it needs an accurate map and a clean tracker. That is the main risk for any real use and the first thing the real project must address (damage-trained model M5d, map-confidence fallback, tracker filtering).

# M5d results (2026-11-11): training with damage (`full_dmg`)
`full_dmg` = the same network as `full`, trained while the inputs are randomly damaged (15 % of samples without any map, map shifts up to 0.3 m, rotations up to 5 degrees, 0-3 % wrong cells, tracker noise 0 / 0.05 / 0.1 m). Clean-input comparison with `hist_dmg` (`reports/eval_full_dmg.txt`): H1 B PASS (-0.22 m, 42 % gain), H1 C PASS (-0.32 m, 62 %), H2 PASS, H3 PASS, H4 B/C PASS, H6 PASS. So the map still helps and is still read from the right place.

ADE@3s [m], `full` (clean-trained) -> `full_dmg`:
| input | B near | C near | A | E |
|---|---|---|---|---|
| clean | 0.19 -> **0.31** | 0.19 -> 0.19 | 0.11 -> 0.11 | 0.05 -> 0.05 |
| no map | 0.57 -> 0.52 | 0.75 -> 0.73 | 0.17 -> 0.16 | 0.14 -> 0.07 |
| map shifted 0.2 m | 0.37 -> 0.36 | 0.21 -> 0.21 | | |
| map shifted 0.5 m | 0.44 -> 0.42 | 0.25 -> 0.25 | | |
| 5 % wrong cells | 0.28 -> 0.33 | 0.20 -> 0.20 | | |
| rotated 5 / 10 deg | 0.29 / 0.37 -> 0.32 / 0.37 | 0.19 / 0.21 -> 0.20 / 0.21 | | |
| tracker noise 0.1 m | 0.48 -> **0.38** | 0.54 -> **0.26** | 0.34 -> **0.19** | 0.29 -> **0.14** |
| tracker noise 0.2 m | 0.80 -> **0.51** | 0.91 -> **0.39** | 0.63 -> **0.35** | 0.59 -> **0.30** |
Opponents under the same noise (0.1 m; `hist_dmg` is also noise-trained): constant velocity 0.84 / 0.92 / 0.33 / 0.30, `hist_dmg` 0.59 / 0.78 / 0.22 / 0.15. `full_dmg` is best in all four groups. At 0.2 m it is still best in all four.
H8 for `full_dmg`: H8a PASS (89 % react, control 37 %), H8b FAIL (0 % stop before the wall), the same as `full`.

**Verdict on the pre-registered rule (steps.md, M5d): NOT PASSED.**
(a) clean B near 0.31 vs 0.19 (+63 %, allowed +5 %): FAIL; clean C 0.19: ok.
(b) no map: B 0.52 vs hist 0.51 (+2 %): ok; C 0.73 vs 0.45 (+62 %): FAIL.
(c) every map-damage row better than `hist` on B and C: PASS.
Final model stays `full`.

What it shows:
1. **Noise training works, and is cheap.** Under tracker noise the errors roughly halve in every group, and the model is clearly better than constant velocity and than the no-map network.
2. **Map-damage training does NOT make the corner robust; it only makes it less precise.** In B the damaged-map rows hardly improved (shift 0.2 m: 0.37 -> 0.36) while the clean error rose from 0.19 to 0.31. The corner prediction needs the exact position of the corner; a model that is told to tolerate 30 cm errors cannot also read the corner to a few cm. C does not need that precision (0.19 both ways).
3. **The fallback without a map is still weak in C** (0.73 vs 0.45), even with 15 % of the training samples having no map. A plausible (unproven) reason: early stopping was done on clean validation data, so no checkpoint was chosen for the no-map case.
4. **So a good map is part of the system, not something the network can compensate.** For the real project: accurate map registration, a map-confidence signal, and a fallback to a no-map predictor (or constant velocity) when the map is unreliable. Tracker noise is handled by filtering plus noise training.

# M5d2 results (2026-11-11): noise-only training (`full_hn`) -> FINAL MODEL
`full_hn` = the same network as `full`, trained with tracker noise on the history (0 / 0.05 / 0.1 m) and the map untouched. Clean-input comparison with `hist_hn` (`reports/eval_full_hn.txt`): H1 B PASS (-0.33 m, 62 % gain), H1 C PASS (-0.32 m, 62 %), H2 PASS (-19 %), H3 PASS, H4 B/C PASS (+0.44 / +0.64 m), H6 PASS. H8a PASS (86 % react, control 36 %), H8b FAIL (as for `full`).
ADE@3s [m]:
| input | B near | C near | A | E |
|---|---|---|---|---|
| clean | 0.20 | 0.19 | 0.11 | 0.05 |
| tracker noise 0.05 m | 0.22 | 0.21 | 0.14 | 0.08 |
| tracker noise 0.1 m | 0.27 | 0.25 | 0.18 | 0.13 |
| tracker noise 0.2 m | 0.39 | 0.36 | 0.32 | 0.27 |
| same, 0.1 m: constant velocity / `hist_hn` | 0.84 / 0.59 | 0.92 / 0.78 | 0.33 / 0.22 | 0.30 / 0.15 |
| same, 0.2 m: constant velocity / `hist_hn` | 1.00 / 0.71 | 1.06 / 0.89 | 0.56 / 0.39 | 0.54 / 0.34 |
| map shifted 0.2 / 0.5 m | 0.38 / 0.46 | 0.22 / 0.27 | | |
| 5 % wrong cells | 0.32 | 0.24 | | |
| map rotated 5 / 10 deg | 0.29 / 0.40 | 0.20 / 0.22 | | |
| no map | 0.66 | 0.97 | 0.28 | 0.21 |
**Verdict on the pre-registered rule: PASSED** ((a) clean B 0.20 and C 0.19, limit 0.20; (b) under 0.1 m noise better than constant velocity and `hist_hn` in B, C, A and E; (c) under 0.2 m noise far better than constant velocity in A and E). **The final model is `full_hn`.**
Reading:
1. Noise-only training gives the whole noise gain (at 0.1 m: 0.48 -> 0.27 in B, 0.54 -> 0.25 in C, 0.34 -> 0.18 in A, 0.29 -> 0.13 in E, compared with `full`) at no cost on clean data. Map damage is the part that cost accuracy (M5d).
2. The map weaknesses stay: a 0.2 m shift or a 10 degree rotation roughly doubles the corner error in B (0.20 -> 0.38 / 0.40), and with no map the model is much worse than the no-map network (B 0.66, C 0.97, E 0.21 against 0.13 for constant velocity). These are not fixed by training; they need system measures (accurate registration, a map-confidence signal, a fallback to `hist_hn`), listed for the real project.

# LAST-RUN results (2026-11-12): wedge-only leak check (H11), transfer (H10), seed check
**H11 wedge-only leak check: NO LEAK.** Control model = the `full` network whose crop is only the robot's modelled camera cone (73 degrees, 8 m), no walls, no shadows. Check of the camera model against the real crops: 95.0 % of the known crop cells lie inside the modelled cone (above the 90 % bar, so the test is fair); 47.6 % of the cone cells are known (the rest is shadow or outside the crop).
ADE@3s on near events [m]: B: `hist` 0.51, `wedge` 0.49, `noocc` 0.26, `full` 0.19. C: `hist` 0.45, `wedge` 0.44, `noocc` 0.39, `full` 0.19.
B: wedge minus hist = -0.018 [-0.024, -0.013] m, gain 3.6 % (rule: LEAK needs upper bound below 0 AND gain at least 5 %) -> NO LEAK. The wedge explains 7 % [5, 10] of the gain that `noocc` gets in B (C: 20 % [4, 37]). Honest reading: robot placement alone carries a statistically visible but tiny signal (about 2 cm); the rest of the B gain without obstacles (0.51 -> 0.26) comes from the shape of the free/unknown region, which is the shadow of the real walls, the same information a real sensor produces.
**H10 transfer (exploratory, report only):** train without scenario S, test on S (`full_hn` with map vs `hist_hn` without), ADE@3s near events:
| held out | constant velocity | `hist_hn` | `full_hn` held out | `full_hn` S in training |
|---|---|---|---|---|
| B | 0.74 | 0.74 | 0.66 | 0.20 |
| C | 0.85 | 0.95 | 0.78 | 0.19 |
| D | 0.37 | 0.54 | 0.44 | 0.25 |
The script labels all three "TRANSFERS" (the map model beats the no-map model by 11-19 %, intervals below 0), but that is a lenient threshold. Honest reading: on an unseen scenario type the map model keeps only a small part of its advantage (B: 11 % instead of 62 %), is only slightly better than constant velocity in B and C, and is WORSE than constant velocity in D (0.44 vs 0.37). The model learned the cause and effect of the scenario types in the simulator; it is not a general understanding of walls and corners. This is the main limit for the thesis claim and the main argument for diverse data in the real project.
**Seed check (`seed_check.py`, `full` minus `concat`, ADE@3s near events):** B: seed means -0.0104 / -0.0235 / -0.0083, mean -0.014, all 3 seeds negative, 10 of 15 seed-fold cells negative; C: -0.0223 / -0.0170 / -0.0101, mean -0.017, 3 of 3 seeds, 11 of 15 cells. Rule (sign the same for every seed, most folds): met, narrowly. Reading: the transformer's edge over simple fusion is real in direction but small (1.4-1.7 cm, about 7 %) and flips sign in about a third of single folds. It stays "a small, consistent edge", not a headline; `concat` remains a strong cheaper alternative.
