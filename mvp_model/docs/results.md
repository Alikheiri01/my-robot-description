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
| tracker noise 0.05 / 0.1 / 0.2 m | 0.29 / 0.48 / 0.80 | 0.31 / 0.54 / 0.91 | strong sensitivity (see note) |
Note on noise: the table was made before fair opponents were added; the next run of `inspect_model.py` also prints constant velocity and `hist` under the same noise (they suffer too, constant velocity most). Only the comparison with them tells if the map model is worse than its opponents under noise.
H8a (remove the obstacle at the stop point, C): 86 % of predictions move toward constant velocity, control (an edit 2.5 m aside) 29 % -> PASS. The model reacts to the obstacle that explains the stop.
H8b (put a wall in the way in A/E walking): 0 % stop before the wall, change 0.25 m vs control 0.14 m -> FAIL (exploratory). Reading: the model learned the situations of the training scenes (a wall at the end of a corridor), not "walls stop people" in general. A wall in an open room is a situation it never saw.
What this means: the model is accurate but brittle; it needs an accurate map and a clean tracker. That is the main risk for any real use and the first thing the real project must address (damage-trained model M5d, map-confidence fallback, tracker filtering).
