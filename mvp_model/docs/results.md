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
1. **The idea works in simulation.** Seeing the map cuts the near-event error by about 60 % at corners and stops, with no harm elsewhere, and the model uses the actual geometry of the scene (H4).
2. **The gain is short-range.** By time to the event (ADE@6s, B/C/D): 0-2 s: 0.42 vs 1.47 (hist); 2-4 s: 0.63 vs 1.24; 4-6 s: 0.34 vs 0.40. The map helps when the cause is close and visible, as the dataset analysis predicted. A (small detour) shows no gain, as expected for a control.
3. **Obstacles vs free space (H5).** In C, the occupied cells carry most of the gain (noocc 0.39 vs full 0.19). In B, most of the gain survives without them (noocc 0.26 vs full 0.19): the shape of the free/unknown area already shows where the corridor turns. That is real geometry, not necessarily a leak, but this test cannot fully separate "free-space geometry" from "robot-visibility pattern". The counterfactual test (H8) is the next check.
4. **Architecture (H7).** The transformer beats simple fusion, but only by about 1.5 cm (7 %). Honest claim: the main contribution is making the predictor scene-aware and the evaluation; attention adds a small, measurable edge. `concat` is a strong, cheaper alternative (229k params, about 3x faster to train).
5. **Residual vs direct.** No difference: the network does not need the constant-velocity starting guess on this data.

## Decision
**Final model: `full` (residual).** It was the primary choice before the results, it ties with direct, it is slightly better than concat, and with a missing map it falls back toward constant velocity.

## Warnings (to carry into the next steps)
- **A wrong map is worse than no map:** with another environment's map, B 0.61 / C 0.70 vs no map 0.51 / 0.45; even straight walking (E) goes from 0.05 to 0.32 with a map from another scenario. On a real robot, maps are imperfect, so this is the main sim-to-real risk. M5 tests empty, shifted and noisy maps.
- Behaviour is simulator-defined, in few scene templates; this shows learnable cause and effect from the map, not real-world transfer.
- B's gain may be partly carried by the free/unknown pattern (point 3).
