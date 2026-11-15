# The model in depth: `MotionTransformer` (final model: `full_hn`)

Code: `motion/models.py`. Pictures: `docs/architecture.png` (overview), `reports/figures/architecture.png` (layer by layer, drawn by `figures/make_figures.py`). Short version of the idea: `model-spec.md`. This file explains every layer, what attends to what, and where the 282,584 parameters sit.

## 1. In one paragraph
The pedestrian's last 8 poses become 8 "tokens". The map crop around him becomes 100 "scene tokens". The 8 history tokens first talk to each other (what is he doing?), then they ask the scene tokens (what is around him that explains what he is doing?) in a cross-attention step. A small head turns the result into 12 corrections that are added to a constant-velocity guess. Everything has width d = 64 and 4 attention heads (16 numbers per head).

## 2. Hyper-parameters (defaults in the code)
| item | value |
|---|---|
| token width d | 64 |
| attention heads | 4 (16 numbers each) |
| feed-forward width | 128 (64 -> 128 -> 64, GELU) |
| history encoder layers | 2 (self-attention) |
| scene encoder layers | 1 (self-attention) |
| cross-attention decoder layers | 2 |
| dropout | 0.1 |
| normalisation | LayerNorm before each sub-block ("pre-norm"), residual connection around each sub-block |
| output | 12 points (+0.5 ... +6.0 s), residual on top of constant velocity (CVM, last 4 steps) |
| parameters | 282,584 (all trainable) |
| training | AdamW, lr 2e-3 one-cycle, weight decay 0.01, batch 128, up to 60 epochs, early stop after 10, loss = mean distance (ADE) over the 12 steps, gradient clip 1.0, random left/right mirror, tracker noise on the history (`_hn`) |

## 3. Layer by layer (shapes are per sample)

### 3.1 History branch
1. **Input features** (8 x 6): for each of the 8 poses: x/3, y/3, velocity x and y (difference to the previous pose / 0.5 s; zero for the first), cos(heading), sin(heading). The frame is the pedestrian frame: he is at (0, 0) looking along +x at the last step.
2. **Linear 6 -> 64 + learned position vector** (8 x 64). Each pose becomes a 64-number token; a learned vector per time slot tells the model "how long ago". 960 parameters.
3. **History encoder, 2 layers.** One layer = self-attention (8 x 8 weights per head) + feed-forward 64-128-64. Every history token looks at all 8, so it can compute "I am slowing down", "my heading is changing". 66,944 parameters (33,472 per layer).

### 3.2 Scene branch
1. **One-hot** the crop: (3 x 80 x 80) channels unknown / free / occupied. Resolution 0.1 m, 8 m x 8 m (2 m behind to 6 m ahead, 4 m to each side).
2. **Conv stem**: three 3x3 convolutions with stride 2 (3 -> 32 -> 64 -> 64 channels), 80 -> 40 -> 20 -> 10. Result 10 x 10 x 64 = **100 scene tokens**, each covering a 0.8 m x 0.8 m patch. 56,320 parameters. This is the only place where raw pixels are touched; it makes attention cheap (100 tokens instead of 6,400).
3. **Token position**: each token gets the (x, y) of its patch centre (in the pedestrian frame, scaled to about -1..1) pushed through a small MLP (2 -> 64 -> 64) and added. So the model knows "this patch is 2 m ahead and 1 m to the left". 4,352 parameters. A formula-based position (not a table of 100 learned vectors) was chosen because with 95 environments a table would overfit, and it behaves correctly under the left/right mirror.
4. **Scene encoder, 1 layer**: self-attention among the 100 tokens (100 x 100 weights per head) + feed-forward. Lets patches share context: "these ten patches form one wall", "this free gap leads somewhere". 33,472 parameters.

### 3.3 Cross-attention decoder (where motion meets the map), 2 layers
Each layer has three steps, in this order (`nn.TransformerDecoderLayer`):
1. self-attention among the 8 history tokens (8 x 8),
2. **cross-attention**: queries = the 8 history tokens, keys and values = the 100 scene tokens. For each head, each history token produces 100 weights (summing to 1) and takes the weighted average of the scene tokens' value vectors. This is the one step that mixes map into motion. Weights are 8 x 100 per head, 4 heads, 2 layers,
3. feed-forward 64-128-64.
Output: 8 x 64 "scene-aware history tokens". 100,480 parameters (50,240 per layer: two attention blocks of 16,640 each, feed-forward 16,576, three norms 384).

### 3.4 Head
- Take the last history token (the newest moment) and the average of all 8: 128 numbers.
- LayerNorm -> Linear 128 -> 128 -> GELU -> Linear 128 -> 24. The 24 numbers are 12 (dx, dy) corrections. 19,864 parameters.
- **Residual output**: prediction = constant-velocity guess + corrections. The last linear layer starts at zero, so an untrained model is exactly constant velocity and training only has to learn what constant velocity gets wrong (turns, stops).

### 3.5 Parameter budget
| block | parameters | share |
|---|---|---|
| input linear + position | 960 | 0.3 % |
| history encoder (2 layers) | 66,944 | 23.7 % |
| conv stem | 56,320 | 19.9 % |
| token position MLP | 4,352 | 1.5 % |
| scene encoder (1 layer) | 33,472 | 11.8 % |
| cross-attention decoder (2 layers) | 100,480 | 35.6 % |
| head | 19,864 | 7.0 % |
| robot input (unused in `full`, `full_hn`) | 192 | 0.1 % |
| **total** | **282,584** | |
`tests/test_torch.py` and `speed_test.py` print the same total. Estimated compute: about 20 million multiply-adds per prediction (my own estimate; most of it in the conv stem and the scene encoder). Measured on your laptop CPU: 2.4 ms per prediction (batch 1), 0.56 ms each in a batch of 32.

## 4. What attends to what (summary)
| step | queries | keys / values | matrix per head | question the step can answer |
|---|---|---|---|---|
| history self-attention (x2 in the encoder, x2 in the decoder) | 8 history tokens | 8 history tokens | 8 x 8 | how is he moving, is that changing? |
| scene self-attention (x1) | 100 scene tokens | 100 scene tokens | 100 x 100 | which patches belong together (wall, opening)? |
| **cross-attention (x2)** | 8 history tokens | 100 scene tokens | **8 x 100** | what in the map explains my motion? |
The map never influences the prediction except through the cross-attention step. That is why `hist` (the same network with the history as memory) is a fair no-map comparison, and why the attention weights of the last history token over the 100 scene tokens can be drawn on the map (`figures/make_figures.py --only attention`; not yet run on your models: the script is written, tested only with fake weights).

## 5. Why these choices, and what the experiments say about each
- **Cross-attention instead of concatenating features**: it keeps "motion asks the map" explicit and inspectable. Result: the transformer beats the simple fusion network (`concat`: GRU + CNN, no attention, 229k parameters) by only about 1.5 cm (7 %), consistent over the 3 seeds but small. The thesis claim should rest on the scene-aware design and the evaluation, not on the transformer being much better than simple fusion.
- **Conv stem to 100 tokens**: cheap and avoids overfitting on 95 environments; costs 0.8 m resolution in the tokens, so very small shifts of the map matter (a 0.2 m shift roughly doubles the corner error; see `results.md`, M5).
- **Residual over constant velocity**: ties with the direct output (`full_direct`) in accuracy, but falls back toward constant velocity when the scene gives no information. Kept as the primary family (decided before the results).
- **Small size**: 282k parameters, 2.4 ms on a laptop CPU; the real-project target (at most 1 M parameters, 5 ms on the Jetson) is not yet measured on the Jetson.
- **No robot position input**: avoids leaking where the data-collection script put the robot (the wedge-only control confirmed this is not what the model uses).
- **One trajectory**: the MPC needs one path; multimodal output is future work.

## 6. Known limits of this architecture (carry to the real project)
1. It needs an accurate map: 0.2 m of shift or 10 degrees of rotation roughly doubles the corner error, and with no map at all it is worse than the no-map network. Needs map registration, a map-confidence signal and a fallback.
2. It learned the situation types of the simulator (H8b FAIL, H10 partial transfer): attention can only read what training taught it to look for. More diverse scenes are the main fix.
3. Single pedestrian, single path, no interaction with other people or the robot.
4. Attention weights show where the model reads, not why it predicts a path; use them as illustration only.

## 7. Variants in the experiments (same code, one change each)
`full` (this model) | `hist` (no map: cross-attention looks at the history itself, same size) | `noocc` (occupied cells shown as free) | `concat` (no attention) | `*_direct` (no constant-velocity guess) | `*_hn` (trained with tracker noise; `full_hn` is the final model) | `*_dmg` (trained with map and tracker damage; not used) | `wedge` (map replaced by the camera cone only; leak control).
