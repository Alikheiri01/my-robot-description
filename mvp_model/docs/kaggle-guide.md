# Training on Kaggle (or Colab) -- full guide

**The idea:** your laptop only prepares one small data file and, at the end, evaluates (numpy, light). The GPU on Kaggle does the heavy training. Everything the training needs is: the code (`mvp_model/`) and ONE data file (`mvp_v1_packed.npz`, a few MB).

**Why not upload `mvp_collection`?** `datasets/mvp_v1` is thousands of tiny files: slow to upload and Kaggle handles it badly. Training only needs the numbers inside them, so `pack_dataset.py` puts them into one file (with the same 5 folds, so Kaggle and your laptop agree about which environment is in which fold). The big dataset stays on your laptop, where `evaluate.py` can still use it.

**Device-agnostic:** `train.py` now has `--device auto` (default): GPU if there is one, otherwise CPU. Same code, no edit. Do not mix devices inside one comparison (e.g. `hist` on CPU and `full` on GPU): the random numbers differ slightly. Train all versions on the same platform (Kaggle).

---
## Part A -- on your laptop (about 10 minutes)

**A1. Replace/add the files** (from the chat): `motion/data.py`, `train.py`, `docs/steps.md`, `docs/kaggle-guide.md`, new `pack_dataset.py`, new `kaggle/kaggle_train.ipynb`.

**A2. Delete the smoke-test leftovers** (they were from the 2-epoch test):
```
cd ~/ros2_ws/src/my_robot_description/mvp_model
rm -rf runs
```

**A3. Pack the data:**
```
python3 pack_dataset.py
```
Expect: `[pack] wrote packed/mvp_v1_packed.npz (a few MB, 9984 samples, 8303 usable) -- check passed`. (Takes about a minute: it reads all samples once.)
Pass: that line appears.

**A4. Make the code zip** (without the generated folders):
```
cd ..
zip -r mvp_model_code.zip mvp_model -x "mvp_model/runs/*" "mvp_model/packed/*" "*/__pycache__/*"
```
You now have two files to upload: `mvp_model_code.zip` and `mvp_model/packed/mvp_v1_packed.npz`.

---
## Part B -- Kaggle (first time only: account and dataset)

**B1. Account.** Sign in at kaggle.com. To use a GPU, Kaggle asks you to verify your phone number (Settings -> Phone verification). I cannot check from here whether Kaggle works from Iran for your account; if the verification or the GPU option does not work, use Colab (Part F), it is the same notebook.

**B2. Upload the files as a Dataset.** Create (left menu) -> New Dataset -> drag in the two files -> Title: `mvp-model-input` -> Visibility: Private -> Create. Wait until it says the files are processed. (Kaggle may unpack the zip into a folder; the notebook handles both.)

**B3. New notebook.** Create -> New Notebook. Then File -> Import Notebook -> upload `kaggle/kaggle_train.ipynb`.

**B4. Add the data to the notebook.** Right panel -> Input -> Add Input -> Your Datasets -> `mvp-model-input` -> Add. It appears under `/kaggle/input/...`.

**B5. Choose the GPU.** Right panel -> Session options (or "Settings") -> Accelerator -> **GPU T4 x2**. (If a later error says "no kernel image is available", choose the other GPU type; P100 sometimes does not work with the newest PyTorch.) Internet can stay off.

---
## Part C -- the first check (about 5 minutes, interactive)

Run the cells one by one (click a cell, Shift+Enter). What you should see:
| cell | what it does | expect |
|---|---|---|
| 1 | settings | nothing printed |
| 2 | finds code and data | `platform: Kaggle`, `code : /kaggle/working/mvp_model`, `data : ...mvp_v1_packed.npz (a few MB)` |
| 3 | GPU check | `GPU available: True`, `GPU: Tesla T4`, `GPU test ok: True` |
| 4 | network check | ends with `TORCH CHECK PASSED` |
| 5 | smoke run (1 fold, 2 epochs) | `[train] variant full device cuda (Tesla T4) ...` and a line with test ADE@6s about 0.7. **Note the time in brackets** (on your CPU it was 24 s; the GPU should be a few seconds) |

Pass: cells 2-5 run with no red error. If cell 3 says `GPU available: False`, the accelerator is not set (B5), set it and run again from cell 2.

---
## Part D -- the real training

Two runs. Every run starts from a clean folder, so results of the two runs never mix.

| run | `RUN_NAME` | `VARIANTS` in cell 1 | what it answers |
|---|---|---|---|
| 1 (main) | `'main'` | `['full', 'hist', 'full_direct', 'hist_direct']` | the main test (H1-H4), residual and direct style |
| 2 (ablations) | `'ablations'` | `['noocc', 'concat']` | H5 (obstacles or only the grey pattern) and H7 (is the transformer needed) |

**How to run (recommended, so your laptop can be closed):**
1. Top right: **Save Version** -> **Save & Run All (Commit)** -> Advanced settings: Accelerator GPU -> Save.
2. Kaggle runs the whole notebook in the background (all cells, 12 h limit). You can close the page. The "Versions" panel shows the progress.
3. When it is finished: open the notebook page -> **Output** tab -> download `results_main.zip`.

The alternative is to press Run on cell 6 yourself and keep the tab open; it works too, but an idle or closed tab can end the session. If a session ends in the middle, run cell 6 again: seeds that are finished are skipped.

**What to expect in the log (cell 6):** 15 lines per version (3 seeds x 5 folds): `[train] seed 0 fold 3: ... val ADE 0.5xx test ADE@6s 0.5xx (NN s)`, then `>>> full finished in N min`. Numbers differ by fold (that is why we use intervals). **Time (my estimate, not measured):** about 1-3 minutes per training on a T4, so about 15-45 minutes per version and 1-3 hours for run 1. Your smoke run tells you more: the time in cell 5 times about 25 epochs times 15 trainings.
Kaggle gives about 30 GPU hours per week, so run 1 + run 2 is well inside that.

Cell 7 also runs the evaluation and prints the hypothesis table (H1-H4). Looking at it on Kaggle is optional; the same command runs on your laptop in Part E.

---
## Part E -- what to save and what to do with it

`results_<name>.zip` contains everything, and you need all of it:
| content | why |
|---|---|
| `runs/oof_*.npz` (predictions for every test fold, per variant and seed, plus the shuffled-crop ones) | the evaluation (H1-H7) is computed from these |
| `runs/models/*.pt` (trained weights, one per seed and fold) | the prediction pictures and the counterfactual test (M5), the speed test later. **Without them you would have to train again** |
| `runs/logs/*.log` | the training log: times, per-fold numbers |
| `reports/eval_*.txt` | the evaluation Kaggle already made |

Also write down: GPU type, minutes per version (from the log), and the Kaggle notebook version number.

**On your laptop:**
```
cd ~/ros2_ws/src/my_robot_description/mvp_model
unzip ~/Downloads/results_main.zip        # creates runs/ and reports/ inside mvp_model
unzip ~/Downloads/results_ablations.zip   # after run 2 (answer "A" if asked to merge/overwrite: the file names do not clash)
```
Then evaluate with the real dataset (the command is in `docs/evaluation-plan.md`, section 8; the same `--preds` names as in cell 7). The evaluation is plain numpy and light for your laptop.
**Send me:** `reports/eval_full.txt`, `reports/eval_full_direct.txt`, and the log files (`runs/logs/*.log`). Keep the zip files: do not delete them until the thesis is done.

---
## Part F -- if you use Colab instead of Kaggle
Same notebook. 1) colab.research.google.com -> File -> Upload notebook -> `kaggle_train.ipynb`. 2) Runtime -> Change runtime type -> T4 GPU. 3) Left panel (folder icon) -> upload `mvp_model_code.zip` and `mvp_v1_packed.npz` into `/content` (they disappear when the session ends, so upload again next time; or put them in Google Drive and mount it). 4) Run the cells as in Part C and D. Cell 8 starts the download of the results zip by itself. Colab sessions end when you close the browser or after a few idle hours, so keep the tab open and use the "run cell 6 again" rule if it is cut.

## Troubleshooting
| message | meaning / fix |
|---|---|
| `No train.py and no mvp_model zip found` | the dataset is not added as Input (B4), or the code zip was not uploaded |
| `mvp_v1_packed.npz not found` | the packed file was not uploaded in the same dataset |
| `GPU available: False` | accelerator not chosen (B5) |
| `no kernel image is available for execution on the device` | the GPU type is too old for this PyTorch: choose T4 x2 |
| the notebook "Save & Run All" fails | open the version's log, find the first red error and send it to me |
| session ended in cell 6 | run cell 6 again; finished seeds are skipped |
| `CHECK FAILED` in `pack_dataset.py` | send me the line; do not upload that file |
