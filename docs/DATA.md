> **English** | [中文](/docs/zh/DATA.md)

# Data card (DATA)

This file is the **single authoritative statement** about the data side of the project:
where the datasets come from, how the directory layout has to look, how labels are mapped,
how the split is done, and the three pitfalls that are easiest to fall into.

> The "Datasets" section of the report may cite the structural description in this file
> directly, but you **must run `handwash prepare` yourself and fill in the actual numbers
> (clip count / frame count / split)** — do not copy the estimates given here.

---

## 0. Three hard rules (violate one and your conclusions are void)

1. **Split first, extract frames second.** The unit of splitting must be the **source video**.
   Put adjacent frames of the same video into both the training set and the test set, and
   accuracy is inflated to 0.95+ without a single error being raised.
   The framework has three lines of defense:
   - `io/manifest.py`: the same `clip_id` across splits → raises `DataLeakageError`;
   - `io/split.py`: grouped split by `split.group_key` + a re-check after splitting;
   - `configs/config.yaml`: `split.guard_leakage: true`.

2. **Data does not go into Git.** `data/`, `models/`, and `outputs/` are all already in
   `.gitignore`. Keep the data on a local or shared drive and point the
   `HANDWASH_DATA_ROOT` environment variable at the external location:
   ```bash
   set HANDWASH_DATA_ROOT=E:\datasets\handwash     # Windows
   export HANDWASH_DATA_ROOT=/mnt/data/handwash    # Linux/macOS
   ```

3. **The label mapping is changed in exactly one place.** The mapping from a dataset's raw
   label names to the canonical `Step` **may only** be written in `_DATASET_ALIASES` in
   `src/handwash/core/labels.py`. Writing `if label == "Step1_water"` anywhere else causes a
   silent label misalignment between the training labels and the evaluation labels.

---

## 1. Dataset overview

| Dataset | Scale | Environment | Purpose | Priority |
| --- | --- | --- | --- | --- |
| **PSKUS** | 3185 clips / ~18.4 GB | Real hospital | Primary training data | Required (a subset is fine to start) |
| **METC** | 212 clips / ~2.1 GB / 72 people | Laboratory | Cross-scenario testing | Required |
| **Kaggle** | Small (a few hundred MB) | Mixed | Rapid prototype | Recommended first |
| **Jurmala** | 2427 clips / ~17 GB | Real-world setting | Extended training | Optional |
| **synthetic** | Comes with its own generator | 3D rendering | Smoke test / pretraining | Optional |
| **Self-recorded video** | 4—6 clips per person | Ordinary washbasin | Final validation | Required |

---

## 2. PSKUS (primary dataset)

- Link: <https://zenodo.org/records/4537209>
- Paper: <https://doi.org/10.3390/data6040038>
- Contents: a real hospital environment; the WHO six-step actions plus turning the faucet
  on/off and other actions, with **per-frame annotation**.

**How to obtain it**: `python scripts/download_data.py --dataset pskuss --share 1/4 --extract`
(17.1 GiB, split across 4 members — see [`DATA_COLLABORATION.md`](DATA_COLLABORATION.md))

### 2.1 Real directory layout (**verified against the actual downloaded data**)

```
data/raw/pskuss/                     ← the raw zip shards (11 × DataSet*.zip)
  DataSet1.zip ... DataSet8.zip
  SOURCES.json                       ← source + md5 manifest, **commit this to Git**
  extracted/                         ← unpacked layout (configs point dataset.root here)
    DataSet4/
      Videos/
        2020-06-26_21-26-56_camera104.mp4      ← one video = one clip
      Annotations/
        Annotator1/2020-06-26_21-26-56_camera104.csv
        Annotator1/2020-06-26_21-26-56_camera104.json
        Annotator2/...
      statistics.csv        ← per-file durations for each movement
      summary.csv           ← eight-movement duration summary for this DataSet
```

> The dataset also ships `README.md`, `statistics.csv` and `summary.csv` at the **top level**.
> These are registered as "everyone downloads these" small files (a few hundred KB in total);
> they contain no video.

### 2.2 Annotation format and label mapping (**the easiest thing to get wrong**)

The annotation CSV has exactly three columns:

```csv
frame_time,is_washing,movement_code
0.000,1,0
33.333,1,0
66.667,1,0
```

`movement_code` is an integer 0–7. Its meaning comes from the dataset's own `summary.csv`,
which lists eight movements in order, while `statistics.csv` uses the column order
`movement_1 … movement_7, movement_0`. Aligning the two gives:

| code | Movement (verbatim from `summary.csv`) | Canonical label |
| --- | --- | --- |
| 1 | Palm to palm | `step_1_palm_to_palm` |
| 2 | Palm over dorsum, fingers interlaced | `step_2_palm_over_dorsum` |
| 3 | Palm to palm, fingers interlaced | `step_3_fingers_interlaced` |
| 4 | Backs of fingers to opposing palm, fingers interlocked | `step_4_backs_of_fingers` |
| 5 | Rotational rubbing of the thumb | `step_5_rotational_thumbs` |
| 6 | Fingertips to palm | `step_6_rotational_fingertips` |
| 7 | Turning off the faucet with a paper towel | `faucet_off` |
| 0 | Other movement | `other` |

**This mapping was verified, not guessed.** Method: count the share of each code across
DataSet4's 80 annotation files (77,688 frames) — code 0 accounts for **62.6%** (the largest)
and code 7 for 15.2%, which matches "Other movement is the last entry in `summary.csv` and
dominates total duration". Both the direction and the ordering line up, so the mapping holds.

> **Two things you must state in the report**
>
> 1. **PSKUS has no `faucet_on`**, only code 7 = turning the faucet *off*.
>    So the `faucet_on` class in the `pskuss` label space will have zero support;
>    macro-F1 skips absent classes via `ignore_absent`, but the confusion matrix will still
>    show an all-zero column — **say explicitly that this is the data, not the model**.
> 2. **`other` (code 0) is 62.6% of all frames**, so the classes are severely imbalanced.
>    This is exactly why model selection uses Macro-F1 and not accuracy: a model that predicts
>    `other` for everything scores 62.6% accuracy and is worthless.

### 2.3 Two independent annotators

Every video has two independent annotations (`Annotator1`, `Annotator2`). You can:
* train on one and use the other as a **cross-check on annotation quality** (a strong report point);
* or compute Cohen's kappa between them as a reference "human ceiling".

The adapter uses `Annotator1` by default; change it in the dataset config:

```yaml
# configs/data/pskuss.yaml
datasets:
  pskuss:
    annotators: [Annotator2]      # or ["Annotator2", "Annotator1"] for priority fallback
```

**Recommendation**: run the inter-annotator agreement analysis *before* training. If the two
annotators disagree heavily on a particular step, that step is genuinely hard to distinguish
visually — the model failing on it is then an explainable finding rather than a defect.

---

## 3. METC (cross-scenario test set)

- Link: <https://zenodo.org/records/5808789>
- Contents: laboratory-environment data, 212 clips from 72 participants, with frame-level labels.

**Correct usage (this part matters)**

```
① Train on PSKUS → outputs/e4_pskuss_train/models/best.pt
② Prepare the METC frames (configs/data/metc.yaml)
③ Evaluate on METC with the **same** checkpoint:
     python -m handwash.cli evaluate --checkpoint <the best.pt from above> --splits external
```

**Do not** train on METC and then test on METC — that is no longer a cross-scenario experiment.
The report should give three sets of numbers: PSKUS test (same scenario), METC
(cross-scenario), and self-recorded video (real use).

**Config**: `configs/data/metc.yaml`

---

## 4. Kaggle hand-washing dataset (rapid prototype)

- Link: <https://www.kaggle.com/datasets/realtimear/hand-wash-dataset>
- Seven-class curated version: <https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar>

**Directory layout**

```
data/raw/kaggle/
  ├── Step1_water/<clip>/*.jpg      # or directly Step1_water/*.jpg
  ├── Step2_water/
  └── ...
```

**Note**: `Step7_water` / `not_washing` are folded into `other` in the framework, while the
`kaggle` label space contains only the six steps, so those samples are not counted in
six-class training (see `core/labels.py`).

**Config**: `configs/data/kaggle.yaml` (the default `dataset.name`, used for the one-command
smoke test)

---

## 5. Jurmala (optional extension)

- Link: <https://zenodo.org/records/5808764>
- Scale: 2427 clips / ~17 GB. The collection protocol comes from the same source as PSKUS.
- Usage: like PSKUS it uses a 6-class label space covering the six steps, so it can be added
  to training when time and compute allow. It is **not required to complete the assignment**.

---

## 6. Synthetic hand-washing dataset (optional)

- Data and code: <https://github.com/r-ozakar/synthetic-hand-washing>
- Paper: <https://doi.org/10.3390/jimaging11070208>
- Characteristics: 3D scene generation, including RGB, depth, and hand masks; usable for
  research on synthetic-data pretraining.
- Recommendation: **treat it as an extension experiment only**. If you do run it, the report
  must compare "synthetic pretraining + real-data fine-tuning" against "real-data-only
  training", otherwise its value cannot be demonstrated.

---

## 7. Team-member self-recorded video (final validation)

Filming and annotation rules are in [`SELF_RECORDING.md`](SELF_RECORDING.md) (in Chinese).
Three key points:

1. 4—6 clips per team member, which must cover: **fully correct / one step missing /
   steps swapped / one step too short**;
2. The camera position/angle should follow the public datasets: both hands clearly visible
   throughout (overhead, or 45° from the side-front);
3. **The human ground truth must be recorded** (the real step sequence), otherwise there is
   nothing to compare the model's output against.

Self-recorded video is assessed with `handwash assess` (completeness), not with
`handwash evaluate` (frame-level classification) — because the value of self-recorded video
lies in the **complete procedure**, and a human cannot produce frame-level labels for it
either.

---

## 8. Common problems (look up by the error you see)

| Error | Cause | Fix |
| --- | --- | --- |
| `DatasetNotFoundError` | wrong `dataset.root` path | check the `datasets` section of `configs/config.yaml`; or use `HANDWASH_DATA_ROOT` |
| `标签空间 X 无法识别标签：'yyy'` (label space X cannot recognize label 'yyy') | the dataset's label names were never registered | add the mapping to `_DATASET_ALIASES` in `core/labels.py` (L1, the change needs review) |
| `DataLeakageError` | the same source video across splits | make sure `split.group_key=original_video`; make sure you split before extracting frames |
| `manifest 缺少必需列` (manifest is missing a required column) | an old-format manifest is in use | delete `data/processed/*/manifest.csv` and re-run `handwash prepare` |
| "视频过短，已跳过" (video too short, skipped) during frame extraction | the video is shorter than `min_frames_per_clip` | lower that value, or check whether the video is corrupt |
| accuracy > 0.98 | almost certainly data leakage | see above; also check whether test data ended up in train |

**Frame-extraction parameter recommendations**: an extraction rate of 5 fps (`fps: 5.0`) is
enough for action-level tasks (on a 30fps video that means 1 frame in every 6), and storage
drops to roughly 1/6; write `resize_hw: [256, 256]` to disk and let the model crop to 224.

---

## 9. Split and statistics (self-check list)

Once `handwash prepare` has finished, check `data/processed/<dataset>/split_report.json`:

- [ ] The clip-count ratio across the three splits is close to 0.7/0.15/0.15 (small datasets
      will deviate, which is normal);
- [ ] **No** source video appears in two splits at the same time (already enforced by the
      framework);
- [ ] Every split contains samples of all six steps (`label_frames` should contain no 0);
- [ ] `val` is not empty (otherwise you cannot select a model by early stopping);
- [ ] Each video's frame count and duration are plausible (outliers usually mean a decode
      failure or label misalignment).

Writing these check results into the data section of the report is more persuasive than any
adjective.
