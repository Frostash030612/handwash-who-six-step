> **English** | [中文](/docs/zh/CONFIG.md)

# Configuration Reference (CONFIG)

Config keys correspond **word for word** to the dataclass fields in
`../src/handwash/core/config.py`, and a misspelled key **raises an error outright**
(it is never silently ignored). This document explains every key.

Recap of usage:

```bash
# configs/config.yaml is loaded by default
python -m handwash.cli train

# layer one override file on top (recommended: write only what differs from the default)
python -m handwash.cli train config=configs/experiments/smoke.yaml

# change a single parameter for one run (key.sub=value, value parsed as a YAML scalar)
python -m handwash.cli train train.epochs=5 runtime.device=cpu

# replace the base configuration (the file must be self-contained)
python -m handwash.cli train --config configs/experiments/smoke.yaml
```

> `config=` (a positional argument) **layers on top**, whereas `--config` **replaces**.
> That distinction is deliberate: layering is for experiment overrides, replacement is
> for a completely independent set of configuration.

---

## schema_version

Configuration-schema version, currently `1`. Increment it on breaking changes, and update
`docs/CHANGELOG` and the migration notes together with it.
**Do not change this number by hand just to "make it run".**

## project

| Key | Type | Description |
| --- | --- | --- |
| `name` | str | Project name, written into artefact metadata |
| `task` | str | Task identifier, default `who_six_step_recognition` |
| `language` | str | Report language, default `zh` |

## runtime

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | int | 42 | Global random seed. **A fixed seed is the precondition for reproducible results** |
| `deterministic` | bool | true | Makes cuDNN use deterministic algorithms (slightly slower but reproducible) |
| `device` | str | auto | `auto` / `cpu` / `cuda` / `cuda:0` |
| `num_workers` | int | 4 | Number of DataLoader processes. **Set it to 0 when Windows errors out** |
| `pin_memory` | bool | true | Usually faster when a GPU is present |
| `log_level` | str | INFO | DEBUG / INFO / WARNING / ERROR |
| `tracking` | str | csv | `none` / `csv` / `tensorboard` / `wandb` |
| `tracking_project` | str | handwash | Project name on the tracking platform |
| `run_name` | str\|null | null | Output directory name. When null a timestamp is used; **pin it for comparison experiments** |

## paths

| Key | Default | Description |
| --- | --- | --- |
| `out_dir` | `outputs` | Root directory for experiment artefacts |
| `models_dir` | `models` | Where external weights are kept (not committed to Git) |
| `cache_dir` | `.cache/handwash` | Cache directory |

Relative paths are always resolved against the **repository root** (not the current
working directory).

## dataset

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | kaggle | Dataset currently in use; it must appear in the `datasets` section |
| `root` | str | `data/raw/kaggle` | Raw dataset directory |
| `variants` | list[str] | `[]` | Dataset variant tags (reserved) |
| `label_space` | str\|null | null | Label-space name; null means the same as `name` |
| `include_non_wash` | bool | false | Whether auxiliary actions such as turning the tap on/off count as classes |
| `prep.fps` | float | 5.0 | **Frame extraction rate**. 5 fps is enough for action-level tasks |
| `prep.frame_step` | int | 1 | Fixed-stride downsampling (alternative to fps; fps wins if both are set) |
| `prep.resize_hw` | [int,int] | [256,256] | Size written to disk (slightly larger than the model input, leaving room for augmentation) |
| `prep.image_ext` | str | jpg | jpg / jpeg / png |
| `prep.jpeg_quality` | int | 92 | 1—100 |
| `prep.min_frames_per_clip` | int | 10 | Videos shorter than this are skipped |

## datasets

Dictionary of dataset profiles: `datasets.<name>.root / processed_dir / frames_dir / manifest`.
The training script reads only the one profile matching `dataset.name`; the other profiles
do not affect the run.

**`image_path` inside the manifest is a path relative to `<root>`** — so the whole data
directory can be moved away or mounted on another drive without changing the manifest.

## split

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `group_key` | str | original_video | Split grouping key: `original_video` / `clip_id` / `participant` |
| `train` / `val` / `test` | float | 0.70/0.15/0.15 | The three must sum to 1.0 |
| `stratify_by` | str\|null | label_sequence | Stratification basis; null means no stratification |
| `seed` | int | 42 | Random seed for the split |
| `max_frames_per_clip_train` | int\|null | 200 | Long videos are truncated at **equal intervals** (not by taking the first N frames) |
| `max_frames_per_clip_eval` | int\|null | 100 | Same as above, for evaluation sets |
| `guard_leakage` | bool | true | Raise an error as soon as an original video crosses splits |

> `group_key` is the key to preventing data leakage. **Switching it to `clip_id` gives up
> the leakage protection**, unless you can guarantee that clip_id corresponds one-to-one
> to the original video.

## model

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | yolo26n-cls | Human-readable model name (does not affect behaviour) |
| `arch` | str | yolo26n-cls | **Registered name**, see `../src/handwash/models/` |
| `pretrained` | str\|bool | auto | `auto` / `true` / `false` / a weight name / a local path |
| `num_classes` | int\|null | null | **Keep it null**: it is derived automatically from the label space |
| `image_size` | int | 224 | Must be a multiple of 64 (YOLO is unfriendly to sizes that are not multiples of 32) |
| `normalize` | str | zero_one | `imagenet` / `zero_one` / `minus_one_one` |
| `dropout` | float | 0.2 | [0, 1) |
| `temporal.kind` | str | gru | `gru` / `tcn` / `mean_pool` / `none` |
| `temporal.hidden_size` | int | 128 | Hidden dimension of the temporal head |
| `temporal.num_layers` | int | 1 | Number of GRU layers |
| `temporal.bidirectional` | bool | false | true uses future frames and **cannot be used for a live demo** |
| `temporal.window` | int | 16 | Temporal window length (frames) |
| `temporal.stride` | int | 8 | Sliding-window stride |
| `temporal.kernel_size` | int | 3 | TCN convolution kernel |
| `temporal.dilations` | list[int] | [1,2,4,8] | TCN dilation rates |
| `temporal.dropout` | float | 0.1 | Dropout of the temporal head |

**Combining `normalize` with arch (the most common trap)**

| arch | Required normalize | Reason |
| --- | --- | --- |
| `yolo26n-cls` / `yolon-cls` | `zero_one` | ultralytics already normalises internally, so normalising once more means doing it twice |
| `mobilenet_v2` / `resnet18` / `efficientnet_b0` | `imagenet` | These use ImageNet pre-trained weights, so their input distribution must be matched |

`handwash doctor` checks this combination.

## train

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | str | frame | `frame` / `clip` / `hybrid` (decides whether the temporal training path is taken) |
| `epochs` | int | 20 | Number of training epochs |
| `batch_size` | int | 32 | **Means something different in clip mode**: number of windows × window length |
| `eval_batch_size` | int | 64 | Evaluation batch |
| `lr` | float | 3e-4 | Learning rate |
| `weight_decay` | float | 5e-4 | Weight decay |
| `optimizer` | str | adamw | `adamw` / `sgd` |
| `momentum` | float | 0.9 | SGD only |
| `scheduler` | str | cosine | `cosine` / `step` / `none` |
| `warmup_epochs` | float | 1.0 | Number of warm-up epochs |
| `label_smoothing` | float | 0.05 | [0, 1) |
| `class_weights` | str | none | `none` / `balanced` (use when classes are imbalanced) |
| `early_stopping_patience` | int | 5 | Stop once validation Macro-F1 fails to improve for N consecutive epochs |
| `grad_clip_norm` | float | 1.0 | Gradient clipping; 0 disables it |
| `precision` | str | fp32 | `fp32` / `fp16` / `bf16` (fp16 requires CUDA) |
| `accumulate_grad_batches` | int | 1 | Gradient accumulation; enlarge the effective batch when memory runs short |
| `focal_gamma` | float | 0.0 | Switch to focal loss when >0 |
| `augment.*` | | | see below |

`train.augment`: `random_resized_crop`(true) / `crop_scale`([0.7,1.0]) /
`horizontal_flip`(true) / `color_jitter`(0.2) / `rotation_deg`(8.0) /
`gaussian_blur`(0.1) / `randaugment`(false)

> **The model-selection criterion is validation Macro-F1**, not Accuracy. With imbalanced
> classes, Accuracy hides the degradation of the small classes (steps 5 and 6).

## eval

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `splits` | list[str] | [val, test] | Splits to evaluate |
| `metrics` | list[str] | see the file | For the record (the actual metrics are fixed in core/metrics) |
| `save_confusion_matrix` | bool | true | Write a PNG |
| `save_predictions` | bool | true | Write per-frame JSONL (for reviewing failure cases) |
| `bootstrap_ci` | bool | false | Bootstrap confidence interval (to show that an improvement is not fluctuation) |
| `bootstrap_samples` | int | 1000 | Number of bootstrap resamples |
| `extra_datasets` | list[str] | [] | Names of cross-scenario datasets, e.g. `[metc]` |

## assess (completeness judgement thresholds)

| Key | Default | Description |
| --- | --- | --- |
| `smooth_window` | 9 | Majority-vote sliding window (frames) |
| `min_confidence` | 0.4 | Frames below this confidence take no part in the vote |
| `min_segment_frames` | 5 | Minimum number of frames in a valid segment |
| `min_segment_s` | 1.0 | Minimum number of seconds in a valid segment |
| `min_total_duration_s` | 40.0 | Lower bound on the total wash duration (WHO recommends 40—60 s) |
| `reference_total_duration_s` | 50.0 | Reference total duration used for scoring |
| `min_step_duration_s` | 3.0 | Lower bound on a single step's duration (applies when `duration_check=seconds`) |
| `step_duration_ratio` | 0.4 | Minimum fraction of the "fair share" a single step must reach (applies with `ratio`) |
| `allow_repeats` | false | Whether repeated steps are allowed |
| `missing_tolerance` | 0 | How many missed steps still count as "essentially complete" |
| `duration_check` | ratio | `seconds` / `ratio` / `none` |
| `order_check` | true | Whether the order is checked |
| `require_faucet_events` | false | Whether tap on/off events must be detected |
| `report_language` | zh | Report language |

For the judgement criteria in detail see [`PROTOCOL.md`](PROTOCOL.md).

## infer

| Key | Default | Description |
| --- | --- | --- |
| `mode` | clip | `frame` / `clip` / `hybrid` |
| `temporal_apply` | true | Whether a moving average over probabilities is applied |
| `smooth_window` | 9 | Moving-average window |
| `save_frame_predictions` | true | Write per-frame JSONL |
| `save_overlay_video` | false | Write an overlay preview |
| `batch_size` | 64 | Inference batch |
| `tta` | false | Test-time augmentation (slower) |

---

## Quick reference for common edits

```bash
# out of GPU memory
train.batch_size=8 train.accumulate_grad_batches=4
# switch to CPU for a quick pipeline check
runtime.device=cpu runtime.num_workers=0
# Windows DataLoader error
runtime.num_workers=0
# try TCN instead of GRU
model.temporal.kind=tcn train.mode=clip
# turn augmentation off for an ablation (does augmentation actually help?)
train.augment.random_resized_crop=false train.augment.horizontal_flip=false \
train.augment.color_jitter=0.0 train.augment.rotation_deg=0.0
# judge only missed steps / order, ignoring duration
assess.duration_check=none
# allow going back to redo a step
assess.order_check=false
```
