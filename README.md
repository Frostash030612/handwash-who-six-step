> **English** | [中文](/README.zh-CN.md)

# WHO Six-Step Hand Hygiene: Action Recognition and Completeness Assessment

> **Current delivery plan (Chinese):** [step-by-step implementation plan](docs/IMPLEMENTATION_PLAN.zh-CN.md)
> covers dataset quality, cloud YOLO classification training, and local live camera inference.
> Earlier GRU/TCN and PE-based designs remain optional research experiments.

> A live YOLO image classifier predicts the current hand-washing action from each camera
> frame. The prediction stream then checks for **missed steps, out-of-order steps, and
> steps that are too short**.
> The project performs **compliance analysis only — it makes no medical diagnosis.**

- **Primary model**: YOLO classification, trained on cleaned data in the cloud and run locally frame by frame
- **Live display**: past-frame probability averaging for stability; final six-step rules use the recorded prediction timeline
- **Optional experiments**: MobileNetV2, GRU/TCN and PE-based temporal segmentation
- **Data**: PSKUS (main), METC (cross-scenario), Kaggle (rapid prototype), self-recorded clips (final validation)

---

## Table of contents

1. [Five-minute setup](#1-five-minute-setup)
2. [From data to live camera](#2-from-data-to-live-camera)
3. [What is in the project](#3-what-is-in-the-project)
4. [Read this before changing any code](#4-read-this-before-changing-any-code)
5. [FAQ](#5-faq)
6. [Documentation index](#6-documentation-index)

---

## 1. Five-minute setup

On **macOS (local inference)**, run from the repository directory:

```bash
conda create -n handwash python=3.11 -y
conda activate handwash
python -m pip install -e ".[torch,yolo,video]"
python -m handwash.cli doctor
```

`environment.yml` includes NVIDIA CUDA dependencies for a compatible Linux training machine;
do not use it directly on macOS. On a Linux cloud machine with a compatible CUDA setup:

If `python --version` still reports Python 2 after activating conda, run the commands below
with `"$CONDA_PREFIX/bin/python"` instead of `python`.

```bash
conda env create -f environment.yml
conda activate handwash
python -m pip install -e ".[all]"
python -m handwash.cli doctor
```

Once `doctor` is all green (a non-blocking WARN or two is fine), your environment is ready.
**On Windows**, see `pwsh scripts/setup_windows.ps1`.

> No conda? Install Miniconda from <https://docs.conda.io/en/latest/miniconda.html>.
> Prefer not to use conda? `python -m venv .venv` + `pip install -e ".[all]"` works too,
> but record that choice in `docs/EXPERIMENTS.md` and remember that `environment.yml`
> remains the team baseline.

## 2. From data to live camera

### Try the bundled exp.pt camera demo

The repository includes the current seven-class Ultralytics frame classifier `exp.pt`. When
`exp_temporal_head.pt` is present beside it, the demo also loads its GRU temporal head. The
training NDJSON export is excluded; obtain the final dataset from its public source below.
To launch the demo:

```bash
python scripts/run_camera.py --demo-exp
```

Open `http://127.0.0.1:8765/`, select a camera, and start recognition. Use
`--no-temporal-head` to compare the frame-only path. The page marks this as a demo; its report
is saved under `outputs/exp_demo/camera/` and is not a final result. Before publishing, use
`git add .` and inspect `git status --short`. The root demo weights `exp.pt` and
`exp_temporal_head.pt` are included; do not upload the NDJSON export, training weights,
`deliverables/`, or `outputs/`. `.gitignore` does not remove files already present in Git history.

### Final dataset and cloud training

```bash
# 1. Raw data is not bundled. Download PSKUS, then clean and audit the videos/labels.
python scripts/download_data.py --dataset pskuss --all --extract
# For a pipeline trial only, use --files DataSet4.zip --extract instead.
# 2. Create the source-video-grouped frame manifest.
python scripts/prepare_data.py config=configs/data/pskuss.yaml --inspect
python scripts/prepare_data.py config=configs/data/pskuss.yaml

# 3. On the cloud training machine, train the frame classifier and evaluate it.
python scripts/train_model.py config=configs/experiments/live_yolo_frame.yaml --evaluate

# 4. Copy the project's best.pt and matching config to the deployment computer.
# 5. Connect a USB camera and open http://127.0.0.1:8765/ in a local browser.
python scripts/run_camera.py config=configs/experiments/live_yolo_frame.yaml \
  --checkpoint path/to/best.pt
```

The camera page needs a trained project-format checkpoint. The code is present; model quality
and target-computer latency require validation on the final dataset and hardware.

Everything is written under `outputs/<run_name>/`:

```
resolved_config.yaml          the effective config + config_hash (cite this in your report)
history.json / run_log.jsonl  per-epoch metrics
models/best.pt                best checkpoint, selected by validation Macro-F1
eval/eval_*.json              metrics per split
eval/confusion_matrix_*.png   confusion matrices
assess/<clip>.md              the completeness report (paste-ready)
camera/<session>/             live predictions and final report
```

## 3. What is in the project

```
src/handwash/
  core/       contract layer: label space, data contracts, config, metrics, WHO rules
  io/         video decoding, manifest, source-video-safe splitting
  data/       datasets, preprocessing and augmentation, synthetic data (for smoke tests)
  models/     YOLO classification adapter; experimental baselines and temporal heads
  pipelines/  prepare / train / evaluate / infer / assess / live
  camera_app.py + static/   local HTTP service and camera web page
  cli.py      command-line entry point (handwash <subcommand>)
scripts/      thin wrapper scripts + structure guard + one-shot Windows setup
configs/      config.yaml + data/ + models/ + experiments/
docs/         data card, assessment criteria, config reference, experiment log, architecture, RFCs
tests/        unit tests mirroring the source layers
```

The difference between the commands people confuse most:

| Command | What it does | When to use it |
| --- | --- | --- |
| `prepare_data.py` | scan -> **split by source video** -> extract frames -> build manifest | after obtaining new data |
| `train_model.py` | train and save a checkpoint | after changing a model or config |
| `evaluate_model.py` | compute metrics on val/test/external | after training |
| `handwash infer` | video -> per-frame action predictions | inspect frame-level model output |
| `run_assess.py` | video -> **missed / out-of-order / duration** report | demos and final validation |
| `handwash camera` | USB camera -> live YOLO classification -> final report | local demonstration after cloud training |

Example: `python -m handwash.cli infer --video demo.mp4 --checkpoint outputs/run/models/best.pt`.

## 4. Read this before changing any code

**This repository has explicit modification rules. They are not suggestions.**

- [`docs/RULES_CARD.md`](docs/RULES_CARD.md) — the **one-page cheat sheet** (worth printing and
  pinning up): three commands to start, which change level applies to you, the Iron Rules,
  the four "never do this", and the PR checklist.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — the **full rules**. Iron Rules, change levels
  (L1 frozen / L2 restricted / L3 free), commit conventions, PR checklist, and a
  troubleshooting table.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layer diagram, the five key contracts,
  the RFC process, and extension guides.

The one-line version: **adding files is L3 (change freely), changing an implementation is L2
(needs review), changing an interface is L1 (needs an RFC)**. The `core/` directory of `handwash`
is L1.

The rules are enforced by three machine checks, not by good intentions:

```bash
python scripts/check_structure.py       # layered dependencies / hard-coded paths / print
python -m pytest -m "not integration and not slow and not gpu"
python -m pre_commit run --all-files
```

## 5. FAQ

<details>
<summary><b>Training says "using synthetic data"?</b></summary>

The config explicitly selects `dataset.name: synthetic` for a smoke run. Missing real-data
manifests cause an error; they do not trigger a synthetic fallback.
**Metrics from synthetic data must never go into the report.** For real training, run
`prepare_data.py` first; see `docs/DATA.md`.
</details>

<details>
<summary><b>Accuracy is suspiciously high (&gt;0.98)?</b></summary>

That is almost always data leakage: frames from the same source video ended up in both the
training and the test split. Check that `split.group_key` is `original_video` and that the
pipeline does **frame extraction after splitting**. `io/manifest.py` and `io/split.py` each
run a check that will fail loudly in this case.
</details>

<details>
<summary><b>ultralytics is missing / the yolo26 weights fail to download?</b></summary>

The primary model needs `ultralytics`: run `conda env update -f environment.yml`.
If `yolo26n-cls.pt` is not available yet, start with an existing weight:

```bash
python scripts/train_model.py model.arch=yolon-cls model.pretrained=yolov8n-cls.pt
```

**Switching weights is a config change, not a code change**, and the report must state which
weights were actually used.
</details>

<details>
<summary><b><code>ConfigError: unknown configuration key</code>?</b></summary>

A key is misspelled, or you added a new parameter without adding it to `core/config.py`.
This is deliberate: **a misspelled key is never silently ignored.** Cross-check with `docs/CONFIG.md`.
</details>

<details>
<summary><b>DataLoader errors on Windows?</b></summary>

Set `runtime.num_workers: 0`. Put it in `configs/local.yaml` rather than editing the shared config.
</details>

<details>
<summary><b>Do I need to commit data or model weights?</b></summary>

Do not commit data or training weights. The root `exp.pt` is the sole demo exception.
`data/`, `models/`, `outputs/`, and other `*.pt` files are ignored. See `docs/DATA.md` for
dataset sources; provide a trained project checkpoint with `--checkpoint` for the final workflow.
</details>

## 6. Documentation index

| Document | Contents |
| --- | --- |
| [`docs/IMPLEMENTATION_PLAN.zh-CN.md`](docs/IMPLEMENTATION_PLAN.zh-CN.md) | **Current step-by-step delivery plan**: cleaned data → cloud training → local camera web app |
| [`docs/PROJECT_PLAN_4_WEEK.md`](docs/PROJECT_PLAN_4_WEEK.md) | Archived course schedule and grading requirements |
| [`docs/PROJECT_PROPOSAL.md`](docs/PROJECT_PROPOSAL.md) | **Proposal content pack** — ready-to-paste text for every template field (due 30 Sep) |
| [`docs/RULES_CARD.md`](docs/RULES_CARD.md) | **One-page cheat sheet** — read this before changing code |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **Modification rules (required reading)**: Iron Rules, change levels, commit and PR flow |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layer diagram, the five contracts, RFC process, extension guides |
| [`docs/DATA.md`](docs/DATA.md) | The five public datasets: sources, layouts, downloads, preparation |
| [`docs/DATA_COLLABORATION.md`](docs/DATA_COLLABORATION.md) | **How four people share 2–17 GB without uploading it anywhere** |
| [`docs/CONFIG.md`](docs/CONFIG.md) | Every configuration key, plus recipes for common changes |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | Assessment criteria: exact definitions of missed / out-of-order / insufficient duration |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Experiment log (in Chinese — internal team working document) |
| [`docs/SELF_RECORDING.md`](docs/SELF_RECORDING.md) | Self-recording protocol and annotation table (in Chinese) |
| [`CHANGELOG.md`](CHANGELOG.md) | Breaking changes and migration notes (in Chinese) |
| [`tests/README.md`](tests/README.md) | How to run the tests, marker meanings, rules for adding tests (in Chinese) |

> Chinese mirrors of the primary documents live in [`docs/zh/`](docs/zh/) and
> [`README.zh-CN.md`](README.zh-CN.md); use the language switcher at the top of each page.
> Documents without an English version are internal team working documents.

---

## References

1. Hand-Washing Video Dataset Annotated According to WHO Guidelines (2021) —
   <https://doi.org/10.3390/data6040038> (the PSKUS dataset paper)
2. Towards Automated Hand Hygiene Assessment in Hospitals (2022) —
   <https://www.edi.lv/wp-content/uploads/2022/12/main.pdf>
3. Learning to Recognize Hand-Washing Activities in Hospital Settings (2020) —
   <https://arxiv.org/abs/2011.11383>
4. Hand Washing Gesture Recognition Using a Synthetic Dataset (2025) —
   <https://doi.org/10.3390/jimaging11070208>
5. Open-source baseline code — <https://github.com/edi-riga/handwash>
6. WHO How to Handwash — <https://www.who.int/publications/m/item/how-to-handwash>
7. Ultralytics YOLO26 — <https://docs.ultralytics.com/models/yolo26>
