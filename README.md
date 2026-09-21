> **English** | [中文](/README.zh-CN.md)

# WHO Six-Step Hand Hygiene: Action Recognition and Completeness Assessment

> Video-based recognition of the WHO six-step hand-washing procedure. The system predicts
> which step is being performed in each frame and then checks for **missed steps,
> out-of-order steps, and steps that are too short**.
> The project performs **compliance analysis only — it makes no medical diagnosis.**

- **Primary model**: YOLO26n-cls (Ultralytics classification model) + a GRU/TCN temporal head
- **Baselines**: MobileNetV2 (matching the open-source baseline), YOLOv8n-cls
- **Data**: PSKUS (main), METC (cross-scenario), Kaggle (rapid prototype), self-recorded clips (final validation)

---

## Table of contents

1. [Five-minute setup](#1-five-minute-setup)
2. [Three commands you can run right now](#2-three-commands-you-can-run-right-now)
3. [What is in the project](#3-what-is-in-the-project)
4. [Read this before changing any code](#4-read-this-before-changing-any-code)
5. [FAQ](#5-faq)
6. [Documentation index](#6-documentation-index)

---

## 1. Five-minute setup

```bash
git clone <repo-url> && cd "Group Project PRS"

conda env create -f environment.yml     # Python 3.12 + PyTorch + dependencies
conda activate handwash
pip install -e ".[all]"                 # install this repo into the env (editable mode)
python -m pre_commit install            # install the pre-commit hooks

python scripts/doctor.py                # self-check: deps, config, data, GPU, model weights
```

Once `doctor` is all green (a non-blocking WARN or two is fine), your environment is ready.
**On Windows** you can instead run `pwsh scripts/setup_windows.ps1`, which performs all of the above.

> No conda? Install Miniconda from <https://docs.conda.io/en/latest/miniconda.html>.
> Prefer not to use conda? `python -m venv .venv` + `pip install -e ".[all]"` works too,
> but record that choice in `docs/EXPERIMENTS.md` and remember that `environment.yml`
> remains the team baseline.

## 2. Three commands you can run right now

```bash
# 1. 30-second smoke test: no dataset needed, verifies the data -> model -> metrics path
python scripts/train_model.py --config configs/experiments/smoke.yaml

# 2. Assess one video and print the WHO completeness report (missed / out-of-order / too short)
python scripts/run_assess.py --video data/external/self_recorded/full/demo.mp4

# 3. Run a real training pipeline on the small Kaggle dataset (download first, see docs/DATA.md)
python scripts/prepare_data.py  --config configs/data/kaggle.yaml
python scripts/train_model.py   --config configs/experiments/exp01_baseline_frame.yaml
python scripts/evaluate_model.py
```

Everything is written under `outputs/<run_name>/`:

```
resolved_config.yaml          the effective config + config_hash (cite this in your report)
history.json / run_log.jsonl  per-epoch metrics
models/best.pt                best checkpoint, selected by validation Macro-F1
eval/eval_*.json              metrics per split
eval/confusion_matrix_*.png   confusion matrices
assess/<clip>.md              the completeness report (paste-ready)
```

## 3. What is in the project

```
src/handwash/
  core/       contract layer: label space, data contracts, config, metrics, WHO rules
  io/         video decoding, manifest, source-video-safe splitting
  data/       datasets, preprocessing and augmentation, synthetic data (for smoke tests)
  models/     YOLO26n-cls adapter, MobileNetV2/ResNet baselines, GRU/TCN temporal heads
  pipelines/  the five workflows: prepare / train / evaluate / infer / assess
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
| `run_assess.py` | video -> **missed / out-of-order / duration** report | demos and final validation |

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

The real manifest does not exist, so the code fell back to synthetic data to verify the pipeline.
**Metrics from synthetic data must never go into the report.**
Run `prepare_data.py` first; see `docs/DATA.md`.
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

**No, and you must not.** `data/`, `models/`, and `outputs/` are already in `.gitignore`.
See `docs/DATA.md` for how to obtain the datasets; checkpoints are shared through
`outputs/<run>/models/` or a shared drive.
</details>

## 6. Documentation index

| Document | Contents |
| --- | --- |
| [`docs/PROJECT_PLAN_4_WEEK.md`](docs/PROJECT_PLAN_4_WEEK.md) | **Four-week team plan (4 members)**: roles, week-by-week tasks, grading traceability, risks |
| [`docs/RULES_CARD.md`](docs/RULES_CARD.md) | **One-page cheat sheet** — read this before changing code |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **Modification rules (required reading)**: Iron Rules, change levels, commit and PR flow |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layer diagram, the five contracts, RFC process, extension guides |
| [`docs/DATA.md`](docs/DATA.md) | The five public datasets: sources, layouts, downloads, preparation |
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
