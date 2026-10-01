> **English** | [中文](/docs/zh/RULES_CARD.md)

# Getting started and change-rule cheat sheet

> This page is the condensed version of [`CONTRIBUTING.md`](../CONTRIBUTING.md).
> **When the card and the full document disagree, the full document wins**; a card that
> was not updated is a documentation defect — send a PR to fix it.
> Print it out and pin it to your desk.

---

## 1. Three commands to get started

```bash
conda env create -f environment.yml && conda activate handwash
pip install -e ".[all]" && python -m pre_commit install
python scripts/doctor.py                 # self-check; go on only when everything is green
python scripts/train_model.py --config configs/experiments/smoke.yaml   # 30-second smoke test
```

> You can run without `pip install -e .`: the scripts under `scripts/` bootstrap their own
> import paths.

---

## 2. Which level am I allowed to touch? (always the first question)

| What I want to touch | Level | What it needs |
| --- | --- | --- |
| Create an experiment config, run experiments, record results | **L3 free** | CI green |
| Add a model file / dataset adapter / script | **L3 free** | CI green |
| Change the **implementation** of an existing function | **L2 restricted** | 1 maintainer review |
| Change `tests/`, `docs/`, `configs/experiments/` | **L2/L3** | Usually free |
| Change a **function signature / return structure / config key name / decision rule** | **L1 frozen** | **Write the RFC first**, then change code |
| Change the semantics of anything under `core/`, the label space order, or the split rules | **L1 frozen** | **Write the RFC first**, then change code |

**Mantra: adding a file is L3, changing an implementation is L2, changing an interface is L1.**

The L1 file list (pause before you touch any of them):
```
src/handwash/core/labels.py  schema.py  config.py  protocol.py  registry.py
src/handwash/errors.py  paths.py  logging.py
src/handwash/io/split.py
configs/config.yaml
```

---

## 3. The ten Iron Rules

| # | Rule |
| --- | --- |
| 1 | Touch only the modules you own; open an issue before crossing module boundaries |
| 2 | UTF-8 + LF + 4 spaces (2 spaces for YAML/JSON) |
| 3 | Tunable parameters go into `configs/`; no magic numbers in code |
| 4 | Update `pyproject.toml` **and** `environment.yml` for any new dependency |
| 5 | `core` must not import torch; `io` must not import models (layered dependencies) |
| 6 | Keep data, training weights, and outputs out of Git; root `exp.pt` is the demo exception |
| 7 | Add a new workflow to both `Makefile` and `scripts/` |
| 8 | No `print` (except in the CLI), no `sys.exit` (in library code) |
| 9 | Raise `HandwashError` subclasses with a message stating field / expected / actual |
| 10 | Hand-washing step strings appear only in `core/labels.py` |

---

## 4. The four absolutes

1. **Never let the same original video cross train/test** (data leakage → every conclusion
   is void). The pipeline is fixed as: **scan → split by video → frame extraction**; the
   order cannot be reversed.
2. **Never reorder the classes in the label space** (existing checkpoints would silently
   shift). Append to the end only.
3. **Never write decision rules into business code** (missed step / order / duration live
   only in `core/protocol.py`).
4. **Never overwrite someone else's commits with `git push --force`** (force-pushing `main`
   is forbidden).

---

## 5. What to run before committing

```bash
python -m pre_commit run --all-files     # fast: format + structure + quick unit tests
make check                               # full: exactly the same as CI
```

Skipping the pre-commit install is fine, but the PR must include the `make check` output.

---

## 6. PR checklist (copy into the PR description)

```markdown
- [ ] The files I changed belong to my modules, or I have maintainer approval
- [ ] I did not modify L1 frozen files (if I did, RFC link: ____)
- [ ] New parameters went into configs/, and I added no new magic numbers
- [ ] New/changed modules have matching tests
- [ ] `make check` passes (paste the output)
- [ ] I ran the smallest affected workflow (paste command and result)
- [ ] If I changed models/data, the experiment is recorded in docs/EXPERIMENTS.md (with config_hash)
- [ ] If historical results are affected, I listed the "experiments that must be re-run"
- [ ] No data, training weights, or outputs were committed; root `exp.pt` is the only weight exception
```

---

## 7. Seeing an error? Check here first

| Error | First reflex |
| --- | --- |
| `ConfigError: unknown config key` | Typo in the key name, or you added a new parameter without registering it in `core/config.py` |
| `DataLeakageError` | The same original video crossed splits; check `split.group_key` |
| Accuracy > 0.98 | Almost certainly leakage, not a good model |
| Abnormally low accuracy | `normalize` does not match `arch` (YOLO uses `zero_one`) |
| `ModuleNotFoundError: handwash` | You did not run `pip install -e .`, and you bypassed the `scripts/` entry points |
| DataLoader error on Windows | `runtime.num_workers: 0` (set in `configs/local.yaml`) |
| Endless traceback | Read the last line first, then the output of `handwash doctor` |

---

## 8. What every experiment run must record

```
run_name          outputs/<run_name>/
config_hash       the config_hash inside resolved_config.yaml
dataset & split   split_report.json
metrics           eval/eval_*.json + confusion_matrix_*.png
one-line verdict  write it into docs/EXPERIMENTS.md
```

**A result without a `config_hash` is not allowed into any report.**

---

## 9. I want to add something new (shortest path)

| Want to add | Steps |
| --- | --- |
| **A new model** | Register `models/xxx.py` with `@register_model("name")` → add the import to `register_all()` in `factory.py` → `configs/models/xxx.yaml` → shape test |
| **A new dataset** | Add the alias to `_DATASET_ALIASES` in `core/labels.py` (append) → `configs/data/xxx.yaml` → register it under `datasets` in `configs/config.yaml` → `docs/DATA.md` |
| **A new experiment** | `cp configs/config.yaml configs/experiments/expNN_description.yaml`, and write only the keys that differ from the defaults |
| **A new metric** | **Append** a function in `core/metrics.py` → wire it into `pipelines/evaluate.py` → unit-test it against a hand-computed example |
| **A new command** | Add the subcommand in `cli.py` → add the target in `Makefile` → add a thin wrapper under `scripts/` (three places in sync) |
| **A threshold change** | Change only the `assess` section in `configs/` (**do not touch the code**) |

---

## 10. Give yourself a visible acceptance test

Before the defense you must be able to demo these three things live:

```bash
# 1) One complete, correct hand-washing clip -> judged complete, order correct
python scripts/run_assess.py --video <complete_video>.mp4

# 2) One deliberately skipped step -> points out exactly which step is missing
python scripts/run_assess.py --video <missed_step_video>.mp4

# 3) One deliberately reordered clip -> points out the order anomaly
python scripts/run_assess.py --video <reordered_video>.mp4
```

For how to prepare and annotate these three clips, see [`SELF_RECORDING.md`](SELF_RECORDING.md) (in Chinese).

---

**Last word: the only purpose of these rules is to make sure "anyone who clones the repo can
reproduce anyone else's results."** If a rule blocks the right thing, do not bypass it
quietly — explain why in the PR and let everyone change the rule together.
