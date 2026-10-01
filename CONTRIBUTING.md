> **English** | [中文](/docs/zh/CONTRIBUTING.md)

# Modification Rules (CONTRIBUTING)

> **The one-line rule: before you touch code, check this table. Which tier are you changing,
> L1, L2, or L3?**

This file is the "law" of this repository. It solves the most common failure mode in a group
project: **five people each change a little something, and two weeks later nobody can run the
full pipeline, and nobody can say which result came from which code.**

- The point of these rules is not to constrain you. It is to guarantee that **anyone who clones
  the repo can reproduce anyone else's results.**
- The cost of breaking a rule is not "getting told off". It is that **the whole team's experimental
  conclusions are void.**
- These rules are enforced by three machine checks: `scripts/check_structure.py`, pre-commit, and
  CI. They do not depend on anyone's good intentions.

---

## 0. Five-minute onboarding (everything a new team member does on day one)

```bash
git clone <repo>                       # 1) get the code
conda env create -f environment.yml    # 2) create the environment (Python 3.12 + torch + deps)
conda activate handwash
pip install -e ".[all]"                # 3) install this repository into the environment
python -m pre_commit install           # 4) install the pre-commit hook (= make hooks)
python -m handwash.cli doctor          # 5) self-check: what is missing, what is misconfigured
make train-smoke                       # 6) run the data -> model -> metrics path in 30 seconds
```

Only when all six steps succeed is your environment "ready". If any step fails, **read the output
of `handwash doctor` first**; do not guess and start editing code.

---

## 1. The ten Iron Rules (if you remember nothing else, remember these ten)

| # | Rule | Why | Enforced by |
| --- | --- | --- | --- |
| **R1** | Change only the modules you own; open an issue before any cross-module change | Stops two people refactoring the same code at once | CODEOWNERS review |
| **R2** | File encoding UTF-8, line endings LF, indent 4 spaces (2 spaces for YAML/JSON) | Otherwise every diff rewrites the whole file and cannot be reviewed | `.editorconfig` + pre-commit |
| **R3** | Every tunable parameter goes into `configs/`; no magic numbers in code | Results must be reproducible and comparable | CI archcheck + review |
| **R4** | A new dependency must be added to `pyproject.toml` and `environment.yml` together | Otherwise others cannot install it, or get different results | review |
| **R5** | Respect layered dependencies: `core` must not import `torch`, `io` must not import `models` | The contract layer stays unit-testable and reusable; a model change does not disturb the data | `scripts/check_structure.py` |
| **R6** | Keep data, training weights, and outputs out of Git; root `exp.pt` is the demo exception | A 20GB dataset would blow up the repository | `.gitignore` + pre-commit large-file check |
| **R7** | A new workflow must be added to both the `Makefile` and the CLI, with matching names | Team members should not have to remember two sets of commands | review |
| **R8** | Do not print runtime information with `print`; do not raise errors with `sys.exit` | Logs must be collectable in one place, errors must be handled in one place | archcheck |
| **R9** | Raise a `HandwashError` subclass, and state in the message which file/field/expected/actual | A good error message is the best documentation | review |
| **R10** | Washing-step strings appear exactly once, in `core/labels.py` | Otherwise labels shift, every metric is ruined, and the cause is very hard to find | review + unit tests |

---

## 2. Change tiers: first decide which tier you are touching

### L1 Frozen zone (L1) - changes require an RFC (write the document first, then change code)

```
src/handwash/core/labels.py      label space (once the class order is frozen, old weights stay meaningful)
src/handwash/core/schema.py      cross-layer data contract (fields / meaning / units)
src/handwash/core/config.py      configuration structure (key names are the team-wide "interface")
src/handwash/core/protocol.py    completeness decision rules (the definition behind every verdict)
src/handwash/core/registry.py    registration mechanism
src/handwash/errors.py  paths.py  logging.py
configs/config.yaml              shared default configuration
src/handwash/io/split.py         data split rules (change the split and all historical metrics become incomparable)
```

**The standard process for an L1 change:**

1. Append an RFC entry to section 5 of `docs/ARCHITECTURE.md` (a template is provided) and state
   clearly: current state -> why it must change -> who is affected -> migration plan -> who approves.
2. Get approval from at least one maintainer, and paste the RFC link into the PR description.
3. Update `CHANGELOG.md` at the same time (mark breaking changes as **BREAKING**).
4. If the change affects results that already exist, **state explicitly in the PR which historical
   experiments must be re-run.**

### L2 Restricted zone (L2) - requires review by 1 maintainer

```
src/handwash/io/**             src/handwash/data/**
src/handwash/models/**         src/handwash/pipelines/**
scripts/**                     tests/**
pyproject.toml  environment.yml  Makefile
.pre-commit-config.yaml  .github/**  docs/** (except ARCHITECTURE/PROTOCOL)
```

You may implement freely, but you **must not change the external contract**:
function signatures, return structures, config key names, and log semantics all stay as they are.
If they need to change, follow the L1 process.

### L3 Free zone (L3) - change anything you like, as long as CI passes

```
configs/experiments/**.yaml    your own experiment configs
configs/models/**.yaml         configs for new models
configs/data/**.yaml           configs for new datasets
docs/EXPERIMENTS.md            experiment result records (in Chinese)
outputs/  data/  models/       local artifacts (never in Git)
notebooks/**.ipynb             personal exploration (clear outputs before committing)
```

**This is where 90% of your work happens.** Want to try a new idea? Create a new
`configs/experiments/expNN_description.yaml`, run it, write the result into `docs/EXPERIMENTS.md`
(in Chinese). No one's approval needed.

> The rule of thumb: **adding a file is L3, changing an interface is L1, changing an implementation
> is L2.**

---

## 3. How to add something new (just copy these patterns)

### 3.1 Add a new model

```python
# src/handwash/models/my_model.py
from handwash.core.registry import register_model
from handwash.models.base import BaseClassifier

@register_model("my-model")            # (1) registration is required; the name is model.arch in the config
class MyModel(BaseClassifier):
    arch_name = "my-model"

    def __init__(self, num_classes: int, *, pretrained=True, dropout: float = 0.2, **kwargs):
        super().__init__(num_classes, dropout=dropout)
        ...                            # (2) build the network yourself

    @property
    def feature_dim(self) -> int: ...  # (3) the temporal head needs this

    def _forward_logits(self, x):      # (4) must return (B, T, C)
        ...                            #     a single-frame model still needs a time dimension

    def embed(self, x):                # (5) only needed if you want two-stage GRU/TCN training
        ...
```

Then import your module inside `register_all()` in `src/handwash/models/factory.py`, add a config
under `configs/models/`, and add a test under `tests/unit/models/` that checks "it runs a forward
pass and the shapes are right". **Do not change the `BaseClassifier` interface** (that is L1).

### 3.2 Add a new dataset

1. **Change only `_DATASET_ALIASES` in `core/labels.py`**: map the dataset's raw label names onto
   the canonical `Step`. This step is the easiest to get wrong and the most important. Map it wrong
   and whatever the model learns is misaligned. If the dataset has new classes (not one of the six
   steps), add another namespace in `LABEL_SPACES`, and **append it at the end only** (order is the
   channel index; inserting invalidates old weights, which makes it an L1 change).
2. Write the data paths and the `assess` thresholds in `configs/data/<name>.yaml`.
3. Register the path under the `datasets:` section of `configs/config.yaml`.
4. If the directory structure differs from existing adapters, change the adapter function in
   `pipelines/prepare.py` (L2).
5. Add to `docs/DATA.md` (in Chinese): source link, scale, directory structure, download command.

### 3.3 Add a new experiment

```bash
cp configs/config.yaml configs/experiments/exp05_my_idea.yaml
# write only the keys that differ from the defaults (overlay semantics, do not copy the whole config)
python -m handwash.cli train config=configs/experiments/exp05_my_idea.yaml
```

When it finishes, record the result in the table in `docs/EXPERIMENTS.md` (in Chinese)
(**config_hash is mandatory**).

### 3.4 Add a new evaluation metric

1. Put pure computation in `core/metrics.py` (L1, but **append** functions only; do not change the
   semantics of existing ones).
2. Add it to the legal values of `EvalConfig.metrics`, and wire it up in `pipelines/evaluate.py`.
3. Add a unit test: give an example you can compute by hand and assert the exact number.

---

## 4. Directory and naming conventions

```
src/handwash/
  paths.py logging.py errors.py     L0 infrastructure (standard library only)
  core/                             L1 contract: schema / labels / config / metrics / protocol
  io/                               L2 persistence and decoding: manifest / split / video / utils
  data/                             L2 datasets and preprocessing
  models/                           L3 models (the only place allowed to import torch/ultralytics)
  pipelines/                        L4 orchestration: prepare / train / evaluate / infer / assess
  cli.py cli_doctor.py              L4 command-line entry points
scripts/                            thin wrapper scripts (multi-step combinations, demos)
configs/                            config.yaml + data/ + models/ + experiments/
docs/                               documentation
tests/                              tests mirroring the src/ layering
data/ models/ outputs/              artifacts (never in Git)
```

Naming conventions (enforced both by machine checks and by review):

| Object | Convention | Example |
| --- | --- | --- |
| module / function / variable | `snake_case` | `build_dataset`, `macro_f1` |
| class | `PascalCase` | `TemporalClassifier` |
| constant | `UPPER_SNAKE` | `CANONICAL_STEPS` |
| private | prefix `_` | `_forward_logits` |
| config key | `snake_case`, **character-for-character identical** to the dataclass field | `train.epochs` |
| class name | appears only in `core/labels.py` | `step_1_palm_to_palm` |
| test file | `test_<module under test>.py` | `test_protocol.py` |
| experiment config | `expNN_description.yaml` | `exp02_yolo26n_gru.yaml` |

---

## 5. Commits and branches (used every day)

### Branches

```
main                  always runnable, always green in CI. Direct pushes are forbidden (pre-commit blocks them)
feat/<module>/<short> new feature, e.g. feat/models/tcn-head
fix/<module>/<short>  bug fix, e.g. fix/io/manifest-dup-frame
exp/<your-name>/<id>  running experiments (usually only touches configs/experiments/)
docs/<short>          documentation only
```

### Commit messages (Conventional Commits, description in English)

```
<type>(<scope>): <one-line description>

scope values: core / io / data / models / pipelines / configs / docs / tests / scripts
type values: feat / fix / refactor / docs / test / chore / perf / BREAKING

Examples:
feat(models): add a TCN temporal head and register the tcn architecture
fix(core): fix division by zero in the duration check when total_wash_s=0
docs(configs): note that the YOLO adapter must use zero_one normalization
BREAKING(core): add an other class to the pskuss label space (old weights need retraining)
```

### Run before every commit (pick one of three; the first is recommended)

```bash
python -m pre_commit run --all-files    # fast: formatting + structure checks + quick unit tests
make check                              # full: lint + archcheck + typecheck + test
python -m pytest -m "not integration and not slow and not gpu"
```

### PR checklist (copy into the PR description and tick every line)

```markdown
- [ ] The files I changed belong to the modules I own, or I have the maintainer's approval
- [ ] I did not modify any L1 frozen file (if I did, the RFC link is attached: ____)
- [ ] Every new parameter went into configs/; no new magic numbers in code
- [ ] Every new or changed module has a matching test (same-named file under tests/)
- [ ] `python -m pre_commit run --all-files` passes
- [ ] I ran the smallest affected workflow locally (paste commands and results: ____)
- [ ] If I changed a model or data, I recorded the experiment in docs/EXPERIMENTS.md (in Chinese, with config_hash)
- [ ] If historical results are affected, I listed the "experiments that need re-running" in the PR
- [ ] I committed no data, training weights, or outputs; root `exp.pt` is the only weight exception
```

---

## 6. Common errors and the correct handling (check here first when you see an error)

| Symptom | Real cause | Correct handling |
| --- | --- | --- |
| `ConfigError: unknown config key` | A config key is misspelled, or you added a new parameter to the YAML without adding it to the dataclass | Check the field names in `core/config.py`; new parameters go through an RFC |
| `DataLeakageError` | The same raw video landed in two splits | Check `split.group_key=original_video`; **you must split before frame extraction** |
| `ManifestError: missing required column` | You are using an old manifest | Delete the old `data/processed/*/manifest.csv` and re-run `prepare` |
| Accuracy suspiciously high (>0.98) | Very likely data leakage, or the test set leaked into training | Check the split; see the "self-check list" in `docs/DATA.md` (in Chinese) |
| Accuracy suspiciously low | Wrong normalization (imagenet for YOLO), or misaligned label mapping | Run `handwash doctor`; check `core/labels.py` |
| `handwash` module not found | You did not run `pip install -e .`, or the script is missing `from _bootstrap import ...` | Fix either one |
| Others cannot reproduce my results | No fixed seed / no recorded config_hash | Use `runtime.seed`, and fill in `config_hash` in the experiment record |
| Out of GPU memory | In clip mode the batch means something different (number of windows x window length) | Lower `train.batch_size` or `model.temporal.window` |
| Windows DataLoader error | Multiprocessing plus non-ASCII paths | `runtime.num_workers: 0` |

---

## 7. Your documentation obligations

**Changing something without documenting it = you did not change it.**

| What you changed | What you must update |
| --- | --- |
| Added or changed a config key | `docs/CONFIG.md` (key-by-key description) (in Chinese) |
| Added a dataset or a label structure | `docs/DATA.md` (in Chinese) |
| Changed a decision threshold or criterion | `docs/PROTOCOL.md` (in Chinese) |
| Ran an experiment (success or failure) | `docs/EXPERIMENTS.md` (in Chinese) (including `config_hash`) |
| A breaking change | `CHANGELOG.md` (marked `BREAKING`) + RFC |
| A new layering or architecture decision | RFC in section 5 of `docs/ARCHITECTURE.md` |

---

## 8. When a rule gets in your way

Rules serve people. If you think a rule is blocking the right thing:

1. **Do not quietly work around it** (for example hiding a magic number in a function default, or
   turning a test into a skip).
2. State in the PR: "I believe R× does not apply here, because ...".
3. The maintainer decides whether to grant a **one-off exemption** (registered file by file with a
   reason in `ALLOWED_EXCEPTIONS` in `scripts/check_structure.py`) or to **change the rule** (via an
   L1 RFC).

**Only two things are non-negotiable: data leakage (the split rules behind R6) and the uniqueness of
the label space (R10).** Break either one and the whole team's time is wasted.
