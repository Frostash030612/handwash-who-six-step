> **English** | [中文](/docs/zh/ARCHITECTURE.md)

# Architecture (ARCHITECTURE)

This document explains **why the layers are drawn this way**, and **why the interfaces in
`core/` cannot be changed on a whim**. Read it before you read the code — it saves a great
deal of "why is this function here?" confusion.

---

## 1. Layer diagram

```
                    ┌──────────────────────────────────────────────┐
   L4  Entry &      │  cli.py  cli_doctor.py  scripts/*.py         │
       orchestration│  pipelines/: prepare train evaluate          │
                    │              infer  assess  common           │
                    └───────────────────┬──────────────────────────┘
                                        │ may import every layer below
                    ┌───────────────────▼──────────────────────────┐
   L3  Models       │  models/: base frame_cnn yolo26_cls          │
                    │           temporal voting factory checkpoint │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
   L2  Data &       │  io/: utils video manifest split             │
       persistence  │  data/: dataset transforms synthetic         │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
   L1  Contracts    │  core/: labels schema config metrics         │
       (most stable)│         protocol registry seeding            │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
   L0  Infrastructure  paths.py  logging.py  errors.py             |
                    │  (standard library only)                     │
                    └──────────────────────────────────────────────┘

   Dependency direction: each layer may point only downward.
   A reverse dependency is a structural violation.
```

### What each layer does, and what it must never do

| Layer | Responsibility | **Absolutely forbidden** |
| --- | --- | --- |
| L0 Infrastructure | Path constants, logging, exception hierarchy | Importing any third-party library (standard library only) |
| L1 Contracts | Data contracts, labels, config, metrics, business rules | Importing torch/ultralytics/cv2/pandas/matplotlib; reading or writing files (`config.py` reading YAML is the sole exception) |
| L2 IO / Data | Encode/decode, manifest, splits, preprocessing, Dataset | Importing models / pipelines |
| L3 Models | Network structure and weight I/O | Importing pipelines / cli |
| L4 Orchestration | Wiring the layers into workflows, command line | Implementing business rules (rules live only in `core/protocol.py`) |

**Why hold this line?** Because the thing that breaks a group project fastest is rules
scattered everywhere: if the missed-step decision exists simultaneously in the evaluation
script, the demo UI, and some notebook, the three will eventually disagree — and nobody
can say which one is right. Layering pins the rules into a single file that everyone else
may only call.

---

## 2. The five critical contracts (change one, and an RFC is required)

### 2.1 Label space — `core/labels.py`

```python
class Step(str, Enum):            # canonical names, unique across the project
    STEP_1 = "step_1_palm_to_palm"
    ...
STEP_ORDER = (STEP_1, ..., STEP_6)   # the **single authoritative order** of the six WHO steps
LabelSpace.get/to_index/to_label/to_space
```

* A raw label from any dataset must first be converted into a `Step` through
  `LabelSpace.canonicalize()`, then into a model channel index through `to_index()`.
  **No step literal may appear anywhere in the code.**
* In `LABEL_SPACES`, the **index order of every namespace is the model output channel
  order**. Append only — never insert or reorder — otherwise the output of every existing
  checkpoint silently shifts out of alignment.

### 2.2 Model output shape — `(B, T, C)`

Every model (single-frame, temporal, YOLO adapter) must implement:

```python
model.logits(x) -> Tensor (B, T, C)   # per-timestep classification logits
model.embed(x)  -> Tensor (B, T, D)   # features feeding the temporal head
model.num_classes: int
model.feature_dim: int
```

A single-frame model must also return `(B, 1, C)` and **must accept `(B, T, 3, H, W)`**
(folding T into the batch). That way training / evaluation / inference share exactly one
set of shape logic.

### 2.3 DataLoader output — also normalized to `(B, T, …)`

`collate_samples` (frame mode) and `clip_collate` (temporal mode) both produce:

```python
{
  "image":       (B, T, 3, H, W) float32
  "label_index": (B, T)  int64      # channel index into the label space
  "frame_index": (B, T)  int64
  "clip_id":     list[str] length B
}
```

**Do not** let any code path produce a 1-D `label_index`: that forces downstream code to
branch, and branches are a breeding ground for bugs.

### 2.4 Request / response objects — `core/schema.py`

`FrameRecord` / `ClipRecord` (data), `Clip` / `Sample` (input),
`FramePrediction` / `ClipPrediction` (predictions), `StepStatistic` /
`ProtocolViolation` / `ProtocolReport` (conclusions), `EvalResult` (metrics).

* Time is always in **seconds (float)**; frame numbers always **start at 0**;
* Every new field must carry a default value (so old call sites keep working) — otherwise
  it is BREAKING;
* Serialization goes through `to_dict()` / `from_dict()` only.

### 2.5 Config structure — `core/config.py`

YAML key names map **character for character** onto dataclass fields; an unknown key is a
hard error.
Config sections: `project / runtime / paths / dataset / datasets / split / model /
train / eval / assess / infer`.

---

## 3. The data flow of one training run (contracts, strung together)

```
original_video (PSKUS/METC/self-recorded)
   │  ① io.video.iter_frames(sample_fps=5)  —— frame extraction (**after** the split)
   ▼
frame images + manifest.csv              ← io.manifest (with 6 invariant checks)
   │  ② io.split.split_clips(group_key=original_video)
   ▼
train / val / test splits                ← split by **original video**, preventing leakage
   │  ③ data.dataset.FrameManifestDataset + data.transforms.build_transforms
   ▼
batch: image (B,T,3,H,W), label_index (B,T)   ← pipelines.common.collate_*
   │  ④ models.factory.build_model(rc).logits(x) -> (B, T, C)
   ▼
logits -> CrossEntropy / FocalLoss -> backprop -> best.pt   ← pipelines.train
   │  ⑤ pipelines.evaluate: Accuracy / Macro-F1 / confusion matrix   ← core.metrics
   ▼
EvalResult (JSON + PNG + CSV on disk)
   │  ⑥ pipelines.infer: video -> per-frame probabilities (T, C)
   ▼
core.protocol.build_report: smoothing -> segmentation -> coverage/order/duration decision
   ▼
ProtocolReport (missed step, out-of-order, insufficient duration, overall score) -> JSON + Markdown
```

Step ⑥ is the **only** place allowed to make a business decision. The evaluation script
looks at metrics only, never at missed steps; the demo UI only calls `build_report` and
never decides order on its own.

---

## 4. Why a registry (`core/registry.py`)

The contract layer cannot import torch, but the pipeline layer needs to fetch an
implementation from a config string. The registry is that decoupling point:

```python
# models/my_model.py (L3, may import torch)
@register_model("my-model")
class MyModel(BaseClassifier): ...

# the import inside register_all() in models/factory.py triggers registration
# pipelines/train.py (L4) talks only to the registry
```

The payoff: a new model = one new file + one import line + one config file, with
**no existing code touched**. That directly reduces the probability of L1/L2 conflicts.

---

## 5. RFC process (the only legal entry point for an L1 change)

Append a record below (newest entry on top), then link it from the PR.

```markdown
### RFC-0003: Adjust the min_segment_s default from 1.0 to 0.5
- Date: 2026-04-11    Author: @some-team-member    Status: accepted
- Current state: step 6 in self-recorded videos is often judged "too short",
  which makes insufficient-duration warnings too frequent.
- Change: AssessConfig.min_segment_s default in core/config.py 1.0 -> 0.5;
          configs/config.yaml kept in sync; docs/PROTOCOL.md criteria note updated.
- Impact: the insufficient_duration count changes in every assess report already
          produced; **the completeness decisions of experiments E1—E3 must be re-run**
          (classification metrics are unaffected).
- Migration: no code changes needed; historical reports stay in their original
          directories, not deleted; note the threshold version when comparing.
- Approvals: @leader-a ✅  @leader-b ✅
```

### Accepted RFCs

#### RFC-0001: Unify model output as `(B, T, C)`
- Date: project initialization    Status: accepted (initial design of this framework)
- Reason: avoids writing a separate training and evaluation branch for "frame model" and
  "temporal model".
- Impact: binding on every model; enforced by `BaseClassifier.check_logits_shape`.

#### RFC-0002: Split first, extract frames second
- Date: project initialization    Status: accepted
- Reason: extracting frames before a random split puts adjacent frames of the same video
  into different sets, inflating the metrics.
- Impact: the pipeline order in `pipelines/prepare.py` is fixed as scan -> split -> frames;
  `io.split` provides three leakage checks (manifest validation / split validation /
  config switch).

---

## 6. Extension guide

| What you want to do | Where to change it | Level |
| --- | --- | --- |
| Swap the model architecture | New `models/xxx.py` + `configs/models/xxx.yaml` | L3 |
| Swap the backbone / temporal head combination | Config only (`model.arch` / `model.temporal.kind`) | L3 |
| Add a dataset | `core/labels.py` alias (append) + `configs/data/xxx.yaml` + `docs/DATA.md` | L1 (alias table) + L3 |
| Change a decision threshold | Config only (`assess` section) | L3 |
| Change a decision **rule** (e.g. allow out-of-order) | `core/protocol.py` | **L1 + RFC** |
| Add a metric | Append a function in `core/metrics.py` + wire into `evaluate.py` | L1 (append-only, non-breaking) |
| Add a CLI subcommand | `cli.py` + `Makefile` (both must change together) | L2 |
| Add a unit test | `tests/unit/<layer>/test_<module>.py` | L3 |

---

## 7. How the machine enforces these rules

| Check | Location | What it blocks |
| --- | --- | --- |
| Formatting and line endings | `.editorconfig` + pre-commit hooks | R2 |
| Layered dependencies / hardcoded paths / print | `scripts/check_structure.py` | R5, R8, R16 |
| Config validity | `scripts/check_config.py` | R3, R14 |
| Contract behavior | `tests/unit/**` | L1 semantics being changed quietly |
| Large files / secrets | pre-commit hooks | R6 |
| PR review | `CODEOWNERS` | L1 / L2 process |
| Full gate | `.github/workflows/ci.yml` | All of the above |

**Change L1 without changing the tests and CI goes red; change the tests to accommodate
the code and review blocks it.** That is the actual mechanism that keeps the framework
from being eroded.
