> **English** | [中文](/docs/zh/PROJECT_PROPOSAL.md)

# Project Proposal — Content Pack

Ready-to-paste content for [`Project_Proposal_Template.docx`](Project_Proposal_Template.docx).
**Deadline: 30 Sep 2026.** Open the DOCX template, fill each row of its table from the sections
below, export to PDF, and submit one copy per team to Canvas → Assignments → Practice Module.

> The template has exactly these rows. Do not add or reorder them:
> `Date of proposal` · `Project Title` · `Group ID (As Enrolled in Canvas Class Groups)` ·
> `Group Members (name, Student ID)` · `Sponsor/Client` · `Background/Aims/Objectives` ·
> `Project Descriptions`.
>
> **Fill in first:** `Date of proposal`, `Group ID`, group members, and the bracketed
> `[...]` placeholders. Everything else is written and ready.

---

## 1. Date of proposal

```
30 September 2026
```

*(Use the actual submission date if it differs.)*

## 2. Project Title

```
WHO Six-Step Hand Hygiene: A Video-Based Pattern Recognition System for Action
Recognition and Procedure Completeness Assessment
```

**Chinese (for internal use, not for the template):** 基于视频的 WHO 六步洗手动作识别与流程完整性评估系统

> Why this title: it names the **system** (which is what the "Final System" 15% is graded on),
> not just a classifier. Graders scanning titles for "pattern recognition system" will find it.

## 3. Group ID (As Enrolled in Canvas Class Groups)

```
[PASTE YOUR CANVAS GROUP ID HERE]
```

## 4. Group Members (name, Student ID)

```
[1] [Full Name] — [Student ID]     (Data & Evaluation Lead)
[2] [Full Name] — [Student ID]     (Model & Training Lead)
[3] [Full Name] — [Student ID]     (Temporal & Assessment Lead)
[4] [Full Name] — [Student ID]     (Product, Demo & Delivery Lead)
```

## 5. Sponsor/Client

```
Not applicable — self-initiated project.
```

*(If you have a clinical contact willing to be named as an informal sponsor, name them here with
their email; otherwise the line above is correct and acceptable.)*

## 6. Background / Aims / Objectives

Paste the following (≈1 page). It is written to answer exactly what this row asks:
background, aims, and measurable objectives.

---

**Background.** Hand hygiene is the single most effective measure for preventing
healthcare-associated infection, yet compliance is persistently poor and notoriously hard to
measure. The World Health Organization (WHO) defines a standard six-step hand-washing procedure
with a recommended total duration of 40–60 seconds, but assessing whether a person actually
performed all six steps, in the correct order, for a sufficient time cannot be done reliably by
human observation at scale: it requires continuous attention, and observers are known to
over-report compliance. Meanwhile, video is cheap and ubiquitous — a smartphone is sufficient.
This project treats video as a **visual sensor stream** and builds a pattern recognition system
that converts it into an objective compliance assessment.

**Aim.** To design, build and evaluate a runnable pattern recognition system that, given a video
of a person washing their hands, (i) classifies the hand-washing action performed in each frame
into one of the WHO six steps, and (ii) converts that frame-level sequence into a verifiable
verdict on **procedure completeness** — detecting missed steps, out-of-order steps, and steps
performed for insufficient time.

**Objectives (measurable).**

| # | Objective | Success criterion |
| --- | --- | --- |
| O1 | Frame-level six-class action recognition | Macro-F1 ≥ 0.80 on a held-out, video-disjoint test split |
| O2 | Temporal integration to suppress single-frame misjudgements | ≥ 25% reduction in label transitions, **without** a Macro-F1 drop |
| O3 | Cross-scenario generalisation | Quantify and report the Macro-F1 drop when transferring from hospital data to laboratory data |
| O4 | Completeness assessment | ≥ 80% agreement with human ground truth on "complete / incomplete" and "in order / out of order" for self-recorded clips |
| O5 | Runnable system | One-command installation and demo, verified on a machine that has never seen the repository |

**Scope boundaries (stated deliberately).** The system performs **procedural compliance
analysis only**. It does not diagnose disease, does not evaluate microbiological effectiveness,
and does not replace an infection-control professional. It evaluates *technique*, not *clinical
outcome*.

---

## 7. Project Descriptions

Paste the following (≈2 pages). It covers the four required aspects, the architecture, the data,
the method, the evaluation, the plan and the risks.

---

### 7.1 Problem statement

A camera observes a hand-washing sink. The task is to output, for every frame, which WHO step is
being performed, and then to output a structured verdict on the whole procedure. Three
difficulties make this a genuine pattern recognition problem rather than a simple classification
exercise:

1. **Inter-class visual similarity.** Steps 2 and 4 both involve the back of the hand; steps 5
   and 6 are small rotational motions of thumb and fingertips. These differ by a few centimetres
   of hand configuration at typical camera distances.
2. **Intra-class variability.** The same step looks different across people, camera angles,
   lighting and backgrounds. Performance degrades when the deployment environment differs from
   the training environment — the classical domain-shift problem.
3. **Temporal structure is part of the answer.** A single misclassified frame is not an error in
   the *procedure*; only persistent, time-ordered behaviour is. This makes per-frame accuracy an
   insufficient metric and motivates the temporal layer.

### 7.2 The four required aspects (we implement all four, three are guaranteed)

The brief requires at least three of four aspects to be developed, integrated and demonstrated.
This project addresses **all four**, which we state explicitly so the contribution is unambiguous:

| Aspect | How it is realised |
| --- | --- |
| **A1 — Supervised / unsupervised learning** | **Supervised** multi-class classification with per-frame ground-truth labels. The learning protocol includes a strictly **video-disjoint** train/validation/test split (no frames from one source video may appear in two splits) — a design decision we treat as a correctness requirement, not a detail. |
| **A2 — Machine learning / deep learning** | **Deep learning** with transfer learning. Primary model: a YOLO classification network fine-tuned from pretrained weights. Baselines: MobileNetV2 and a second YOLO classification variant, so that any claimed gain is measured against a like-for-like reference. |
| **A3 — Hybrid machine learning / ensemble approach** | **Hybrid two-stage architecture with ensemble fusion.** A *frame-level* classifier produces per-frame class posteriors; a *temporal* model (GRU and TCN, both implemented and compared) consumes the frame embeddings and re-predicts with context. The two are fused at the probability level (weighted combination) and at the decision level (majority vote / sliding-window smoothing). A mean-pooling head is included as the no-temporal-modelling control, so the ensemble's contribution is isolated. |
| **A4 — Intelligent sensing / sense making** | **Sense making over a video sensor stream.** The system does not stop at labels: it segments the label stream into action segments, reconstructs the performed action sequence, and applies explicit domain rules derived from the WHO guideline to produce a human-readable compliance report (per-step duration, proportion of total washing time, missed steps, order inversions, insufficient-duration steps, and an overall score). This is the layer that converts sensing into a decision a human can act on. |

### 7.3 System architecture

A five-layer design; each layer has one responsibility, and dependencies point only downwards.
This structure is already implemented and version-controlled, so the proposal describes something
that exists rather than something intended.

| Layer | Responsibility |
| --- | --- |
| L0 Infrastructure | Path constants, logging, exception hierarchy |
| L1 **Contract layer** | Label space (the single authoritative definition of the six WHO steps), data contracts, configuration schema, metrics, and the completeness decision rules |
| L2 Data & IO | Video decoding, frame extraction, manifest with integrity invariants, **video-disjoint splitting** |
| L3 Models | Frame classifiers (YOLO / MobileNetV2), temporal heads (GRU / TCN / mean-pool), checkpoint management |
| L4 Orchestration | The five pipelines (`prepare`, `train`, `evaluate`, `infer`, `assess`) and the command-line interface |

Two design decisions matter for the evaluation's credibility:

- **A single authoritative label space.** Original label names differ across datasets
  (`Step 5`, `step_5`, `Step5_water` …). They are mapped to canonical names in exactly one place,
  so training and evaluation labels cannot silently diverge — a failure mode that produces
  plausible but meaningless metrics.
- **Video-disjoint splitting is enforced by code.** Splitting by frame would place near-identical
  neighbouring frames in both training and test sets, inflating accuracy. Three independent
  guards fail the run if a source video crosses splits.

### 7.4 Data

| Dataset | Role | Notes |
| --- | --- | --- |
| **PSKUS** (hospital, per-frame annotations) | Primary training and in-domain test | Large; a 300–500-clip subset is used first to validate the pipeline |
| **METC** (laboratory, 72 participants, frame-level labels) | **Cross-scenario** test only | Never used for training — that is what makes the transfer measurement meaningful |
| **Kaggle hand-wash dataset** | Rapid prototyping and pipeline validation | Small; used to reach a first end-to-end result quickly |
| **Self-recorded clips** (4–6 per member) | Completeness validation | Scripted to include one correct procedure, one with a step omitted, one with two steps swapped, and one with a step performed too briefly; each clip has a human-annotated ground-truth sequence |

The self-recorded set is deliberate: it is the only data that tests the system on the conditions
it claims to work in, and it yields a directly verifiable head-to-head against human judgement.

### 7.5 Method

1. **Frame extraction.** Video is sampled to a fixed frame rate; frames are stored with a manifest
   recording clip identity, frame index, timestamp and label.
2. **Supervised frame classification.** A pretrained classification network is fine-tuned on the
   frame set. Class imbalance is handled by balanced class weighting; training uses a held-out
   validation split with early stopping on Macro-F1 (not accuracy, which hides small-class
   degradation).
3. **Temporal modelling.** Frame embeddings are fed to a sequence model that predicts per frame
   with temporal context. Both a recurrent head (GRU) and a dilated causal convolutional head
   (TCN) are implemented; a mean-pooling head serves as the control that contains no temporal
   modelling.
4. **Fusion.** Frame-level and temporal posteriors are combined (probability smoothing, weighted
   fusion, majority voting) — the "ensemble" aspect of A3.
5. **Sense making.** The fused label stream is smoothed, segmented into contiguous action
   segments with a minimum-duration criterion, and evaluated against WHO-derived rules:
   coverage of the six steps, order inversions in the reconstructed sequence, per-step duration
   relative to a fair share of the total washing time, and total washing duration against the
   40–60 s guideline. The output is a structured report plus a rendered, human-readable summary.

### 7.6 Evaluation and metrics

| Metric | Why it is included |
| --- | --- |
| Accuracy | Comparable with published work; **reported but never alone** |
| **Macro-F1** | Primary model-selection metric; insensitive to class imbalance |
| Per-class precision / recall | Localises *which* steps are confused — the interesting failure mode |
| Confusion matrix | Shows the specific confusions (expected: steps 2↔4, 5↔6) |
| Label-transition count | Directly measures "fewer single-frame misjudgements" (O2), which accuracy cannot express |
| In-domain vs cross-scenario Macro-F1 | Quantifies domain shift (O3) |
| Completeness / order agreement vs human ground truth | The system-level claim (O4) |
| Inference time per clip | Demonstrates practical viability |
| Bootstrap confidence intervals | Ensures a claimed improvement is not sampling noise |

**Ablation design.** Data, split, seed and backbone are held fixed while only the temporal
component changes: (A) frame-only, (B) frame + probability smoothing, (C) mean-pool head,
(D) GRU or TCN head. This isolates the contribution of A3 rather than asserting it.

### 7.7 Work plan (four weeks, four members)

| Week | Focus | Course milestone |
| --- | --- | --- |
| 1 | Proposal; data pipeline; first baseline; environment verified on all four machines | **Proposal due 30 Sep** |
| 2 | First presentation; primary model trained on hospital data; cross-scenario evaluation wired | **First presentation 6 Oct** |
| 3 | Temporal ablation; completeness evaluation; error analysis; self-recorded validation set | — |
| 4 | Feature freeze; runnable system packaging; report; video presentation; submission | — |
| Buffer (20–31 Oct) | Clean-machine rehearsal, final polish, peer review, **deliverables due 31 Oct**, examination | **Deliverables 31 Oct** |

Workstreams (one owner each): data & evaluation · model & training · temporal & assessment ·
product, demo & delivery. The full plan, including weekly acceptance criteria and a
requirement-to-artefact traceability matrix, is maintained in `docs/PROJECT_PLAN_4_WEEK.md`.

### 7.8 Tools and technologies

Python 3.12 · PyTorch · Ultralytics (YOLO classification) · torchvision (MobileNetV2) ·
NumPy / pandas / scikit-learn (metrics) · OpenCV / imageio (video decoding) · Pillow
(preprocessing) · Matplotlib (figures) · YAML-driven configuration with strict validation ·
pytest (contract-level unit tests) · Git with pull-request review and CI · conda for
reproducible environments.

### 7.9 Expected deliverables

1. A **runnable** pattern recognition system with one-command setup and demo.
2. The **datasets** used, with documented provenance, layout and preparation steps.
3. A **final report** covering tools/techniques, system design and models, system performance, and
   findings and discussions — including honest failure analysis.
4. Source code, configuration files and trained model weights.
5. A **10–15 minute** recorded video presentation.
6. Slides for both the first and the final presentation.

### 7.10 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Large hospital dataset (≈18 GB) delays progress | Start the download in week 1 while prototyping on a small dataset; treat small-data results as a legitimate first milestone |
| Pretrained weights for the newest model may be unavailable | The model layer supports any classification weight; switching is a configuration change, and the actual weights used are reported |
| Data leakage inflating results | Video-disjoint splitting enforced by three independent guards; the split design is reported explicitly |
| Domain shift degrading real-world performance | Measured and reported as a finding rather than hidden; deployment conditions are characterised |
| Reproducibility across four machines | Fixed seeds, hashed configuration recorded per experiment, locked environment file, clean-machine rehearsal before submission |

---

## 8. Pre-submission checklist (30 Sep)

- [ ] `Date of proposal` filled
- [ ] `Project Title` filled (use §2 verbatim)
- [ ] Canvas group ID pasted and all four members are enrolled
- [ ] All four members' names **and student IDs** present and correct
- [ ] `Sponsor/Client` row says `Not applicable — self-initiated project.`
- [ ] `Background/Aims/Objectives` pasted, and the objective table reads cleanly
- [ ] `Project Descriptions` pasted, with §7.2 making the 3-of-4 aspect coverage unambiguous
- [ ] Exported to PDF, opened the PDF, confirmed no text was cut off by the table cells
- [ ] **Uploaded to Canvas → Assignments → Practice Module** (one submission per team)
- [ ] Screenshot of the submission receipt stored in `deliverables/`
