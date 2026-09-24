# Project Proposal

**Date of proposal:** 30 September 2026

**Project Title:** WHO Six-Step Hand Hygiene: A Video-Based Pattern Recognition System for Action Recognition and Procedure Completeness Assessment

**Group ID (As Enrolled in Canvas Class Groups):** 43

**Group Members (name, Student ID):**

- [1] Shen Ziyi — A035****J (Data & Evaluation)
- [2] Wang Lepeng — A035****L (Model & Training)
- [3] Zhu Jianyu — A035****L (Temporal & Assessment)
- [4] Xu Wenzhe — A032****W (Product, Demo & Delivery)

**Sponsor/Client:** Not applicable — self-initiated project.

---

## Background / Aims / Objectives

**Background.** Hand hygiene is the single most effective measure for preventing healthcare-associated infection, yet compliance is persistently poor and notoriously hard to measure. The World Health Organization (WHO) defines a standard six-step hand-washing procedure with a recommended total duration of 40-60 seconds, but assessing whether a person actually performed all six steps, in the correct order, for a sufficient time cannot be done reliably by human observation at scale: it requires continuous attention, and observers are known to over-report compliance. Meanwhile, video capture is cheap and ubiquitous - a smartphone is sufficient. This project treats video as a visual sensor stream and builds a pattern recognition system that converts it into an objective compliance assessment.

**Aim.** To design, build and evaluate a runnable pattern recognition system that, given a video of a person washing their hands, (i) classifies the hand-washing action performed in each frame into one of the WHO six steps, and (ii) converts that frame-level sequence into a verifiable verdict on procedure completeness - detecting missed steps, out-of-order steps, and steps performed for insufficient time.

**Objectives (measurable).**

| # | Objective | Success criterion |
| --- | --- | --- |
| O1 | Frame-level six-class action recognition | Macro-F1 >= 0.80 on a held-out, video-disjoint test split |
| O2 | Temporal integration to suppress single-frame misjudgements | >= 25% reduction in label transitions, without a Macro-F1 drop |
| O3 | Cross-scenario generalisation | Quantify and report the Macro-F1 drop transferring from hospital to laboratory data |
| O4 | Completeness assessment | >= 80% agreement with human ground truth on complete/incomplete and in-order/out-of-order |
| O5 | Runnable system | One-command installation and demo, verified on a machine that has never seen the repository |

**Scope boundaries.** The system performs procedural compliance analysis only. It does not diagnose disease, does not evaluate microbiological effectiveness, and does not replace an infection-control professional. It evaluates technique, not clinical outcome.

---

## Project Descriptions

### 1. Problem statement

A camera observes a hand-washing sink. The task is to output, for every frame, which WHO step is being performed, and then to output a structured verdict on the whole procedure. Three difficulties make this a genuine pattern recognition problem rather than a simple classification exercise.

(1) Inter-class visual similarity. Steps 2 and 4 both involve the back of the hand; steps 5 and 6 are small rotational motions of thumb and fingertips. These differ by a few centimetres of hand configuration at typical camera distances.

(2) Intra-class variability. The same step looks different across people, camera angles, lighting and backgrounds. Performance degrades when the deployment environment differs from the training environment - the classical domain-shift problem.

(3) Temporal structure is part of the answer. A single misclassified frame is not an error in the procedure; only persistent, time-ordered behaviour is. This makes per-frame accuracy an insufficient metric and motivates the temporal layer.

### 2. The four required aspects (we implement all four, three are guaranteed)

The brief requires at least three of four aspects to be developed, integrated and demonstrated. This project addresses all four, stated explicitly so that the contribution is unambiguous.

| Aspect | How it is realised |
| --- | --- |
| A1 Supervised / unsupervised learning | Supervised multi-class classification with per-frame ground-truth labels. The learning protocol uses a strictly video-disjoint train/validation/test split (no frames from one source video may appear in two splits) - treated as a correctness requirement, not a detail. |
| A2 Machine learning / deep learning | Deep learning with transfer learning. Primary model: a YOLO classification network fine-tuned from pretrained weights. Baselines: MobileNetV2 and a second YOLO classification variant, so every claimed gain is measured against a like-for-like reference. |
| A3 Hybrid machine learning / ensemble approach | Hybrid two-stage architecture with ensemble fusion. A frame-level classifier produces per-frame class posteriors; a temporal model (GRU and TCN both implemented and compared) consumes the frame embeddings and re-predicts with context. The two are fused at the probability level (weighted combination) and at the decision level (majority vote, sliding-window smoothing). A mean-pooling head is included as the no-temporal-modelling control, isolating the ensemble's contribution. |
| A4 Intelligent sensing / sense making | Sense making over a video sensor stream. The system does not stop at labels: it segments the label stream into action segments, reconstructs the performed action sequence, and applies explicit domain rules derived from the WHO guideline to produce a human-readable compliance report (per-step duration, proportion of total washing time, missed steps, order inversions, insufficient-duration steps, overall score). |

### 3. System architecture

A five-layer design; each layer has one responsibility and dependencies point only downwards. The structure is fixed in the repository as a single source of truth (the configuration contract plus the directory convention): the contract layer, the data and IO layer and the orchestration layer are already implemented and version-controlled, and the model layer is implemented against the same contract in weeks 2-3.

L0 Infrastructure - path constants, logging, exception hierarchy. L1 Contract layer - the label space (single authoritative definition of the six WHO steps), data contracts, configuration schema, metrics, and the completeness decision rules. L2 Data and IO - video decoding, frame extraction, manifest with integrity invariants, video-disjoint splitting. L3 Models - frame classifiers (YOLO, MobileNetV2), temporal heads (GRU, TCN, mean-pool), checkpoint management. L4 Orchestration - the five pipelines (prepare, train, evaluate, infer, assess) and the command-line interface.

Two design decisions matter for the credibility of the evaluation. First, a single authoritative label space: original label names differ across datasets (Step 5, step_5, Step5_water), and they are mapped to canonical names in exactly one place, so training and evaluation labels cannot silently diverge - a failure mode that produces plausible but meaningless metrics. Second, video-disjoint splitting is enforced by code: splitting by frame would place near-identical neighbouring frames in both training and test sets and inflate accuracy, so three independent guards fail the run if a source video crosses splits.

### 4. Data

PSKUS (hospital environment, per-frame annotations) provides the primary training and in-domain test data; a 300-500 clip subset is used first to validate the pipeline before scaling up. METC (laboratory environment, 72 participants, frame-level labels) is used for cross-scenario testing only and never for training - that is what makes the transfer measurement meaningful. The Kaggle hand-wash dataset provides a small, fast corpus for rapid prototyping. Finally, 4-6 self-recorded clips per team member provide completeness validation: the set is scripted to include one correct procedure, one with a step omitted, one with two steps swapped, and one with a step performed too briefly, each with a human-annotated ground-truth sequence. The self-recorded set is deliberate - it is the only data that tests the system under the conditions it claims to work in.

### 5. Method

Frame extraction: video is sampled to a fixed frame rate and frames are stored with a manifest recording clip identity, frame index, timestamp and label.

Supervised frame classification: a pretrained classification network is fine-tuned on the frame set. Class imbalance is handled with balanced class weighting, and model selection uses Macro-F1 on a held-out validation split rather than accuracy, which hides degradation on small classes.

Temporal modelling: frame embeddings are fed to a sequence model that predicts per frame with temporal context. Both a recurrent head (GRU) and a dilated causal convolutional head (TCN) are implemented; a mean-pooling head serves as the control containing no temporal modelling.

Fusion: frame-level and temporal posteriors are combined by probability smoothing, weighted fusion and majority voting - the ensemble aspect of A3.

Sense making: the fused label stream is smoothed, segmented into contiguous action segments subject to a minimum-duration criterion, and evaluated against WHO-derived rules - coverage of the six steps, order inversions in the reconstructed sequence, per-step duration relative to a fair share of the total washing time, and total washing duration against the 40-60 second guideline. The output is a structured report plus a rendered, human-readable summary.

### 6. Evaluation and metrics

We report accuracy for comparability with published work but never in isolation. Macro-F1 is the primary model-selection metric because it is insensitive to class imbalance. Per-class precision and recall, together with the confusion matrix, localise which steps are confused - the expected confusions are steps 2 with 4 and steps 5 with 6. A label-transition count directly measures the reduction in single-frame misjudgements, which accuracy cannot express. In-domain and cross-scenario Macro-F1 quantify domain shift. Agreement with human ground truth on completeness and ordering tests the system-level claim. Inference time per clip demonstrates practical viability, and bootstrap confidence intervals ensure that a claimed improvement is not sampling noise.

For the ablation, data, split, seed and backbone are held fixed while only the temporal component changes: (A) frame-only, (B) frame plus probability smoothing, (C) mean-pool head, (D) GRU or TCN head. This isolates the contribution of A3 rather than asserting it.

### 7. Work plan (four weeks, four members)

Week 1: proposal, data pipeline, first baseline, environment verified on all four machines (proposal due 30 Sep). Week 2: first presentation on 6 Oct, primary model trained on hospital data, cross-scenario evaluation wired. Week 3: temporal ablation, completeness evaluation, error analysis, self-recorded validation set. Week 4: feature freeze, packaging of the runnable system, report, video presentation. A buffer period from 20 to 31 October is reserved for the clean-machine rehearsal, final polish, peer review, submission (due 31 Oct) and the written examination.

Workstreams, each with a single owner: data and evaluation; model and training; temporal and assessment; product, demo and delivery. The full plan, including weekly acceptance criteria and a requirement-to-artefact traceability matrix, is maintained in docs/PROJECT_PLAN_4_WEEK.md.

### 8. Tools and technologies

Python 3.12, PyTorch, Ultralytics (YOLO classification), torchvision (MobileNetV2), NumPy / pandas / scikit-learn (metrics), OpenCV and imageio (video decoding), Pillow (preprocessing), Matplotlib (figures), YAML-driven configuration with strict validation, pytest (contract-level unit tests), Git with pull-request review and continuous integration, and conda for reproducible environments.

### 9. Expected deliverables

- A runnable pattern recognition system with one-command setup and demo.
- The datasets used, with documented provenance, layout and preparation steps.
- A final report covering tools and techniques, system design and models, system performance, and findings and discussions - including honest failure analysis.
- Source code, configuration files and trained model weights.
- A 10-15 minute recorded video presentation.
- Slides for both the first and the final presentation.
- A 1-2 page individual report per member (personal contribution; what was learnt that is most useful; how it applies elsewhere).
- Peer review submitted through the peer review system.

### 10. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Large hospital dataset (about 18 GB) delays progress | Start the download in week 1 while prototyping on a small dataset; treat small-data results as a legitimate first milestone |
| Pretrained weights for the newest model may be unavailable | The model layer supports any classification weight; switching is a configuration change, and the actual weights used are reported |
| Data leakage inflating results | Video-disjoint splitting enforced by three independent guards; the split design is reported explicitly |
| Domain shift degrading real-world performance | Measured and reported as a finding rather than hidden; deployment conditions are characterised |
| Reproducibility across four machines | Fixed seeds, hashed configuration recorded per experiment, locked environment file, clean-machine rehearsal before submission |

---

## Pre-submission checklist

- [ ] Date of proposal filled
- [ ] Project title filled
- [ ] Canvas group ID pasted and all four members enrolled
- [ ] All four members' names and student IDs correct
- [ ] Sponsor/Client row set to Not applicable
- [ ] Background/Aims/Objectives pasted and the objective table renders cleanly
- [ ] Project Descriptions pasted with the 3-of-4 aspect coverage unambiguous
- [ ] Exported to PDF and confirmed no text was cut off by table cells
- [ ] Uploaded to Canvas - Assignments - Practice Module (one submission per team)
- [ ] Submission receipt screenshot stored under deliverables/
