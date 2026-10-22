# Handwashing Action Recognition and Local Assessment

**Final technical report · English edition · 22 October 2026**

## 1. Objective and system boundary

The project observes self-recorded handwashing actions, recognizes six steps plus background,
and assesses whether a recorded sequence contains the required action order and sufficient
recognized washing time. The final product is a local camera dashboard with one model selected
per session. The application uses no audio. It saves sampled images and results locally.

The primary method is YOLO + Adaptive Boundary-Aware GRU. YOLO and YOLO + GRU provide structural
baselines; VideoMAE provides a larger video-transformer accuracy reference. The system does
not infer cleanliness directly from action labels.

## 2. Data and annotations

There are 32 self-recorded videos and 6,987 sampled frames. The development partition has
14 training and eight validation videos; ten videos are reserved as the reported test set.
The collection includes correctly ordered continuous actions, deliberately incorrect orders,
and isolated step-4/step-5 clips. Sampling targets 5 fps. The seven class IDs are background
0 and steps 1–6. Annotated transitions are ignored according to validity flags.

Original video 8 has the reviewed reference chain 124536, with step 5 starting at 29 seconds.
The [dataset guide](../dataset/README.md) documents the schema variants, all clip IDs and the
reviewed correction. Raw videos and labels are included so the delivered results are inspectable.

## 3. Recognition architectures

YOLO26m-cls produces frame probabilities and 1,280-dimensional features at an inference input
size of 1088. The ordinary GRU predicts classes from the feature sequence. The boundary-aware
GRU adds a learned boundary branch and boundary-based decoding, followed by adaptive action
confirmation. Both temporal heads have hidden size 128 and dropout 0.3. The deployed boundary
head is unidirectional. Its checkpoint retains decoder threshold 0.3 and peak gap four samples.

Adaptive chain confirmation replaces a fixed three-second admission rule. It combines a
one-second smoothing window with segment-duration statistics and confidence evidence. The
configured duration outlier threshold is z = -2 with at least four comparable segments.
Only this primary method uses the confirmation algorithm. It does not force correct order;
incorrect and repeated actions remain representable.

VideoMAE Base processes 16-frame clips resized to 224 square, sampled at 5 fps with stride four.
It uses generic Kinetics pretraining and the same self-recorded data source. Input resolution
and temporal context differ from the YOLO methods, so results compare complete systems under
their actual settings. See the [model card](MODEL_CARD.md) for class and weight contracts.

## 4. Training provenance

YOLO and VideoMAE use the 14/8 training-validation split. The retained temporal-head script
selects epochs and decoding on that split, then initializes and refits heads and feature
normalization on all 22 development videos. Selected epochs are five for GRU and three for
boundary GRU. The ten reported test clips are absent from its final-fit list.

This protocol difference is disclosed because it prevents a strict equal-data architecture-only
ablation claim. No PSKUS training data or PSKUS-trained model assets are included in the final
model set. This English release freezes existing artifacts and does not retrain them.

## 5. Evaluation protocol and results

The final saved comparison aligns 632 valid time centers across ten videos. Accuracy and
seven-class macro F1 use identical reference labels at those centers. Output boundary F1 uses
one-to-one matching with annotation transition intervals plus 0.4 seconds tolerance. Raw chains
use common centers; the boundary model alone applies its adaptive final-chain algorithm to the
dense 5 fps stream. This is a complete-method comparison.

| Model | Accuracy | Macro F1 | Output boundary F1 | Raw chain match | Final chain match | Composite |
|---|---:|---:|---:|---:|---:|---:|
| YOLO | 95.25% | 93.84% | 76.92% | 2/10 | 2/10 | 66.61% |
| YOLO + GRU | 96.04% | 93.58% | 87.59% | 6/10 | 6/10 | 81.71% |
| YOLO + Adaptive Boundary-Aware GRU | 96.52% | 93.71% | 94.40% | 8/10 | 10/10 | 95.81% |
| VideoMAE | 99.53% | 99.02% | 97.48% | 10/10 | 10/10 | 98.85% |


The primary method improves the aggregate result and final-chain match over the ordinary GRU.
It achieves 10/10 final chains. VideoMAE leads frame accuracy, macro F1 and output boundary F1,
and also achieves 10/10 chains. The primary model does not win every metric; its macro F1 is
slightly below standalone YOLO. The [evaluation report](../evaluation/REPORT.md) contains all
40 raw/final predicted chains, boundary counts and metric definitions.

## 6. Camera workflow and assessment

A user selects a camera and one model, starts preview, records up to 120 seconds, then finishes.
The backend performs incremental processing and caches predictions. The final report consumes
that cache; it does not rerun the full recording through YOLO. Pending received frames still
have to complete. Live predictions and washing-time estimates are provisional.

Passing requires chain 123456 and at least 40 seconds of recognized actions. Background and
unknown time are excluded. There is no required duration per step. Short washing segments may
contribute to total time while remaining unconfirmed as distinct chain steps. Missing evidence
can produce a review-required duration verdict. Confidence cannot be reported as accuracy
without manual labels for the new recording.

## 7. Deployment comparison

The current local camera entry uses MPS when available and otherwise CPU. VideoMAE can run on
the local computer and emit streaming predictions, but it needs an initial 16-frame window
(about three seconds) and centered predictions use about 1.4 seconds of future context plus
computation. This differs from immediate frame-based feedback. The primary method is retained
as the camera-oriented choice, while VideoMAE remains the strongest internal accuracy reference.

Historical short synthetic timing gives approximately 10.64 million parameters for the
YOLO/boundary architecture and 86.23 million for VideoMAE. Its latency samples compare different
inference units and do not establish sustained throughput, power or isolated memory use. The
[provenance guide](../provenance/README.md) explains these limits.

## 8. Limitations and interpretation

The ten videos have been repeatedly inspected and used to review labels and decoding; video 8
was corrected after predictions were viewed. Reported performance is retrospective internal
validation, not independent evidence of new-person or new-viewpoint generalization. Training
quantities differ for the final temporal heads. Camera placement can affect predictions, even
when confidence is high. These limits constrain the conclusion without changing the delivered
technical outcome.

## 9. Reproducibility and delivery

The package includes frozen weights, local source, the self-recorded dataset, annotation files,
saved prediction exports, metric calculation code, dependency pins and SHA-256 manifests. The
quick-start guide covers installation, one-model camera use, offline metric recalculation and
troubleshooting. Historical training scripts document origin but require path/environment
adaptation for a new cloud run. The English localization changes presentation and launch
packaging; it does not produce new model measurements.

A course-specific final report with team details and references, presentation slides, recorded
presentation and individual reports are separate submission artifacts. This document is the
technical report for the delivered system.
