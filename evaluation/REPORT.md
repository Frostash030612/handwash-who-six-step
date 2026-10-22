# Final Four-Model Evaluation

## Scope and protocol

This is the saved retrospective comparison from 21 October 2026: ten previously reviewed
self-recorded videos and 632 exactly aligned valid centers. No new inference was run for
this English edition. Accuracy and macro F1 use the seven-class labels at common centers.
VideoMAE predictions use 16-frame windows; matching YOLO-family predictions use those same
center times. This aligns evaluation times without making the input contexts identical.

Output boundary F1 measures changes in output class labels for all four methods. Boundaries
are matched one-to-one with the annotated change interval plus 0.4 seconds tolerance.
It is not a score of the learned boundary branch alone. Reported boundary time errors are
protocol-defined distances from allowed annotation intervals, not exact continuous-frame
localization errors. Zero error can mean a match inside an allowed interval.

Raw chains collapse adjacent equal action labels. Final chains are the raw common-center
chains for YOLO, YOLO + GRU and VideoMAE. Only the boundary model uses its frozen adaptive
confirmation on dense 5 fps predictions. Thus the final-chain comparison includes the
complete algorithms, not just the backbone architectures.

Composite = 0.40 x macro F1 + 0.30 x final-chain exact-match rate + 0.30 x output boundary F1.
The weights are inherited from the saved comparison and were not changed to favor a model.

## Aggregate results

| Model | Accuracy | Macro F1 | Output boundary F1 | Raw chain match | Final chain match | Composite |
|---|---:|---:|---:|---:|---:|---:|
| YOLO | 95.25% | 93.84% | 76.92% | 2/10 | 2/10 | 66.61% |
| YOLO + GRU | 96.04% | 93.58% | 87.59% | 6/10 | 6/10 | 81.71% |
| YOLO + Adaptive Boundary-Aware GRU | 96.52% | 93.71% | 94.40% | 8/10 | 10/10 | 95.81% |
| VideoMAE | 99.53% | 99.02% | 97.48% | 10/10 | 10/10 | 98.85% |

## Boundary counts and matched-boundary error

| Model | TP | FP | FN | Mean absolute error (seconds) |
|---|---:|---:|---:|---:|
| YOLO | 60 | 36 | 0 | 0.000000 |
| YOLO + GRU | 60 | 17 | 0 | 0.013181 |
| YOLO + Adaptive Boundary-Aware GRU | 59 | 6 | 1 | 0.013404 |
| VideoMAE | 58 | 1 | 2 | 0.006825 |

## All per-video predictions

Every chain below is read from the saved result JSON. A correct chain does not imply every frame label is correct.

### 4

Reference: **1 -> 2 -> 3 -> 4 -> 5 -> 6**; 67 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 2 -> 6 -> 1 -> 3 -> 1 -> 3 -> 1 -> 3 -> 4 -> 5 -> 6 -> 4 -> 6 -> 4 | 1 -> 2 -> 6 -> 1 -> 3 -> 1 -> 3 -> 1 -> 3 -> 4 -> 5 -> 6 -> 4 -> 6 -> 4 | Incorrect |
| YOLO + GRU | 1 -> 2 -> 6 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 6 -> 3 -> 4 -> 5 -> 6 | Incorrect |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |
| VideoMAE | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |

### 8

Reference: **1 -> 2 -> 4 -> 5 -> 3 -> 6**; 59 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 3 -> 2 -> 4 -> 5 -> 3 -> 4 -> 6 -> 4 -> 6 | 1 -> 3 -> 2 -> 4 -> 5 -> 3 -> 4 -> 6 -> 4 -> 6 | Incorrect |
| YOLO + GRU | 1 -> 3 -> 2 -> 4 -> 5 -> 3 -> 4 -> 6 -> 4 -> 6 | 1 -> 3 -> 2 -> 4 -> 5 -> 3 -> 4 -> 6 -> 4 -> 6 | Incorrect |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 4 -> 5 -> 3 -> 4 -> 6 | 1 -> 2 -> 4 -> 5 -> 3 -> 6 | Correct |
| VideoMAE | 1 -> 2 -> 4 -> 5 -> 3 -> 6 | 1 -> 2 -> 4 -> 5 -> 3 -> 6 | Correct |

### test_correct_01

Reference: **1 -> 2 -> 3 -> 4 -> 5 -> 6**; 71 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 5 -> 6 | Incorrect |
| YOLO + GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |
| VideoMAE | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |

### test_correct_02

Reference: **1 -> 2 -> 3 -> 4 -> 5 -> 6**; 75 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 2 -> 3 -> 4 -> 5 -> 4 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 4 -> 6 | Incorrect |
| YOLO + GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |
| VideoMAE | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |

### test_correct_03

Reference: **1 -> 2 -> 3 -> 4 -> 5 -> 6**; 73 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 3 -> 2 -> 3 -> 4 -> 5 -> 6 -> 3 -> 6 | 1 -> 3 -> 2 -> 3 -> 4 -> 5 -> 6 -> 3 -> 6 | Incorrect |
| YOLO + GRU | 1 -> 3 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 3 -> 2 -> 3 -> 4 -> 5 -> 6 | Incorrect |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |
| VideoMAE | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |

### test_correct_04

Reference: **1 -> 2 -> 3 -> 4 -> 5 -> 6**; 68 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 2 -> 3 -> 4 -> 5 -> 3 -> 6 -> 2 | 1 -> 2 -> 3 -> 4 -> 5 -> 3 -> 6 -> 2 | Incorrect |
| YOLO + GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 2 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 2 | Incorrect |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 2 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |
| VideoMAE | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | 1 -> 2 -> 3 -> 4 -> 5 -> 6 | Correct |

### test_error_12356

Reference: **1 -> 2 -> 3 -> 5 -> 6**; 54 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 2 -> 3 -> 5 -> 6 -> 3 -> 6 | 1 -> 2 -> 3 -> 5 -> 6 -> 3 -> 6 | Incorrect |
| YOLO + GRU | 1 -> 2 -> 3 -> 5 -> 6 | 1 -> 2 -> 3 -> 5 -> 6 | Correct |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 3 -> 5 -> 6 | 1 -> 2 -> 3 -> 5 -> 6 | Correct |
| VideoMAE | 1 -> 2 -> 3 -> 5 -> 6 | 1 -> 2 -> 3 -> 5 -> 6 | Correct |

### test_error_12564

Reference: **1 -> 2 -> 5 -> 6 -> 4**; 52 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 1 -> 2 -> 5 -> 6 -> 4 | 1 -> 2 -> 5 -> 6 -> 4 | Correct |
| YOLO + GRU | 1 -> 2 -> 5 -> 6 -> 4 | 1 -> 2 -> 5 -> 6 -> 4 | Correct |
| YOLO + Adaptive Boundary-Aware GRU | 1 -> 2 -> 5 -> 6 -> 4 | 1 -> 2 -> 5 -> 6 -> 4 | Correct |
| VideoMAE | 1 -> 2 -> 5 -> 6 -> 4 | 1 -> 2 -> 5 -> 6 -> 4 | Correct |

### test_error_32654

Reference: **3 -> 2 -> 6 -> 5 -> 4**; 52 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 3 -> 1 -> 3 -> 1 -> 3 -> 2 -> 6 -> 5 -> 4 | 3 -> 1 -> 3 -> 1 -> 3 -> 2 -> 6 -> 5 -> 4 | Incorrect |
| YOLO + GRU | 3 -> 2 -> 6 -> 5 -> 4 | 3 -> 2 -> 6 -> 5 -> 4 | Correct |
| YOLO + Adaptive Boundary-Aware GRU | 3 -> 2 -> 6 -> 5 -> 4 | 3 -> 2 -> 6 -> 5 -> 4 | Correct |
| VideoMAE | 3 -> 2 -> 6 -> 5 -> 4 | 3 -> 2 -> 6 -> 5 -> 4 | Correct |

### test_error_645321

Reference: **6 -> 4 -> 5 -> 3 -> 2 -> 1**; 61 common valid centers.

| Model | Raw predicted chain | Final predicted chain | Final match |
|---|---|---|---|
| YOLO | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | Correct |
| YOLO + GRU | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | Correct |
| YOLO + Adaptive Boundary-Aware GRU | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | Correct |
| VideoMAE | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | 6 -> 4 -> 5 -> 3 -> 2 -> 1 | Correct |

## Interpretation and limits

The saved composite ranking is **VideoMAE, Adaptive Boundary-Aware GRU, GRU, YOLO**.
VideoMAE has the highest internal frame accuracy, macro F1 and output boundary F1.
The boundary-aware method has the strongest aggregate result among the YOLO family and
matches all ten final action chains. Its macro F1 is slightly below standalone YOLO,
so it should not be described as winning every metric even within the YOLO family.

The deployment choice favors the boundary-aware method for incremental frame-based feedback
and its smaller recorded architecture. VideoMAE can run locally and can stream delayed
updates. Its input-window delay prevents treating it as equivalent to immediate feedback.
The existing timing evidence is short synthetic timing with different inference units,
not a sustained throughput, power or isolated memory benchmark.

The ten videos were repeatedly inspected, labels and decoding were reviewed, and video 8
was corrected after seeing predictions. The results support an internal comparison, not
independent new-scene validation. Temporal heads and the other models also have different
final-fit quantities, as documented in the training provenance.

## Machine-readable records

- The local ZIP includes `result.json` with aggregate metrics and all per-video chains.
- [metrics.csv](metrics.csv): four-model summary.
- [action_chains.csv](action_chains.csv): 40 video/model rows.
- The local ZIP includes `frame_predictions.csv` with saved YOLO-family predictions.
- The local ZIP includes `videomae_predictions.csv` with saved VideoMAE predictions.

To recalculate metrics from these exports, follow the root README. Predictions
and numeric results were preserved; model display names were translated to English.
