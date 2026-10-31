# Six-Step Handwashing Recognition and Assessment

**Final project release · 31 October 2026**

This repository contains the camera application, the relevant training and
evaluation code, and the frozen ten-video comparison. The primary camera method is
**YOLO + Adaptive Boundary-Aware GRU**. YOLO and YOLO + GRU are structural baselines;
VideoMAE Base is the offline accuracy reference.

## Final internal comparison

| Model | Accuracy | Macro F1 | Output boundary F1 | Final chain match |
|---|---:|---:|---:|---:|
| YOLO | 95.25% | 93.84% | 76.92% | 2/10 |
| YOLO + GRU | 96.04% | 93.58% | 87.59% | 6/10 |
| YOLO + Adaptive Boundary-Aware GRU | 96.52% | 93.71% | 94.40% | 10/10 |
| VideoMAE | 99.53% | 99.02% | 97.48% | 10/10 |

Ten repeatedly reviewed self-recorded clips and 632 aligned valid centers were used.
VideoMAE leads internal accuracy. The boundary-aware method is selected for local camera
feedback and is the only model with adaptive action-chain confirmation. These are
retrospective internal results, not independent new-scene generalization. The temporal
heads were finally refit on 22 development clips, whereas YOLO and VideoMAE used the
14/8 selection partition; this is not a strict equal-data architecture ablation.

## Repository contents

- [`src/handwash/`](src/handwash/): camera backend, temporal models, adaptive chain
  confirmation, assessment and dashboard.
- [`scripts/run_self_scene_camera.py`](scripts/run_self_scene_camera.py): local camera launcher.
- [`scripts/compare_self_scene_common_centers.py`](scripts/compare_self_scene_common_centers.py):
  recalculate the saved four-model metrics.
- [`scripts/train_videomae_self_scene.py`](scripts/train_videomae_self_scene.py): retained
  self-recorded VideoMAE training source.
- [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md): final system methods, results and limits.
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md): model inputs, labels, training and deployment.
- [`dataset/README.md`](dataset/README.md): annotation and split description.
- [`evaluation/REPORT.md`](evaluation/REPORT.md): all ten videos and 40 predicted chains.
- [`provenance/README.md`](provenance/README.md): actual training and deployment evidence.

The current branch focuses on the camera system and the final self-recorded results.
Historical experiments remain in Git history and the `self-scene-final-20261022` tag.

## Final report and presentation

The [submission folder](submission/README.md) contains the 45-page final report, both
presentation slide decks, and the fully narrated 13-minute-53-second video with the
system demonstration. The editable [Overleaf source](docs/final_report/Camera_Based_Handwashing_Assessment_Overleaf_Source.zip)
remains available separately. The team's Canvas submission ZIP also contains the
annotated dataset and runnable system with model weights; individual reports and peer
reviews are submitted separately.

## Complete submission archive

The [submission assets](submission_assets/README.md) contain the complete group ZIP as
three Git LFS parts. They include the self-recorded dataset, sampled frames, reviewed
annotations, four model checkpoints, runnable application, report, slides and narrated
video. Run the restore script after cloning with Git LFS to recover the single ZIP for
Canvas submission.

## Local camera setup

The self-recorded raw videos, sampled images, frame annotations and model weights are
included in the Git LFS submission archive. Extract its `System/runtime/models/`
directory to a local path. Install Python 3.12 and dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r docs/final_release/requirements-camera.txt -e '.[compare]'
.venv/bin/python scripts/run_self_scene_camera.py --runtime-dir /absolute/path/to/runtime
```

Open `http://127.0.0.1:8876/`, grant camera permission, select one model and begin a
session. If the port is occupied, add `--port 8877`. The dashboard requires both the
ordered chain 1–6 and at least 40 seconds of recognized washing. Other and unknown
frames do not count. There is no fixed per-step duration threshold.

VideoMAE can run locally and stream delayed predictions. Its first 16-frame window
needs about three seconds of input; centered labels use about 1.4 seconds of future
context plus compute time. It is not equivalent to immediate frame-by-frame feedback.

To recalculate the saved results, supply the prediction CSVs from the local package:

```sh
.venv/bin/python scripts/compare_self_scene_common_centers.py \
  --frame-predictions /path/to/frame_predictions.csv \
  --videomae-predictions /path/to/videomae_predictions.csv \
  --output reproduced_metrics
```

The public Git LFS archive includes the self-recorded dataset and weights. Credentials
and camera session histories are excluded. Individual reports remain separate.
