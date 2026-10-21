> **English** | [中文](/README.zh-CN.md)

# Six-Step Handwashing Recognition and Assessment

**Final technical version: 2026-10-22.** A local camera application recognizes seven
classes (other plus six handwashing actions), builds an action chain and checks total
washing time. Model development is frozen; this repository records the final implementation.

## Final models and data

- **Primary camera method:** YOLO + Adaptive Boundary-Aware GRU.
- **Structural baselines:** YOLO and YOLO + GRU.
- **Offline accuracy reference:** VideoMAE Base with generic Kinetics pretraining.
- **Dataset:** 32 self-recorded clips; 14/8 development split and 10 test clips.
  The retained GRU-head script refits on all 22 development clips after selection.
  See the [training provenance](docs/FINAL_RELEASE.md#training-provenance-and-scope).
- **Assessment:** six steps in order and at least 40 seconds of recognized washing.
  There is no separate minimum duration for each step.

The final model set uses no PSKUS data or PSKUS-trained weights. Public-data experiments
elsewhere in the repository are historical work, not the current camera model bundle.

## Start the camera application

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/final_release/requirements-camera.txt -e ".[compare]"
python scripts/run_self_scene_camera.py --runtime-dir /path/to/runtime config=configs/experiments/self_scene_camera.yaml
```

Supply `runtime/models/` from the local final package; see the
[exact weight layout](docs/SELF_SCENE_CAMERA.zh-CN.md). A GitHub clone alone does not
include the self-recorded videos or these weights. The old root `exp.pt` is a historical
demo and is not a substitute for the final weights.

Open http://127.0.0.1:8876/ and grant camera permission. Select one model, preview the
camera, start recognition, then finish. The final report uses cached predictions.
Refreshing a completed session clears its previous on-screen result without deleting
saved local recordings. MPS is used when available, otherwise CPU; this entry point
does not select CUDA. One session is served at a time.

The local final archive includes installation and launch helpers, model files,
data and saved results. See [final release instructions](docs/FINAL_RELEASE.md).

## Recorded internal comparison

Ten previously reviewed test clips, 632 matched valid time centers:

| Model | Accuracy | Macro F1 | Output boundary F1 | Final action chain |
| --- | ---: | ---: | ---: | ---: |
| YOLO | 95.25% | 93.84% | 76.92% | 2/10 |
| YOLO + GRU | 96.04% | 93.58% | 87.59% | 6/10 |
| YOLO + Adaptive Boundary-Aware GRU | 96.52% | 93.71% | 94.40% | 10/10 |
| VideoMAE | 99.53% | 99.02% | 97.48% | 10/10 |

Only the boundary model uses adaptive final-chain confirmation. This is an internal
complete-method comparison, not proof of independent generalization or an architecture-only
ablation. Read the [evaluation protocol and limitations](docs/SELF_SCENE_FOUR_MODEL_COMPARISON.zh-CN.md).

VideoMAE has the highest internal accuracy. Its 16-frame window needs about 3 seconds
before the first prediction and about 1.4 seconds of future context for centered labels,
plus computation. It can stream delayed updates, but is not an immediate camera coach.
The primary method is the selected local camera solution, not the highest-accuracy method.

## Release files

- [Final technical release](docs/FINAL_RELEASE.md) / [中文](docs/FINAL_RELEASE.zh-CN.md)
- [Frozen weight and decoder manifest](docs/final_release/model_manifest.json)
- [Seven-class label map](docs/final_release/label_map.json)
- [Final metric table](docs/final_release/metrics.csv)
- [Per-video predicted chains](docs/final_release/action_chains.csv)
- [Camera deployment details](docs/SELF_SCENE_CAMERA.zh-CN.md)
- [Current results and historical experiment index](docs/EXPERIMENTS.md)
- [Coursework deliverables](deliverables/README.md)

Source code and reports are on GitHub. The local technical ZIP also contains weights,
raw videos, sampled frames, annotations and prediction exports. It excludes account
credentials, camera-session history and caches. It is not the final report/presentation ZIP.

## Historical work and contributor information

The following documents describe earlier public-data work. They are preserved for
provenance and are not additional tasks required to finish this release:

- [Public-data comparison](docs/FAIR_VIDEO_COMPARISON.zh-CN.md)
- [Previous implementation plan](docs/IMPLEMENTATION_PLAN.zh-CN.md)
- [Coursework plan](docs/PROJECT_PLAN_4_WEEK.md)
- [Architecture](docs/ARCHITECTURE.md), [configuration](docs/CONFIG.md),
  [contribution rules](CONTRIBUTING.md)

No medical diagnosis is made. Outputs are model-based assessments of the recorded actions.
