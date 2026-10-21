> **English** | [中文](FINAL_RELEASE.zh-CN.md)

# Final technical release 2026-10-22

The model implementation is frozen. This release packages existing checkpoints,
annotations and saved results; it does not retrain models or introduce a new experiment.
The camera application uses YOLO + Adaptive Boundary-Aware GRU as the primary method.
YOLO and YOLO + GRU are structural baselines; VideoMAE is the offline accuracy reference.

## Frozen system

- Seven classes: other and steps 1–6. See [label map](final_release/label_map.json).
- YOLO26m-cls backbone, 1088 inference input. Camera images are first fit and
  replicate-padded to 1080 square, following the existing deployment pipeline.
- Temporal heads: 1280-dimensional features, hidden size 128, dropout 0.3;
  boundary GRU is unidirectional and includes a learned boundary branch.
- Boundary decoder: threshold 0.3, minimum peak gap 4 samples. Only the boundary
  model uses adaptive chain confirmation: 1 s majority smoothing, duration outlier
  z = -2, minimum 4 comparable segments, with confidence evidence.
- VideoMAE Base: 224 square, 16 frames at 5 fps, stride 4. Generic Kinetics
  pretraining is used. No PSKUS training data or weights are used in this final set.
- One selected model per camera session. Finalization reuses collected predictions.
- Pass criterion: complete chain 1→2→3→4→5→6 and at least 40 s of recognized washing.
  Other/unknown time is excluded; there is no separate per-step minimum-duration rule.
- Local deployment uses Apple MPS when available, otherwise CPU. This entry point
  does not select CUDA. Python dependencies are required; weights are supplied separately.

## Training provenance and scope

The dataset has 32 self-recorded clips: a development partition of 14 training and
8 validation clips, plus 10 test clips (original videos 4 and 8 and eight additions).
VideoMAE and the frozen YOLO backbone use the 14/8 development partition.

**The retained temporal-head training script uses a different final-fit protocol:**
it selects an epoch count and boundary decoder using the 14/8 partition, then
initializes a new head and refits it on all 22 development clips. Feature normalization
is recomputed for that final-fit set. The current head checkpoints retain the
selected epochs (GRU 5, boundary GRU 3) and match the hashes used for the ten-video
YOLO-family evaluation. They do not embed a complete list of training clip IDs.
This release preserves that provenance; it does not describe all four models as
trained exclusively on the same 14 clips or as a controlled architecture-only ablation.
The retained training script is included in the local package as a historical source,
not as a launch step. Test clips are excluded by its final-fit list.

## Frozen results

The authoritative final table is the saved 2026-10-21 retrospective comparison:
10 clips, 632 exactly aligned valid centers. See [results](SELF_SCENE_FOUR_MODEL_COMPARISON.zh-CN.md),
[summary CSV](final_release/metrics.csv) and [per-video chains](final_release/action_chains.csv).
These are existing results, not measurements produced during packaging.

| Model | Accuracy | Macro F1 | Output boundary F1 | Final chain match |
| --- | ---: | ---: | ---: | ---: |
| YOLO | 95.25% | 93.84% | 76.92% | 2/10 |
| YOLO + GRU | 96.04% | 93.58% | 87.59% | 6/10 |
| YOLO + Adaptive Boundary-Aware GRU | 96.52% | 93.71% | 94.40% | 10/10 |
| VideoMAE | 99.53% | 99.02% | 97.48% | 10/10 |

Frame and boundary metrics use the common centers. Only the boundary model's final
chain uses its dense 5 fps adaptive confirmation; the other chains use raw common-center
labels. This is a complete-method comparison. The ten clips were repeatedly inspected,
and video 8's labels were corrected after viewing predictions. Do not claim independent
new-scene generalization. Its final order is 1→2→4→5→3→6, with step 5 starting at 29 s.

The metrics inside older checkpoints refer to an earlier two-video evaluation and
older annotations. Do not quote them as the final table. Weight files remain byte-identical.
VideoMAE ranks first for internal accuracy. Its initial 16-frame window requires about
3 s of input; centered labels need about 1.4 s of future context plus computation.
It can stream delayed updates. We do not claim it cannot run on this computer, or that
the primary model has the highest accuracy. Existing short synthetic timing is not a
sustained-load, power or isolated peak-memory benchmark.

## Deliverable layout

The local `handwash-final-20261022.zip` contains:

- `source/`: committed code and documents at the recorded revision; historical public-data
  weights and personal presentation files are excluded.
- `runtime/models/`: the six files for the four frozen model choices, with SHA-256 hashes.
- `dataset/base/`: 24 annotated frame folders; original IDs 1–8, supplementary IDs 9–24.
- `dataset/test8/`: eight additional annotated frame folders.
- `dataset/raw/`: 32 original videos named by package clip ID.
- `dataset/`: manifests, current label correction and full split/annotation inventory.
- `evaluation/`: saved probability/classification exports, final result JSON and reports.
- `provenance/`: retained training scripts and selected historical deployment evidence.
- `install.sh`, `install.ps1`, `start_camera.py`: installation and launch helpers.
- `SHA256SUMS.json`, `verify_files.py`: file-integrity inventory and verifier.

Recordings and weights stay in the local submission package; they are not published to
GitHub. Existing camera session recordings, account credentials and caches are excluded.
The archive is the technical handoff, not the complete coursework submission: final
report, final slides, presentation video and individual work remain separate tasks.

## Run the packaged application

The [environment snapshot](final_release/deployment_environment.json) records existing package versions;
installation pins the key runtime dependencies. Install Python 3.12 first. On macOS/Linux, in the extracted package:

```sh
PYTHON_BIN=python3.12 bash install.sh
.venv/bin/python start_camera.py
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
.\.venv\Scripts\python.exe start_camera.py
```

Open http://127.0.0.1:8876/ in a camera-capable browser and grant camera permission.
Choose a model, preview the camera, start, then finish. Close any previous server on
8876 or use `start_camera.py --port 8877`. Installation needs internet; inference uses
included local weights. First-time model loading can take longer than subsequent updates.
The package is assembled on macOS; Windows installation is documented but not newly validated.

To reproduce the saved metric calculation without training or rerunning inference:

```sh
.venv/bin/python source/scripts/compare_self_scene_common_centers.py \
  --frame-predictions evaluation/frame_predictions.csv \
  --videomae-predictions evaluation/videomae_predictions.csv \
  --output reproduced_metrics
```

`python verify_files.py` checks packaged file hashes only. It does not measure model quality.
No new model experiments, camera captures or training runs are required to use this release.
