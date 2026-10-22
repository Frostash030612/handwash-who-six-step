# Model Card

## Intended use

A local coursework prototype recognizes six handwashing actions plus background from
self-recorded camera views and summarizes action order and recognized washing duration.
This is an action-observation system, not a measurement of microbial cleanliness.

## Model choices

| Choice | Input | Temporal method | Chain handling | Role |
|---|---|---|---|---|
| YOLO | 1088 x 1088 | Per-frame classification | Raw collapsed labels | Baseline |
| YOLO + GRU | 1088 x 1088 | GRU over frozen YOLO features | Raw collapsed labels | Temporal baseline |
| YOLO + Adaptive Boundary-Aware GRU | 1088 x 1088 | GRU with learned boundary branch | Boundary decoding and adaptive confirmation | Primary camera method |
| VideoMAE Base | 16 frames at 224 x 224 | Transformer video classification | Raw collapsed labels | Offline accuracy reference |

Camera frames in the YOLO path are first fit and replicate-padded to a 1080 square;
the checkpoint inference input is 1088. Temporal heads use 1,280-dimensional features,
hidden size 128 and dropout 0.3. The boundary GRU is unidirectional. Its saved decoder
uses threshold 0.3 and a minimum peak gap of four sampled frames.

Adaptive confirmation combines a one-second majority smoothing window with duration
outlier detection (z = -2, at least four comparable segments) and confidence evidence.
It belongs only to the boundary-aware method and does not impose the normal action order.
The implementation is in `source/src/handwash/core/action_chain.py`.

VideoMAE uses 16 frames at 5 fps and a four-sample window stride. Its generic pretraining
source is `MCG-NJU/videomae-base-finetuned-kinetics`. The final models use self-recorded
training data; PSKUS data and PSKUS-trained weights are not part of this release.

## Class mapping

- **0 — Other** (`other`).
- **1 — Palm to palm** (`step1`).
- **2 — Palm over dorsum** (`step2`).
- **3 — Fingers interlaced** (`step3`).
- **4 — Backs of fingers** (`step4`).
- **5 — Rotational thumbs** (`step5`).
- **6 — Fingertips to palm** (`step6`).

## Training and evaluation constraints

The YOLO backbone and VideoMAE use the 14-training / 8-validation development partition.
The retained temporal-head script selects on that partition and then refits the heads
and feature normalization on all 22 development clips. The selected epoch counts are
five for ordinary GRU and three for boundary GRU. This is not an equal-training-quantity
architecture-only ablation. The ten test clips are excluded from that final-fit list.

The final evaluation is retrospective and repeatedly reviewed. Video 8 was corrected
after predictions had been examined. Use the saved ten-clip report, not historical metrics
embedded in older checkpoints. Generalization to unseen users or viewpoints is unmeasured.

## Assets and integrity

`runtime/models/` contains `yolo_best.pt`, `yolo_gru.pt`, `boundary_gru.pt`, and
`videomae/{model.safetensors,config.json,preprocessor_config.json}`. The [model manifest](final_release/model_manifest.json)
records file sizes, SHA-256 hashes, class contracts and decoder parameters. Checkpoint hashes and class contracts are recorded in the model manifest.
