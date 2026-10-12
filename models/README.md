# PSKUS DataSet1–4 comparison checkpoints

These are the checkpoints used in the locked full-video comparison documented in
[`docs/FAIR_VIDEO_COMPARISON.zh-CN.md`](../docs/FAIR_VIDEO_COMPARISON.zh-CN.md).
The source videos and annotations are not included in this repository.

| File | Role | SHA-256 |
| --- | --- | --- |
| `pskuss_ds1234/yolo/best.pt` | Seven-class frame classifier used by the YOLO baseline and as the YOLO+GRU feature extractor | `7ec0e5ec7a7c3475c0fb9b0a1c745c847057ad2be1ccbbd6fda7764ddef20613` |
| `pskuss_ds1234/yolo_gru/head.pt` | GRU temporal head paired with the YOLO checkpoint above | `eb0dfa16c286f13594458afd48d9df75fc86822398d99d1f5cde9784ee5e2189` |
| `pskuss_ds1234/videomae_center_stable/model.safetensors` | Improved VideoMAE checkpoint selected by full-video validation Macro-F1 | `068726bf4d69ae6b9903c4b64c3b769c757ae00424b6536c374197fcbc7365ce` |
| `pskuss_ds1234/boundary_gru/head.pt` | Boundary-aware YOLO+GRU candidate, trained from the matching YOLO+GRU head | `27b029d400d93c008b60c2ad2f1bd0e968184cf53a1f3c8b17ff4ef3975a431d` |

VideoMAE configuration and preprocessing metadata are stored beside its weights.
The model comparison uses the fixed DataSet1–4 video split; see the report for
the split fingerprint, annotation policy, evaluation metrics, and limitations.
`pskuss_ds1234/web_inference.json` records the validation-selected smoothing settings
used by the local comparison page (`python scripts/run_compare.py`).
The boundary-aware checkpoint's validation and locked-test summaries, decoder
settings, and training provenance are stored beside `head.pt`. Its test metrics
and limitations are described in the fair-comparison report; the source videos,
annotations, and cached YOLO features are not included.
