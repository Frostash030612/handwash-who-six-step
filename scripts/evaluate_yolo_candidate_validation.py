#!/usr/bin/env python
"""Compare a YOLO candidate on complete PSKUS validation videos.

The locked test split is not opened. Both methods use the same 5 fps video
centres and dual-annotator consensus labels, including action transitions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.exp_demo import file_sha256
from handwash.pipelines.yolo_spatial_attention import install_spatial_attention
from scripts.compare_full_video_models import _centres, _item, _save_raw
from scripts.evaluate_yolo_full_videos import decode as decode_yolo, predict as predict_yolo
from scripts.full_video_eval_common import RAW_SETTING, public_metrics, score, videos_from_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"
    )
    parser.add_argument(
        "--adapter", type=Path, help="Optional attention adapter paired with the candidate YOLO checkpoint"
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument(
        "--split-reference",
        type=Path,
        default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo/video_splits.csv",
    )
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison/yolo/raw_probabilities",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_spatial_attention_validation",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=0, help="Small smoke run; omit for the complete validation set"
    )
    args = parser.parse_args()
    if args.batch < 1 or args.limit < 0:
        parser.error("batch must be positive; limit cannot be negative")
    for required in (
        args.base,
        args.split_reference,
        args.baseline_cache,
        *((args.adapter,) if args.adapter else ()),
    ):
        if not required.exists():
            parser.error(f"Missing required input: {required}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")

    import torch
    from ultralytics import YOLO

    videos = [
        video for video in videos_from_reference(args.root, args.split_reference) if video.split == "val"
    ]
    if len(videos) != 92:
        parser.error(f"Expected 92 locked validation videos, found {len(videos)}")
    if args.limit:
        videos = videos[: args.limit]
    model = YOLO(str(args.base))
    if args.adapter:
        install_spatial_attention(model, adapter_path=args.adapter)
    imgsz = int(model.overrides.get("imgsz", 320))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    candidates: list[dict] = []
    baseline: list[dict] = []
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for number, video in enumerate(videos, start=1):
        frames, frame_ids, fps = decode_yolo(video, 5.0)
        _starts, positions = _centres(frame_ids, num_frames=16, window_stride=4)
        if not len(positions):
            raise ValueError(f"Too short: {video.source}/{video.stem}")
        probabilities = predict_yolo(model, frames, imgsz, device, args.batch)
        item = _item(video, probabilities[positions], frame_ids, positions, fps)
        filename = item["video_id"].replace("/", "__") + ".npz"
        with np.load(args.baseline_cache / filename, allow_pickle=False) as old:
            for key in ("centre_frame_ids", "targets", "usable"):
                if not np.array_equal(item[key], old[key]):
                    raise ValueError(f"Baseline and candidate have different {key}: {item['video_id']}")
            baseline_item = {**item, "probs": old["probs"].astype(np.float32)}
        candidates.append(item)
        baseline.append(baseline_item)
        _save_raw(args.output / "raw_probabilities", item)
        print(
            f"Candidate {number}/{len(videos)} {item['video_id']} "
            f"centres={len(positions)} agreed={int(item['usable'].sum())}",
            flush=True,
        )

    baseline_score = score(baseline, RAW_SETTING)
    candidate_score = score(candidates, RAW_SETTING)
    arrays = []
    for old, new in zip(baseline, candidates, strict=True):
        keep = new["usable"]
        arrays.append((new["targets"][keep], new["probs"][keep].argmax(1), old["probs"][keep].argmax(1)))
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(1000):
        chosen = rng.integers(0, len(arrays), size=len(arrays))
        joined = [np.concatenate([arrays[int(index)][part] for index in chosen]) for part in range(3)]
        differences.append(
            float(
                f1_score(joined[0], joined[1], labels=list(range(7)), average="macro", zero_division=0)
                - f1_score(joined[0], joined[2], labels=list(range(7)), average="macro", zero_division=0)
            )
        )
    summary = {
        "split": "val",
        "videos": len(videos),
        "base_sha256": file_sha256(args.base),
        "adapter_sha256": file_sha256(args.adapter) if args.adapter else None,
        "split_reference_sha256": file_sha256(args.split_reference),
        "baseline_raw": public_metrics(baseline_score),
        "candidate_raw": public_metrics(candidate_score),
        "delta_macro_f1": candidate_score["macro_f1"] - baseline_score["macro_f1"],
        "paired_video_bootstrap_95pct_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        "elapsed_s": time.perf_counter() - started,
        "note": "Validation-only model selection. No test videos were read. Input frames and scoring centres match the archived YOLO baseline exactly.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "videos",
                    "baseline_raw",
                    "candidate_raw",
                    "delta_macro_f1",
                    "paired_video_bootstrap_95pct_interval",
                    "elapsed_s",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
