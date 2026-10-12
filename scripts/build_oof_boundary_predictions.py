#!/usr/bin/env python
"""Fit a fold-excluded temporal model and export its held-out TRAIN predictions.

The frozen YOLO feature extractor was trained on all 430 training videos, so
these are temporal-head out-of-fold predictions, not fully end-to-end OOF.
Each fold's GRU and boundary head are initialized and trained without that
fold; model selection uses a separate inner split of the remaining videos.
No validation or test video is used for training or threshold selection here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.boundary_gru import boundary_targets_from_consensus, decode_boundaries
from handwash.pipelines.exp_demo import file_sha256
from scripts.compare_full_video_models import _centres
from scripts.evaluate_dense_boundary_validation import segment_f1
from scripts.full_video_eval_common import RAW_SETTING, public_metrics, score
from scripts.train_boundary_yolo_gru import _train as train_boundary
from scripts.train_current_yolo_gru import _train as train_gru


def _fold(video_id: str) -> int:
    return int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % 5


def _inner(video_id: str) -> bool:
    return int(hashlib.sha256(("inner:" + video_id).encode()).hexdigest()[:8], 16) % 10 == 0


def _template(item: dict) -> tuple[np.ndarray, dict]:
    frame_ids = item["frame_ids"]
    _starts, positions = _centres(frame_ids.tolist(), num_frames=16, window_stride=4)
    consensus = item["consensus"][positions]
    return positions, {
        "video_id": item["video_id"], "centre_frame_ids": frame_ids[positions],
        "targets": consensus, "usable": consensus >= 0,
        "fps": float(item["source_fps"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-reference", type=Path,
                        default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo/video_splits.csv")
    parser.add_argument("--feature-cache", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--baseline-head", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_gru_20261012/head.pt")
    parser.add_argument("--gru-epochs", type=int, default=25)
    parser.add_argument("--boundary-epochs", type=int, default=20)
    args = parser.parse_args()
    fold_root = args.output / f"fold_{args.fold}"
    if fold_root.exists() and any(fold_root.iterdir()):
        parser.error(f"Refusing to overwrite fold: {fold_root}")
    payload = torch.load(args.baseline_head, map_location="cpu", weights_only=True)
    decoder = payload["decoder"]
    if decoder["method"] != "boundary_segment":
        parser.error("This OOF pipeline requires the fixed boundary-segment decoder")
    yolo_sha = file_sha256(PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt")
    if payload["exp_sha256"] != yolo_sha:
        parser.error("YOLO and archived boundary head mismatch")
    with args.split_reference.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    train_ids = [row["video_id"] for row in rows if row["split"] == "train"]
    if len(train_ids) != 430:
        parser.error("Expected 430 locked training videos")
    fit, inner, held = [], [], []
    for video_id in train_ids:
        filename = video_id.replace("/", "__") + ".npz"
        with np.load(args.feature_cache / filename, allow_pickle=False) as cached:
            if str(cached["model_sha256"]) != yolo_sha:
                raise ValueError(f"Feature cache uses another YOLO model: {filename}")
            item = {key: cached[key] for key in (
                "features", "frame_ids", "consensus", "stable", "source_fps",
            )}
        item["video_id"] = video_id
        edge_targets, edge_mask = boundary_targets_from_consensus(item["consensus"])
        item["boundary_targets"] = edge_targets
        item["boundary_mask"] = edge_mask
        item["weak_boundary_intervals"] = []
        if _fold(video_id) == args.fold:
            held.append(item)
        elif _inner(video_id):
            inner.append(item)
        else:
            fit.append(item)
    if min(len(fit), len(inner), len(held)) < 20:
        parser.error(f"Split too small: fit={len(fit)}, inner={len(inner)}, held={len(held)}")
    print(f"Fold {args.fold}: fit={len(fit)} inner={len(inner)} held={len(held)}", flush=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    baseline, gru_history, gru_best = train_gru(
        {"train": fit, "val": inner}, epochs=args.gru_epochs,
        batch_size=8, seed=42 + args.fold, device=device, patience=5,
    )
    inner_templates = []
    for item in inner:
        positions, template = _template(item)
        inner_templates.append({"cache_item": item, "positions": positions, "template": template})
    boundary, boundary_history, boundary_best, _edge_weight = train_boundary(
        data=fit, val_templates=inner_templates, epochs=args.boundary_epochs,
        batch_size=8, patience=5, seed=42 + args.fold, device=device,
        boundary_weight=0.2, boundary_positive_weight=None,
        bidirectional=False, gap_weak_loss_weight=0.0,
        repeat_minority_sources=False, temporal_consistency_weight=0.0,
        segment_consistency_weight=0.0, transition_supervision_weight=0.0,
        baseline_hparams=baseline.hparams,
        baseline_state={key: value.detach().cpu() for key, value in baseline.state_dict().items()},
    )
    boundary = boundary.to(device).eval()
    evaluated = []
    fold_root.mkdir(parents=True, exist_ok=False)
    prediction_root = args.output / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for number, item in enumerate(held, start=1):
            features = torch.from_numpy(item["features"].astype(np.float32)).unsqueeze(0).to(device)
            class_logits, edge_logits, _state = boundary(features)
            class_probs = class_logits[0].softmax(-1).cpu().numpy()
            edge_probs = edge_logits[0].sigmoid().cpu().numpy()
            positions, template = _template(item)
            decoded = decode_boundaries(class_probs, edge_probs,
                                        threshold=float(decoder["threshold"]),
                                        min_gap=int(decoder["min_gap"]))
            evaluated.append({**template, "probs": decoded[positions]})
            target = prediction_root / (item["video_id"].replace("/", "__") + ".npz")
            if target.exists():
                raise ValueError(f"OOF prediction already exists: {target}")
            np.savez_compressed(target, class_probs=class_probs, edge_probs=edge_probs,
                                frame_ids=item["frame_ids"], fold=args.fold,
                                video_id=item["video_id"],
                                yolo_sha256=yolo_sha, split_sha256=file_sha256(args.split_reference))
            if number % 20 == 0:
                print(f"OOF fold {args.fold}: {number}/{len(held)}", flush=True)
    metrics = score(evaluated, RAW_SETTING)
    f1, counts = segment_f1(evaluated, RAW_SETTING)
    metadata = {
        "fold": args.fold, "fit_videos": len(fit), "inner_videos": len(inner),
        "held_videos": len(held), "gru_best_epoch": gru_best,
        "boundary_best_epoch": boundary_best, "decoder": decoder,
        "temporal_head_fold_excluded": True,
        "yolo_backbone_fold_excluded": False,
        "split_sha256": file_sha256(args.split_reference),
        "oof_metrics": {**public_metrics(metrics), "who_segment_f1_at_50": f1,
                        "segment_counts": counts},
    }
    torch.save({"hparams": boundary.hparams,
                "state_dict": {key: value.detach().cpu() for key, value in boundary.state_dict().items()}},
               fold_root / "head.pt")
    (fold_root / "gru_history.json").write_text(json.dumps(gru_history, indent=2), encoding="utf-8")
    (fold_root / "boundary_history.json").write_text(json.dumps(boundary_history, indent=2), encoding="utf-8")
    (fold_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
