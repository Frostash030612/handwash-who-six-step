#!/usr/bin/env python
"""Train a new GRU head for the current full-data YOLO checkpoint.

The old exp_temporal_head.pt is fingerprint-bound to exp.pt. This script keeps
that demo weight intact and fits a separate head using the same locked video
split and two-annotator stable labels as the current YOLO/VideoMAE comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if __package__ in (None, ""):
    from _bootstrap import PROJECT_ROOT
else:
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.exp_demo import (
    ExpFeatureExtractor,
    ExpTemporalHead,
    file_sha256,
    save_temporal_head,
)
from scripts.full_video_eval_common import videos_from_reference
from scripts.train_exp_temporal_head import _extract_video
from scripts.train_videomae_pskus import VideoInfo, _read_csv


def _labels(video: VideoInfo, frame_ids: np.ndarray, stable_radius: int) -> tuple[np.ndarray, np.ndarray]:
    left, left_times = _read_csv(video.primary)
    right, right_times = _read_csv(video.secondary)
    # Some originals have unequal annotator tail lengths. Keep only the
    # overlapping, time-aligned region; never invent a label for the tail.
    shared = min(len(left), len(right), len(left_times), len(right_times))
    agreed = np.asarray([
        a >= 0 and a == b and abs(left_times[index] - right_times[index]) <= 0.001
        for index, (a, b) in enumerate(zip(left[:shared], right[:shared], strict=True))
    ], dtype=bool)
    consensus = np.full(len(frame_ids), -1, dtype=np.int64)
    stable = np.full(len(frame_ids), -1, dtype=np.int64)
    for number, frame_id in enumerate(frame_ids):
        index = int(frame_id)
        if index >= len(agreed) or not agreed[index]:
            continue
        consensus[number] = left[index]
        lo, hi = max(0, index - stable_radius), min(len(agreed), index + stable_radius + 1)
        if all(agreed[lo:hi]) and all(left[other] == left[index] for other in range(lo, hi)):
            stable[number] = left[index]
    return consensus, stable


def _read_or_extract(video: VideoInfo, extractor: ExpFeatureExtractor | None,
                     checkpoint: Path, checksum: str, cache_root: Path, fps: float,
                     stable_radius: int, device: str) -> tuple[dict, ExpFeatureExtractor | None]:
    cache = cache_root / f"{video.source}__{video.stem}.npz"
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                if (str(saved["model_sha256"]) == checksum and float(saved["sample_fps"]) == fps
                        and int(saved["stable_radius"]) == stable_radius):
                    item = {key: saved[key] for key in (
                        "features", "frame_ids", "consensus", "stable", "source_fps"
                    )}
                    item["video_id"] = f"{video.source}/{video.stem}"
                    item["split"] = video.split
                    return item, extractor
        except (OSError, ValueError, KeyError):
            pass
    if extractor is None:
        extractor = ExpFeatureExtractor(checkpoint, device=device, cache_dir=cache_root / "runtime")
    features, _probs, ids = _extract_video(extractor, video.path, fps)
    frame_ids = np.asarray(ids, dtype=np.int64)
    consensus, stable = _labels(video, frame_ids, stable_radius)
    import cv2
    cap = cv2.VideoCapture(str(video.path))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    item = {
        "video_id": f"{video.source}/{video.stem}", "split": video.split,
        "features": features.astype(np.float16), "frame_ids": frame_ids,
        "consensus": consensus, "stable": stable, "source_fps": np.asarray(source_fps),
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{key: item[key] for key in (
        "features", "frame_ids", "consensus", "stable", "source_fps"
    )}, model_sha256=checksum, sample_fps=fps, stable_radius=stable_radius)
    temporary.replace(cache)
    return item, extractor


def _crop_batch(batch: list[dict], rng: random.Random, *, min_length: int = 64,
                max_length: int = 160) -> tuple[torch.Tensor, torch.Tensor]:
    crops = []
    for item in batch:
        length = len(item["features"])
        if length == 0:
            raise ValueError(f"Feature cache has no frames: {item['video_id']}")
        size = min(length, rng.randint(min_length, max_length))
        action_positions = np.flatnonzero(item["stable"] > 0)
        if len(action_positions) and rng.random() < 0.75:
            focus = int(rng.choice(action_positions.tolist()))
            start = min(max(0, focus - rng.randint(0, size - 1)), length - size)
        else:
            start = rng.randint(0, length - size)
        crops.append((item, start, size))
    longest = max(size for _item, _start, size in crops)
    features = torch.zeros(len(batch), longest, batch[0]["features"].shape[1])
    targets = torch.full((len(batch), longest), -1, dtype=torch.long)
    for row, (item, start, size) in enumerate(crops):
        features[row, :size] = torch.from_numpy(item["features"][start:start + size].astype(np.float32))
        targets[row, :size] = torch.from_numpy(item["stable"][start:start + size])
    return features, targets


@torch.inference_mode()
def _validation(head: ExpTemporalHead, videos: list[dict], device: str) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    head.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for item in videos:
        features = torch.from_numpy(item["features"].astype(np.float32)).unsqueeze(0).to(device)
        probs = head(features)[0].softmax(dim=-1)[0].cpu().numpy()
        starts = list(range(0, max(0, len(probs) - 16 + 1), 4))
        if len(probs) >= 16 and (not starts or starts[-1] != len(probs) - 16):
            starts.append(len(probs) - 16)
        centres = [start + 8 for start in starts]
        agreed = item["consensus"][centres]
        keep = agreed >= 0
        targets.extend(agreed[keep].tolist())
        predictions.extend(probs[centres][keep].argmax(axis=1).tolist())
    return (float(accuracy_score(targets, predictions)),
            float(f1_score(targets, predictions, labels=list(range(7)), average="macro", zero_division=0)))


def _train(data: dict[str, list[dict]], *, epochs: int, batch_size: int,
           seed: int, device: str, patience: int) -> tuple[ExpTemporalHead, list[dict], int]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    train = data["train"]
    feature_dim = train[0]["features"].shape[1]
    head = ExpTemporalHead(feature_dim=feature_dim)
    all_train = np.concatenate([item["features"].astype(np.float32) for item in train])
    head.feature_mean.copy_(torch.from_numpy(all_train.mean(axis=0)))
    head.feature_std.copy_(torch.from_numpy(all_train.std(axis=0) + 1e-3))
    del all_train
    counts = Counter(int(label) for item in train for label in item["stable"] if label >= 0)
    weights = torch.tensor([1 / max(counts.get(index, 0), 1) ** 0.5 for index in range(7)],
                           dtype=torch.float32)
    weights /= weights.mean()
    head = head.to(device)
    weights = weights.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    history: list[dict] = []
    best_state = None
    best_epoch = 0
    best_f1 = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        shuffled = train[:]
        rng.shuffle(shuffled)
        head.train()
        losses = []
        for offset in range(0, len(shuffled), batch_size):
            features, targets = _crop_batch(shuffled[offset:offset + batch_size], rng)
            features, targets = features.to(device), targets.to(device)
            logits = head(features)[0]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 7), targets.clamp_min(0).reshape(-1),
                weight=weights, reduction="none",
            ).reshape_as(targets)
            keep = targets >= 0
            if not keep.any():
                continue
            loss = (loss * keep).sum() / keep.sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_accuracy, val_f1 = _validation(head, data["val"], device)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                 "val_accuracy": val_accuracy, "val_macro_f1": val_f1}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("GRU did not produce a validation checkpoint")
    head.load_state_dict(best_state)
    return head.cpu().eval(), history, best_epoch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--split-reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--stable-radius", type=int, default=7)
    args = parser.parse_args()
    if (args.output / "head.pt").exists():
        parser.error(f"Refusing to overwrite existing GRU head: {args.output / 'head.pt'}")
    if not args.model.is_file():
        parser.error(f"YOLO checkpoint missing: {args.model}")
    videos = videos_from_reference(args.root, args.split_reference)
    model_checksum = file_sha256(args.model)
    data: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    extractor = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    for number, video in enumerate(videos, start=1):
        item, extractor = _read_or_extract(video, extractor, args.model, model_checksum, args.cache_root,
                                           args.sample_fps, args.stable_radius, device)
        data[video.split].append(item)
        print(f"Features {number}/{len(videos)} {video.split} {item['video_id']} "
              f"frames={len(item['frame_ids'])} stable={int((item['stable'] >= 0).sum())}", flush=True)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not data["train"] or not data["val"] or not data["test"]:
        parser.error("Original-video split is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    head, history, best_epoch = _train(data, epochs=args.epochs, batch_size=args.batch_size,
                                       seed=args.seed, device=device, patience=args.patience)
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {
        "best_epoch": best_epoch, "best_validation_macro_f1": history[best_epoch - 1]["val_macro_f1"],
        "split_reference": str(args.split_reference), "split_reference_sha256": file_sha256(args.split_reference),
        "source_videos": {split: len(group) for split, group in data.items()},
        "stable_training_labels": {str(key): value for key, value in sorted(Counter(
            int(label) for item in data["train"] for label in item["stable"] if label >= 0
        ).items())},
        "sample_fps": args.sample_fps, "stable_radius": args.stable_radius,
        "yolo_model_sha256": model_checksum,
        "feature_cache_root": str(args.cache_root), "elapsed_s": time.perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_temporal_head(args.output / "head.pt", head, exp_sha256=model_checksum,
                       fps=args.sample_fps, metrics=metadata)
    print(f"Saved GRU head: {args.output / 'head.pt'} (best epoch {best_epoch})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
