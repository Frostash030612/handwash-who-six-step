#!/usr/bin/env python
"""Train an offline temporal-convolution residual over the fixed boundary GRU.

Uses the original PSKUS DataSet1-4 train/validation split and frozen YOLO
features. The residual starts at exactly zero, so its initial predictions are
identical to the existing boundary GRU. The locked test split is not opened.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.boundary_gru import (
    BoundaryAwareGRU,
    boundary_targets_from_consensus,
    load_boundary_gru,
)
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES, file_sha256
from scripts.full_video_eval_common import public_metrics, videos_from_reference
from scripts.train_boundary_yolo_gru import (
    _crop_batch,
    _predict_items,
    _prepare_eval_templates,
    _score_predictions,
)


class TemporalConvRefiner(nn.Module):
    """Correct frozen GRU class logits using nearby hidden representations."""

    def __init__(self, base: BoundaryAwareGRU, *, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd and at least 3")
        self.base = base
        self.base.requires_grad_(False)
        hidden = int(base.hparams["hidden"]) * (2 if base.hparams["bidirectional"] else 1)
        self.correction = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, len(EXP_CLASS_NAMES), 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            normalized = (features - self.base.feature_mean) / self.base.feature_std
            hidden, state = self.base.rnn(self.base.encoder(normalized))
            previous = torch.cat((hidden[:, :1], hidden[:, :-1]), dim=1)
            changes = torch.abs(hidden - previous)
            edge_logits = self.base.boundary_classifier(torch.cat((hidden, changes), dim=-1)).squeeze(-1)
            base_logits = self.base.classifier(hidden)
        delta = self.correction(hidden.transpose(1, 2)).transpose(1, 2)
        return base_logits + delta, edge_logits, state


@torch.inference_mode()
def _predict_refiner(head: TemporalConvRefiner, templates: list[dict], device: str) -> list[dict]:
    head.eval()
    result = []
    for entry in templates:
        features = (
            torch.from_numpy(entry["cache_item"]["features"].astype(np.float32)).unsqueeze(0).to(device)
        )
        class_logits, edge_logits, _state = head(features)
        result.append(
            {
                **entry["template"],
                "frame_probs": class_logits[0].softmax(-1).cpu().numpy(),
                "edge_probs": edge_logits[0].sigmoid().cpu().numpy(),
                "positions": entry["positions"],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument(
        "--split-reference",
        type=Path,
        default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo/video_splits.csv",
    )
    parser.add_argument(
        "--yolo", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"
    )
    parser.add_argument(
        "--base-head", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_gru_20261012/head.pt"
    )
    parser.add_argument(
        "--feature-cache", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_temporal_conv_refiner"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience and batch size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")
    for required in (args.split_reference, args.yolo, args.base_head, args.feature_cache):
        if not required.exists():
            parser.error(f"Missing input: {required}")
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for training")

    videos = [
        video
        for video in videos_from_reference(args.root, args.split_reference)
        if video.split in ("train", "val")
    ]
    counts = {split: sum(video.split == split for video in videos) for split in ("train", "val")}
    if counts != {"train": 430, "val": 92}:
        parser.error(f"Unexpected locked split: {counts}")
    video_by_id = {f"{video.source}/{video.stem}": video for video in videos}
    base, payload = load_boundary_gru(args.base_head)
    if payload.get("exp_sha256") != file_sha256(args.yolo):
        parser.error("Boundary GRU and YOLO checkpoint mismatch")
    decoder = payload["decoder"]
    device = "cuda:0"
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    rows = []
    for video in videos:
        path = args.feature_cache / f"{video.source}__{video.stem}.npz"
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["model_sha256"]) != payload["exp_sha256"]:
                raise ValueError(f"Feature cache and YOLO checkpoint mismatch: {path}")
            row = {
                "video_id": f"{video.source}/{video.stem}",
                "split": video.split,
                "features": cached["features"].astype(np.float32),
                # This refiner targets the boundary region. Use every frame
                # where both annotators agree, including the agreed frames
                # omitted by the older stable-radius classification loss.
                "stable": cached["consensus"].astype(np.int64),
                "consensus": cached["consensus"].astype(np.int64),
                "frame_ids": cached["frame_ids"].astype(np.int64),
                "source_fps": float(cached["source_fps"]),
            }
        row["boundary_targets"], row["boundary_mask"] = boundary_targets_from_consensus(row["consensus"])
        rows.append(row)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    templates = _prepare_eval_templates(val, video_by_id)
    baseline_predictions = _predict_items(base.to(device), templates, device)
    baseline = _score_predictions(baseline_predictions, decoder)
    if abs(baseline["macro_f1"] - 0.953701893445979) > 1e-5:
        raise RuntimeError("Existing validation result was not reproduced")
    head = TemporalConvRefiner(base, kernel_size=args.kernel_size).to(device)
    initial = _score_predictions(_predict_refiner(head, templates, device), decoder)
    if abs(initial["macro_f1"] - baseline["macro_f1"]) > 1e-8:
        raise RuntimeError("Zero-initialized refiner did not reproduce the baseline")
    classes = Counter(int(label) for row in train for label in row["stable"] if label >= 0)
    class_weights = torch.tensor(
        [1 / max(classes.get(index, 0), 1) ** 0.5 for index in range(7)],
        dtype=torch.float32,
        device=device,
    )
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(head.correction.parameters(), lr=5e-4, weight_decay=1e-2)
    rng = random.Random(42)
    initial_state = {key: value.detach().cpu().clone() for key, value in head.correction.state_dict().items()}
    best_f1 = baseline["macro_f1"]
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        ordered = list(train)
        rng.shuffle(ordered)
        head.train()
        losses = []
        for offset in range(0, len(ordered), args.batch_size):
            features, labels, _edge_targets, _edge_mask, _weak = _crop_batch(
                ordered[offset : offset + args.batch_size], rng
            )
            features = features.to(device)
            labels = labels.to(device)
            valid = labels >= 0
            if not valid.any():
                continue
            logits, _edge_logits, _state = head(features)
            per_frame = nn.functional.cross_entropy(
                logits.reshape(-1, 7),
                labels.clamp_min(0).reshape(-1),
                weight=class_weights,
                reduction="none",
            ).reshape_as(labels)
            class_loss = (per_frame * valid).sum() / valid.sum()
            correction_penalty = head.correction[-1].weight.square().mean()
            loss = class_loss + 0.05 * correction_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.correction.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        selected = _score_predictions(_predict_refiner(head, templates, device), decoder)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_macro_f1": selected["macro_f1"],
                "val_transition_macro_f1": selected["transition_macro_f1"],
                "val_sequence_exact_videos": selected["sequence_exact_videos"],
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if selected["macro_f1"] > best_f1 + 1e-5:
            best_f1, best_epoch, stale = selected["macro_f1"], epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.correction.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        best_state = initial_state
        best_epoch = 0
    head.correction.load_state_dict(best_state)
    final = _score_predictions(_predict_refiner(head, templates, device), decoder)
    metadata = {
        "base_head_sha256": file_sha256(args.base_head),
        "yolo_sha256": file_sha256(args.yolo),
        "split_reference_sha256": file_sha256(args.split_reference),
        "train_videos": len(train),
        "val_videos": len(val),
        "kernel_size": args.kernel_size,
        "classification_labels": "dual_annotator_consensus_including_transition_neighbours",
        "best_epoch": best_epoch,
        "baseline": public_metrics(baseline),
        "candidate": public_metrics(final),
        "elapsed_s": time.perf_counter() - started,
        "test_used_for_selection": False,
    }
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "validation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if best_epoch:
        torch.save(
            {
                "format": "boundary-gru-temporal-conv-refiner/1",
                "class_names": list(EXP_CLASS_NAMES),
                "base_head_sha256": metadata["base_head_sha256"],
                "yolo_sha256": metadata["yolo_sha256"],
                "kernel_size": args.kernel_size,
                "state_dict": best_state,
            },
            args.output / "refiner.pt",
        )
    print(
        json.dumps(
            {"best_epoch": best_epoch, "baseline": metadata["baseline"], "candidate": metadata["candidate"]}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
