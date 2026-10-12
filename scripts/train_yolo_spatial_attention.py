#!/usr/bin/env python
"""Train a small attention-pooling adapter on the locked PSKUS DataSet1-4 split.

The YOLO26m backbone is frozen. Selection uses stable, dual-annotator
validation frames only; no held-out test videos or images enter this script.
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
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES, file_sha256
from handwash.pipelines.yolo_spatial_attention import (
    ADAPTER_FORMAT,
    install_spatial_attention,
)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    truth: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            output = model(images)
            logits = output[1] if isinstance(output, tuple) else output
        truth.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return {
        "images": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(
            f1_score(truth, predictions, labels=list(range(7)), average="macro", zero_division=0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"
    )
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_spatial_attention"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-balance-exponent",
        type=float,
        default=0.0,
        help="Training loss weight count**(-exponent); 0 keeps the original loss",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs, patience, batch must be positive; workers cannot be negative")
    if not 0.0 <= args.class_balance_exponent <= 0.5:
        parser.error("class-balance-exponent must be in [0, 0.5]")
    if not torch.cuda.is_available():
        parser.error("This experiment requires a CUDA GPU")
    for required in (args.base, args.data / "summary.json", args.data / "video_splits.csv"):
        if not required.is_file():
            parser.error(f"Missing required file: {required}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")

    summary = json.loads((args.data / "summary.json").read_text(encoding="utf-8"))
    if tuple(summary["classes"]) != EXP_CLASS_NAMES or summary["video_split"] != {
        "train": 430,
        "val": 92,
        "test": 93,
    }:
        parser.error("Dataset does not match the fixed seven-class DataSet1-4 protocol")

    from ultralytics import YOLO
    from ultralytics.data.augment import classify_transforms

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    yolo = YOLO(str(args.base))
    if yolo.model.yaml.get("scale") != "m":
        parser.error("This adapter experiment expects the existing YOLO26m checkpoint")
    image_size = int(yolo.overrides.get("imgsz", 320))
    transform = classify_transforms(size=image_size)
    folder = args.data / "ultralytics"
    train = ImageFolder(folder / "train", transform=transform)
    val = ImageFolder(folder / "val", transform=transform)
    if tuple(train.classes) != EXP_CLASS_NAMES or tuple(val.classes) != EXP_CLASS_NAMES:
        parser.error("ImageFolder class order differs from the locked seven classes")
    train_loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    class_counts = Counter(train.targets)
    class_weights = torch.tensor(
        [max(class_counts[index], 1) ** (-args.class_balance_exponent) for index in range(7)],
        dtype=torch.float32,
    )
    class_weights /= class_weights.mean()

    device = "cuda:0"
    class_weights = class_weights.to(device)
    model = yolo.model.to(device).eval()
    model.requires_grad_(False)
    baseline = evaluate(model, val_loader, device)
    print(json.dumps({"stage": "baseline_val_stable", **baseline}), flush=True)
    pool = install_spatial_attention(yolo).to(device)
    classifier = model.model[-1].linear
    classifier.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": pool.parameters(), "lr": 5e-4},
            {"params": classifier.parameters(), "lr": 5e-5},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.eval()  # Keep frozen BatchNorm statistics fixed.
        pool.train()
        classifier.train()
        losses: list[float] = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                output = model(images)
                logits = output[1] if isinstance(output, tuple) else output
                loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([*pool.parameters(), *classifier.parameters()], 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
        result = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": result["accuracy"],
            "val_macro_f1": result["macro_f1"],
            "gate": float(pool.gate_logit.sigmoid().detach().cpu()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if result["macro_f1"] > best_f1 + 1e-5:
            best_f1, best_epoch, stale = result["macro_f1"], epoch, 0
            payload = {
                "format": ADAPTER_FORMAT,
                "base_sha256": file_sha256(args.base),
                "class_names": list(EXP_CLASS_NAMES),
                "image_size": image_size,
                "best_epoch": epoch,
                "pool_state_dict": {
                    key: value.detach().cpu().clone() for key, value in pool.state_dict().items()
                },
                "linear_state_dict": {
                    key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()
                },
            }
            temporary = args.output / "best.tmp.pt"
            torch.save(payload, temporary)
            temporary.replace(args.output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    metadata = {
        "base": str(args.base),
        "base_sha256": file_sha256(args.base),
        "data": str(args.data),
        "split_reference_sha256": file_sha256(args.data / "video_splits.csv"),
        "train_images": len(train),
        "val_images": len(val),
        "train_class_counts": dict(class_counts),
        "class_balance_exponent": args.class_balance_exponent,
        "baseline_val_stable": baseline,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "elapsed_s": time.perf_counter() - started,
        "history": history,
        "test_used_for_selection": False,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "baseline": baseline,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
