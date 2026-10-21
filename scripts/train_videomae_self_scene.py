#!/usr/bin/env python3
"""Fine-tune public VideoMAE weights on the project's own annotated clips only."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

GENERIC_CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"
LABELS = ("other", "step1", "step2", "step3", "step4", "step5", "step6")
LABEL_ALIASES = {
    "other": 0, "others": 0, "other_or_nonstandard": 0,
    **{f"step{i}": i for i in range(1, 7)},
    **{f"step_{i}": i for i in range(1, 7)},
}


@dataclass(frozen=True)
class Clip:
    clip_id: str
    folder: Path
    rows: tuple[dict[str, str], ...]
    split: str


@dataclass(frozen=True)
class Window:
    clip: Clip
    center_row: int
    frame_rows: tuple[int, ...]
    label: int


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _read_clip(folder: Path, split: str) -> Clip:
    label_path = folder / "frame_labels.csv"
    if not label_path.is_file():
        raise FileNotFoundError(f"缺少逐帧标注：{label_path}")
    with label_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"标注文件没有帧：{label_path}")
    for row in rows:
        image = folder / row["image"]
        if not image.is_file():
            raise FileNotFoundError(f"缺少抽帧图片：{image}")
        if not _valid_center(row):
            continue
        label = (row.get("label") or row.get("training_label")
                 or row.get("evaluation_label") or "").strip().lower()
        code = (row.get("movement_code_train") or row.get("movement_code") or "").strip()
        if code and code.lstrip("-").isdigit():
            movement = int(code)
            if 0 <= movement <= 6:
                continue
        if label not in LABEL_ALIASES:
            raise ValueError(f"无法映射标签 {label!r}：{label_path}")
    return Clip(folder.name.removesuffix("_frames"), folder, rows, split)


def _label_id(row: dict[str, str]) -> int:
    code = (row.get("movement_code_train") or row.get("movement_code") or "").strip()
    if code and code.lstrip("-").isdigit():
        movement = int(code)
        if 0 <= movement <= 6:
            return movement
    label = (row.get("training_label") or row.get("label")
             or row.get("evaluation_label") or "").strip().lower()
    return LABEL_ALIASES[label]


def _valid_center(row: dict[str, str]) -> bool:
    return _bool(row.get("valid_for_training"))


def _load_self_clips(
    train_root: Path, test8_root: Path, manifest_path: Path,
) -> tuple[list[Clip], list[Clip], list[Clip]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "self_scene_only_20261019/v1":
        raise ValueError("训练清单不是已核验的自采数据清单")
    if manifest.get("external_dataset_used") is not False:
        raise ValueError("数据清单声明包含外部数据，拒绝继续")
    if manifest.get("fit_video_ids") != [1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23]:
        raise ValueError("训练视频划分与锁定的 14/8 划分不符")
    if manifest.get("validation_video_ids") != [10, 12, 14, 16, 18, 20, 22, 24]:
        raise ValueError("验证视频划分与锁定的 14/8 划分不符")
    fit = [_read_clip(train_root / f"{i}_frames", "train")
           for i in manifest["fit_video_ids"]]
    validation = [_read_clip(train_root / f"{i}_frames", "validation")
                  for i in manifest["validation_video_ids"]]
    # Videos 4 and 8 were held out from the earlier self-scene fit and are part
    # of the final ten-video test; they never contribute training or selection.
    prior_test = [_read_clip(train_root / f"{i}_frames", "test") for i in (4, 8)]
    test8 = [_read_clip(folder, "test") for folder in sorted(test8_root.glob("*_frames"))]
    if len(test8) != 8:
        raise ValueError(f"应有 8 段新增独立测试视频，实际找到 {len(test8)} 段")
    return fit, validation, prior_test + test8


def _windows(clips: list[Clip], num_frames: int, stride: int) -> list[Window]:
    windows: list[Window] = []
    for clip in clips:
        last_start = len(clip.rows) - num_frames
        for start in range(0, last_start + 1, stride):
            center = start + num_frames // 2
            row = clip.rows[center]
            if not _valid_center(row):
                continue
            windows.append(Window(clip, center, tuple(range(start, start + num_frames)), _label_id(row)))
    return windows


class WindowDataset(Dataset):
    def __init__(self, windows: list[Window], *, mean: tuple[float, ...], std: tuple[float, ...],
                 augment: bool = False):
        self.windows, self.mean, self.std, self.augment = windows, mean, std, augment

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        window = self.windows[index]
        images = []
        for row_index in window.frame_rows:
            path = window.clip.folder / window.clip.rows[row_index]["image"]
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if self.augment and random.random() < 0.5:
            images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
        transform = Compose([Resize((224, 224)), ToTensor(), Normalize(self.mean, self.std)])
        pixel_values = torch.stack([transform(image) for image in images])
        row = window.clip.rows[window.center_row]
        time_s = float(row.get("actual_time_s") or row.get("target_time_s")
                       or float(row.get("frame_time_ms", 0.0)) / 1000.0)
        return pixel_values, window.label, window.clip.clip_id, time_s


def _metrics(model, loader, device: str) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    truth: list[int] = []
    prediction: list[int] = []
    with torch.inference_mode():
        for pixels, labels, *_metadata in loader:
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=pixels).logits
            truth.extend(labels.tolist())
            prediction.extend(logits.argmax(dim=-1).cpu().tolist())
    accuracy = float(np.mean(np.asarray(truth) == np.asarray(prediction))) if truth else 0.0
    f1s = []
    for class_id in range(7):
        tp = sum(t == class_id and p == class_id for t, p in zip(truth, prediction, strict=False))
        fp = sum(t != class_id and p == class_id for t, p in zip(truth, prediction, strict=False))
        fn = sum(t == class_id and p != class_id for t, p in zip(truth, prediction, strict=False))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return accuracy, float(np.mean(f1s)), truth, prediction


def _prediction_rows(model, loader, device: str) -> list[dict[str, object]]:
    model.eval()
    output = []
    with torch.inference_mode():
        for pixels, labels, clip_ids, times in loader:
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                probabilities = model(pixel_values=pixels).logits.softmax(dim=-1).float().cpu().numpy()
            labels = labels.tolist()
            for clip_id, time_s, truth, probs in zip(clip_ids, times.tolist(), labels, probabilities, strict=False):
                prediction = int(np.argmax(probs))
                output.append({"clip_id": clip_id, "time_s": float(time_s),
                               "true_label": int(truth), "predicted_label": prediction,
                               "confidence": float(probs[prediction])})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True,
                        help="目录中应有 1_frames 至 24_frames 自采标注目录")
    parser.add_argument("--test8-root", type=Path, required=True,
                        help="新增 8 段独立测试视频的标注帧目录")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="自采数据清单，锁定 14 段训练及 8 段验证")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=GENERIC_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true", help="只校验划分、标签和样本数")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.batch_size < 1 or args.grad_accum < 1 or args.num_frames < 2 or args.window_stride < 1:
        parser.error("batch-size、grad-accum、window-stride 必须为正，num-frames 至少为 2")

    fit, validation, test = _load_self_clips(args.train_root, args.test8_root, args.manifest)
    groups = {name: _windows(clips, args.num_frames, args.window_stride)
              for name, clips in (("train", fit), ("validation", validation), ("test", test))}
    if not groups["train"] or not groups["validation"] or not groups["test"]:
        parser.error("某个划分没有可用的有效中心窗口")
    print(json.dumps({
        "data_source": "self_recorded_only", "pskus_data_used": False,
        "pretrained_checkpoint": args.checkpoint, "pskus_weights_used": False,
        "video_splits": {"train": [clip.clip_id for clip in fit],
                         "validation": [clip.clip_id for clip in validation],
                         "test": [clip.clip_id for clip in test]},
        "window_counts": {key: len(value) for key, value in groups.items()},
        "label_counts": {key: dict(Counter(window.label for window in value))
                         for key, value in groups.items()},
        "input_size": [224, 224], "num_frames": args.num_frames,
    }, ensure_ascii=False, indent=2), flush=True)
    if args.inspect:
        return 0
    if not torch.cuda.is_available():
        parser.error("VideoMAE 训练需要 CUDA")
    if not torch.cuda.is_bf16_supported():
        parser.error("该训练入口要求 GPU 支持 BF16")

    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained(args.checkpoint)
    mean, std = tuple(processor.image_mean), tuple(processor.image_std)
    loaders = {
        "train": DataLoader(WindowDataset(groups["train"], mean=mean, std=std, augment=True),
                            batch_size=args.batch_size, shuffle=True, num_workers=4,
                            pin_memory=True, persistent_workers=True),
        "validation": DataLoader(WindowDataset(groups["validation"], mean=mean, std=std),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True, persistent_workers=True),
        "test": DataLoader(WindowDataset(groups["test"], mean=mean, std=std),
                           batch_size=args.batch_size, shuffle=False, num_workers=4,
                           pin_memory=True, persistent_workers=True),
    }
    names = dict(enumerate(LABELS))
    model = VideoMAEForVideoClassification.from_pretrained(
        args.checkpoint, num_labels=7,
        id2label=names, label2id={name: i for i, name in names.items()},
        ignore_mismatched_sizes=True,
    ).cuda()
    model.gradient_checkpointing_enable()
    counts = Counter(window.label for window in groups["train"])
    weights = torch.tensor([1 / max(counts.get(i, 0), 1) ** 0.5 for i in range(7)], device="cuda")
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    backbone = list(model.videomae.parameters())
    head = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 1e-6}, {"params": head, "lr": 1e-5},
    ], weight_decay=0.05)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "data_source": "self_recorded_only", "pskus_data_used": False,
        "pskus_weights_used": False, "checkpoint": args.checkpoint,
        "input_size": [224, 224], "num_frames": args.num_frames,
        "window_stride": args.window_stride, "video_splits": {
            "train": [clip.clip_id for clip in fit],
            "validation": [clip.clip_id for clip in validation],
            "test": [clip.clip_id for clip in test],
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history = []
    best_f1, best_epoch, patience = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for parameter in model.videomae.parameters():
            parameter.requires_grad = epoch > 2
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (pixels, labels, *_metadata) in enumerate(loaders["train"], start=1):
            pixels, labels = pixels.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=pixels).logits
                loss = criterion(logits, labels) / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().item()) * args.grad_accum
        val_acc, val_f1, _, _ = _metrics(model, loaders["validation"], "cuda")
        record = {"epoch": epoch, "train_loss": running / max(len(loaders["train"]), 1),
                  "val_accuracy": val_acc, "val_macro_f1": val_f1}
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if val_f1 > best_f1:
            best_f1, best_epoch, patience = val_f1, epoch, 0
            model.save_pretrained(args.output / "best_model", safe_serialization=True)
            processor.save_pretrained(args.output / "best_model")
            (args.output / "selection.json").write_text(json.dumps({
                **record, "best_epoch": best_epoch, "selection_metric": "validation_macro_f1",
                "test_evaluated_after_selection": False,
            }, indent=2) + "\n")
        else:
            patience += 1
            if patience >= 5:
                break

    # Test is evaluated once, only after the validation-selected checkpoint is fixed.
    best = VideoMAEForVideoClassification.from_pretrained(args.output / "best_model").cuda()
    test_acc, test_f1, truth, _prediction = _metrics(best, loaders["test"], "cuda")
    rows = _prediction_rows(best, loaders["test"], "cuda")
    by_clip: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_clip.setdefault(str(row["clip_id"]), []).append(row)
    def collapse(values: list[int]) -> list[int]:
        result = []
        for value in values:
            if value > 0 and (not result or result[-1] != value):
                result.append(value)
        return result
    per_video = []
    for clip_id, clip_rows in by_clip.items():
        true_order = collapse([int(row["true_label"]) for row in clip_rows])
        predicted_order = collapse([int(row["predicted_label"]) for row in clip_rows])
        per_video.append({"clip_id": clip_id, "true_order": true_order,
                          "predicted_order": predicted_order,
                          "sequence_match": predicted_order == true_order})
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip_id", "time_s", "true_label",
                                                    "predicted_label", "confidence"))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "test_video_orders.json").write_text(
        json.dumps(per_video, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = {"best_epoch": best_epoch, "test_accuracy": test_acc, "test_macro_f1": test_f1,
               "test_windows": len(truth), "class_names": list(LABELS),
               "test_videos": len(per_video),
               "correct_sequence_videos": sum(bool(row["sequence_match"]) for row in per_video),
               "data_source": "self_recorded_only", "pskus_data_used": False,
               "pskus_weights_used": False}
    (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
