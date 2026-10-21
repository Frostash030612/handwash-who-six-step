#!/usr/bin/env python
"""Fine-tune VideoMAE on dual-annotated PSKUS video clips.

Each training sample is a 16-frame window with a two-annotator consensus
label. The default requires every sampled frame to be stable; the optional
center-stable policy retains a stable centre while including mixed context.
Videos are split before window generation to prevent leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if __package__ in (None, ""):
    from _bootstrap import PROJECT_ROOT
else:
    from ._bootstrap import PROJECT_ROOT

LABELS = (
    "0_other",
    "1_palm_to_palm",
    "2_palm_over_dorsum",
    "3_fingers_interlaced",
    "4_backs_of_fingers",
    "5_rotational_thumbs",
    "6_rotational_fingertips",
)
ANNOTATOR_PAIRS = {
    "DataSet1": ("Annotator2", "Annotator3"),
    "DataSet4": ("Annotator1", "Annotator2"),
}
SPLIT_SEED = 20260922
CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"


@dataclass(frozen=True)
class VideoInfo:
    source: str
    stem: str
    path: Path
    primary: Path
    secondary: Path
    split: str


@dataclass(frozen=True)
class Window:
    video_id: str
    source: str
    split: str
    label: int
    frame_indices: tuple[int, ...]


def _label(row: dict[str, str]) -> int:
    try:
        code = int(row["movement_code"])
        washing = str(row.get("is_washing", "1")).strip() == "1"
    except (KeyError, TypeError, ValueError):
        return -1
    if not washing or code in (0, 7):
        return 0
    return code if 1 <= code <= 6 else -1


def _read_csv(path: Path) -> tuple[list[int], list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels, times = [], []
    for row in rows:
        labels.append(_label(row))
        try:
            times.append(float(row["frame_time"]) / 1000.0)
        except (KeyError, TypeError, ValueError):
            times.append(float("nan"))
    return labels, times


def _annotation_pair(dataset: Path, stem: str) -> tuple[Path, Path] | None:
    """Choose two distinct annotators by a fixed priority, never by their labels.

    Keep the original DataSet1/4 pair for compatibility with existing exp.pt
    and VideoMAE splits. Other partitions prefer Annotator1/2 and fall back to
    the first available distinct pair.
    """
    annotation_root = dataset / "Annotations"
    available = sorted(path for path in annotation_root.glob("Annotator*/")
                       if (path / f"{stem}.csv").is_file())
    by_name = {path.name: path / f"{stem}.csv" for path in available}
    preferred = ANNOTATOR_PAIRS.get(dataset.name, ("Annotator1", "Annotator2"))
    if all(name in by_name for name in preferred):
        return by_name[preferred[0]], by_name[preferred[1]]
    if len(available) >= 2:
        return by_name[available[0].name], by_name[available[1].name]
    return None


def _discover_videos(root: Path, *, full_pskus: bool = False) -> list[VideoInfo]:
    candidates: list[tuple[str, str, Path, Path, Path]] = []
    sources = ([path.name for path in root.iterdir()
                if path.is_dir() and re.fullmatch(r"DataSet\d+", path.name)]
               if full_pskus and root.is_dir() else list(ANNOTATOR_PAIRS))
    for source in sorted(sources):
        dataset = root / source
        for video in sorted((dataset / "Videos").glob("*.mp4")):
            if full_pskus:
                pair = _annotation_pair(dataset, video.stem)
                if pair is None:
                    continue
                primary, secondary = pair
            else:
                fixed_pair = ANNOTATOR_PAIRS[source]
                primary = dataset / "Annotations" / fixed_pair[0] / f"{video.stem}.csv"
                secondary = dataset / "Annotations" / fixed_pair[1] / f"{video.stem}.csv"
                if not (primary.is_file() and secondary.is_file()):
                    continue
            candidates.append((source, video.stem, video, primary, secondary))

    def ranked(items):
        return sorted(items, key=lambda item: hashlib.sha256(
            f"{SPLIT_SEED}:{item[0]}/{item[1]}".encode()).hexdigest())

    # Preserve the original DataSet1+4 membership: both existing checkpoints
    # saw those training videos, so reshuffling them into test would leak data.
    groups = ([ranked([item for item in candidates if item[0] in ANNOTATOR_PAIRS]),
               ranked([item for item in candidates if item[0] not in ANNOTATOR_PAIRS])]
              if full_pskus else [ranked(candidates)])
    result = []
    for group in groups:
        train_end = round(len(group) * 0.70)
        val_end = train_end + round(len(group) * 0.15)
        for i, (source, stem, video, primary, secondary) in enumerate(group):
            split = "train" if i < train_end else "val" if i < val_end else "test"
            result.append(VideoInfo(source, stem, video, primary, secondary, split))
    return result


def _preserve_video_splits(videos: list[VideoInfo], reference: Path) -> list[VideoInfo]:
    """Reuse earlier video splits before fine-tuning a checkpoint trained on them."""
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    known = {}
    for row in rows:
        key = row["video_id"]
        if key in known or row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"旧划分有重复视频或无效划分：{key}")
        known[key] = row
    current = {f"{video.source}/{video.stem}" for video in videos}
    missing = set(known) - current
    if missing:
        raise ValueError(f"旧划分中有视频未找到，不能安全续训：{sorted(missing)[:5]}")
    new = [video for video in videos if f"{video.source}/{video.stem}" not in known]
    new.sort(key=lambda video: hashlib.sha256(
        f"{SPLIT_SEED}:{video.source}/{video.stem}".encode()).hexdigest())
    train_end = round(len(new) * 0.70)
    val_end = train_end + round(len(new) * 0.15)
    assigned = {
        f"{video.source}/{video.stem}":
        "train" if index < train_end else "val" if index < val_end else "test"
        for index, video in enumerate(new)
    }
    result = []
    for video in videos:
        key = f"{video.source}/{video.stem}"
        if key in known:
            row = known[key]
            if row["primary"] != video.primary.parent.name or row["secondary"] != video.secondary.parent.name:
                raise ValueError(f"旧标注员配对变化，不能安全续训：{key}")
            split = row["split"]
        else:
            split = assigned[key]
        result.append(replace(video, split=split))
    return result


def _windows(videos: list[VideoInfo], *, num_frames: int, sample_fps: float,
             window_stride: int, stable_radius: int,
             label_policy: str = "full_stable") -> list[Window]:
    if label_policy not in {"full_stable", "center_stable"}:
        raise ValueError(f"Unknown window label policy: {label_policy}")
    windows: list[Window] = []
    rejected_alignment = 0
    for video in videos:
        left, left_times = _read_csv(video.primary)
        right, right_times = _read_csv(video.secondary)
        shared = min(len(left), len(right), len(left_times), len(right_times))
        if not shared:
            rejected_alignment += 1
            continue
        agreed = [left[i] >= 0 and left[i] == right[i]
                  and abs(left_times[i] - right_times[i]) <= 0.001 for i in range(shared)]
        stable = [False] * shared
        for i, label in enumerate(left[:shared]):
            lo, hi = max(0, i - stable_radius), min(shared, i + stable_radius + 1)
            stable[i] = agreed[i] and all(agreed[j] and left[j] == label for j in range(lo, hi))

        import cv2

        cap = cv2.VideoCapture(str(video.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        source_stride = max(1, round(fps / sample_fps))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or shared)
        cap.release()
        max_index = min(shared, frame_count)
        video_id = f"{video.source}/{video.stem}"
        span = num_frames * source_stride
        hop = window_stride * source_stride
        for start in range(0, max(0, max_index - span + source_stride), hop):
            frame_ids = tuple(start + j * source_stride for j in range(num_frames))
            if frame_ids[-1] >= max_index:
                continue
            if label_policy == "full_stable":
                label = left[frame_ids[0]]
                if label < 0 or not all(stable[idx] and left[idx] == label for idx in frame_ids):
                    continue
            else:
                centre = frame_ids[num_frames // 2]
                label = left[centre]
                if not stable[centre]:
                    continue
            windows.append(Window(video_id, video.source, video.split, label, frame_ids))

    if rejected_alignment:
        print(f"双标注缺失或时间轴不齐，跳过 {rejected_alignment} 段视频", flush=True)
    return windows


def _cap_train_other(windows: list[Window], ratio: float) -> list[Window]:
    """Deterministically cap only training `other`; retain natural val/test."""
    if ratio < 0:
        raise ValueError("other-to-wash ratio 不能为负")
    wash_count = sum(w.split == "train" and 1 <= w.label <= 6 for w in windows)
    budget = round(wash_count * ratio)
    others = [w for w in windows if w.split == "train" and w.label == 0]
    others.sort(key=lambda w: hashlib.sha256(
        f"{w.video_id}:{','.join(map(str, w.frame_indices))}".encode()).digest())
    keep = {(w.video_id, w.frame_indices) for w in others[:budget]}
    return [w for w in windows if w.split != "train" or w.label != 0
            or (w.video_id, w.frame_indices) in keep]


def _extract_frames(videos: list[VideoInfo], windows: list[Window], cache_root: Path) -> dict[tuple[str, int], Path]:
    import cv2

    by_id = {f"{v.source}/{v.stem}": v for v in videos}
    wanted: dict[str, set[int]] = {}
    for window in windows:
        wanted.setdefault(window.video_id, set()).update(window.frame_indices)
    paths: dict[tuple[str, int], Path] = {}
    for number, (video_id, indices) in enumerate(sorted(wanted.items()), start=1):
        video = by_id[video_id]
        out_dir = cache_root / video.source / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        target = {idx: out_dir / f"f{idx:06d}.jpg" for idx in indices}
        remaining = {idx for idx, path in target.items() if not path.is_file()}
        cap = cv2.VideoCapture(str(video.path))
        frame_index = 0
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in remaining:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(target[frame_index]), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                remaining.remove(frame_index)
            frame_index += 1
        cap.release()
        if remaining:
            raise RuntimeError(f"视频提前结束，{video_id} 缺少源帧：{min(remaining)}")
        for idx, path in target.items():
            paths[(video_id, idx)] = path
        print(f"解码 {number}/{len(wanted)}：{video_id}，缓存 {len(indices)} 帧", flush=True)
    return paths


def _split_report(videos: list[VideoInfo], windows: list[Window]) -> dict[str, object]:
    video_splits: dict[str, set[str]] = {}
    for video in videos:
        video_splits.setdefault(f"{video.source}/{video.stem}", set()).add(video.split)
    if any(len(splits) != 1 for splits in video_splits.values()):
        raise RuntimeError("数据泄漏：同一原始视频出现在多个划分")
    return {
        "source_videos": Counter(video.split for video in videos),
        "windows": {split: Counter(LABELS[w.label] for w in windows if w.split == split) for split in ("train", "val", "test")},
    }


class ClipDataset:
    def __init__(self, windows: list[Window], paths: dict[tuple[str, int], Path], *, train: bool,
                 mean: tuple[float, float, float], std: tuple[float, float, float]):
        self.windows, self.paths, self.train = windows, paths, train
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        from PIL import Image
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        window = self.windows[index]
        frames = [Image.open(self.paths[(window.video_id, idx)]).convert("RGB") for idx in window.frame_indices]
        if self.train and random.random() < 0.5:
            frames = [frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for frame in frames]
        transform = Compose([
            Resize((224, 224)), ToTensor(),
            Normalize(self.mean, self.std),
        ])
        clip = torch.stack([transform(frame) for frame in frames])
        return clip, window.label


def _metrics(model, loader, device, amp: bool) -> tuple[float, float, list[int], list[int]]:
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(pixel_values=pixel_values).logits
            predictions.extend(logits.argmax(-1).cpu().tolist())
            targets.extend(labels.tolist())
    return (accuracy_score(targets, predictions),
            f1_score(targets, predictions, average="macro", zero_division=0), targets, predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/videomae_pskus")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="解码帧缓存目录；默认 <output>/frame_cache，可放到另一块磁盘")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--stable-radius", type=int, default=7)
    parser.add_argument("--window-label-policy", choices=("full_stable", "center_stable"),
                        default="full_stable",
                        help="Full-stable legacy labels or consensus-stable centre labels with mixed-context clips")
    parser.add_argument("--other-to-wash-ratio", type=float, default=None,
                        help="仅限制训练集 other 窗口的上限；如 1.0 表示不超过六步窗口总数，验证/测试不变")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2,
                        help="Initially train only the classifier; 0 for continued fine-tuning")
    parser.add_argument("--skip-test-eval", action="store_true",
                        help="Do not inspect the held-out test set during model selection")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-6,
                        help="主干网络微调学习率；较小值有助于稳定预训练权重")
    parser.add_argument("--head-lr", type=float, default=1e-5,
                        help="分类头学习率")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true", help="只统计标注一致性与可组成窗口数，不训练")
    parser.add_argument("--full-pskus", action="store_true", help="包含全部 DataSet*，保留已有 DataSet1+4 划分")
    parser.add_argument("--datasets", default=None,
                        help="使用指定分片，逗号分隔；配合 --full-pskus 使用")
    parser.add_argument("--split-reference", type=Path,
                        help="续训时保留既有 video_splits.csv 中所有视频的划分")
    args = parser.parse_args()

    if (args.datasets or args.split_reference) and not args.full_pskus:
        parser.error("--datasets 和 --split-reference 必须配合 --full-pskus")
    if args.freeze_backbone_epochs < 0:
        parser.error("--freeze-backbone-epochs must be non-negative")

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    videos = _discover_videos(args.root, full_pskus=args.full_pskus)
    if args.datasets:
        selected = {name.strip() for name in args.datasets.split(",") if name.strip()}
        if not selected or any(not re.fullmatch(r"DataSet(?:[1-9]|1[01])", name) for name in selected):
            parser.error(f"无效的 --datasets：{args.datasets}")
        available = {video.source for video in videos}
        if selected - available:
            parser.error(f"指定分片没有双标注视频：{sorted(selected - available)}")
        videos = [video for video in videos if video.source in selected]
    if args.split_reference:
        videos = _preserve_video_splits(videos, args.split_reference)
    if not videos:
        parser.error(f"在 {args.root} 找不到同时具有指定双人标注的视频")
    windows = _windows(videos, num_frames=args.num_frames, sample_fps=args.sample_fps,
                       window_stride=args.window_stride, stable_radius=args.stable_radius,
                       label_policy=args.window_label_policy)
    natural_windows = len(windows)
    if args.other_to_wash_ratio is not None:
        windows = _cap_train_other(windows, args.other_to_wash_ratio)
    report = _split_report(videos, windows)
    print(f"双人标注视频：{dict(report['source_videos'])}")
    print(f"各划分窗口类别：{report['windows']}")
    print(f"可用窗口：{len(windows)}（原始稳定窗口 {natural_windows}）")
    if args.inspect:
        return 0

    if not torch.cuda.is_available():
        parser.error("VideoMAE 训练需要 CUDA；请在 AutoDL 的 NVIDIA 实例中运行")

    if not any(window.split == "train" for window in windows) or not any(window.split == "val" for window in windows):
        parser.error("train 或 val 没有可用窗口；检查双标注和稳定区间后再训练")

    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root or args.output / "frame_cache"
    paths = _extract_frames(videos, windows, cache_root)
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained(args.checkpoint)
    mean = tuple(float(v) for v in processor.image_mean)
    std = tuple(float(v) for v in processor.image_std)
    split_data = {split: [w for w in windows if w.split == split] for split in ("train", "val", "test")}
    datasets = {
        split: ClipDataset(items, paths, train=(split == "train"), mean=mean, std=std)
        for split, items in split_data.items()
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        for split, dataset in datasets.items() if len(dataset)
    }

    id2label = dict(enumerate(LABELS))
    label2id = {label: i for i, label in id2label.items()}
    model = VideoMAEForVideoClassification.from_pretrained(
        args.checkpoint, num_labels=len(LABELS), label2id=label2id, id2label=id2label,
        ignore_mismatched_sizes=True,
    ).cuda()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    counts = Counter(w.label for w in split_data["train"])
    weights = torch.tensor([1.0 / max(counts.get(i, 0), 1) ** 0.5 for i in range(len(LABELS))], device="cuda")
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    backbone = list(model.videomae.parameters())
    head = list(model.classifier.parameters())
    if not torch.cuda.is_bf16_supported():
        parser.error("当前 CUDA 设备不支持 BF16；请改用受控的 FP32 训练")
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": head, "lr": args.head_lr},
    ], weight_decay=0.05)
    best_f1, best_epoch, patience = -1.0, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        # A newly initialized classifier benefits from a warm-up; continued
        # fine-tuning of an existing seven-class head may use zero warm-up.
        for parameter in model.videomae.parameters():
            parameter.requires_grad = epoch > args.freeze_backbone_epochs
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        bar = tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}")
        for step, (pixel_values, labels) in enumerate(bar, start=1):
            pixel_values = pixel_values.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels) / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"非有限训练损失：epoch={epoch}, step={step}, loss={loss.item()}")
            loss.backward()
            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().item()) * args.grad_accum
            bar.set_postfix(loss=f"{running / step:.3f}")

        val_acc, val_f1, _, _ = _metrics(model, loaders["val"], "cuda", amp=True)
        row = {"epoch": epoch, "train_loss": running / max(len(loaders["train"]), 1),
               "val_accuracy": val_acc, "val_macro_f1": val_f1}
        history.append(row)
        print(row, flush=True)
        with (args.output / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        if val_f1 > best_f1:
            best_f1, best_epoch, patience = val_f1, epoch, 0
            best_dir = args.output / "best_model"
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
            metadata = {"labels": list(LABELS), "num_frames": args.num_frames, "sample_fps": args.sample_fps,
                        "window_stride": args.window_stride, "checkpoint": args.checkpoint,
                        "annotation_policy": "DataSet1=2/3; DataSet4=1/2; others=1/2 then first available pair",
                        "full_pskus": args.full_pskus, "stable_radius_frames": args.stable_radius,
                        "window_label_policy": args.window_label_policy,
                        "freeze_backbone_epochs": args.freeze_backbone_epochs,
                        "source_datasets": sorted({video.source for video in videos}),
                        "other_to_wash_ratio_train_max": args.other_to_wash_ratio,
                        "split_reference": str(args.split_reference) if args.split_reference else None,
                        "split_reference_sha256": (hashlib.sha256(args.split_reference.read_bytes()).hexdigest()
                                                   if args.split_reference else None),
                        "best_epoch": best_epoch, "val_macro_f1": best_f1}
            (best_dir / "handwash_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            patience += 1
            if patience >= 5:
                print("验证集 Macro-F1 连续 5 轮未提升，提前停止", flush=True)
                break

    if "test" in loaders and not args.skip_test_eval:
        from transformers import VideoMAEForVideoClassification
        best_model = VideoMAEForVideoClassification.from_pretrained(args.output / "best_model").cuda()
        test_acc, test_f1, y_true, y_pred = _metrics(best_model, loaders["test"], "cuda", amp=True)
        from sklearn.metrics import classification_report
        result = {"best_epoch": best_epoch, "test_accuracy": test_acc, "test_macro_f1": test_f1,
                  "test_report": classification_report(y_true, y_pred, labels=list(range(len(LABELS))),
                                                        target_names=list(LABELS), output_dict=True,
                                                        zero_division=0)}
        (args.output / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"测试结果：accuracy={test_acc:.4f}, macro_f1={test_f1:.4f}", flush=True)
    print(f"最佳权重：{args.output / 'best_model'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
