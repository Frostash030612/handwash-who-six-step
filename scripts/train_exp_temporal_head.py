#!/usr/bin/env python
"""训练 exp.pt 的 GRU 时序头（exp.pt 本身冻结），供摄像头 / 本地视频页面使用。

为什么需要：exp.pt 逐帧分类。手部大半移出画面时，单帧信息不足以区分第 3—6 步，
模型会连续数秒把第 3/4/5 步认成第 6 步；平滑窗口只能消除零星抖动，纠正不了
这种同向错误。时序头读取过去帧的特征序列，借上下文判断当前步骤。

划分与 exp.pt 训练集完全一致（同一随机种子、按原始视频 70/15/15）：
时序头只用 train 划分训练，在 val 划分上挑选轮次，test 划分只用于报告。
标签取全部标注者的投票分布，标注者意见不一的帧也参与训练。

用法（首次约 15 分钟，主要是提取特征；特征有缓存，重训只需一两分钟）：
    python scripts/train_exp_temporal_head.py
        [--pskus-dir ../pskus_dataset1_raw/DataSet1 --pskus-dir ../pskus_dataset4_raw/DataSet4]

完成后生成项目根目录的 exp_temporal_head.pt；
``python scripts/run_camera.py --demo-exp`` 会自动加载它。
"""

from __future__ import annotations

import csv
import hashlib
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT
from _common import build_config_parser, load_config_from_args

import numpy as np
import torch
from torch import nn

from handwash.cli import split_argv
from handwash.core.config import AssessConfig, parse_overrides
from handwash.core.labels import STEP_ORDER, Step
from handwash.core.protocol import build_report
from handwash.errors import DataError, HandwashError
from handwash.logging import setup_logging
from handwash.paths import ensure_dir
from handwash.pipelines.common import resolve_device
from handwash.pipelines.exp_demo import (
    DEFAULT_TEMPORAL_HEAD,
    EXP_CLASS_NAMES,
    ExpFeatureExtractor,
    ExpTemporalHead,
    file_sha256,
    save_temporal_head,
)

# 必须与生成 exp.pt 训练集的 prepare_dataset1_dataset4_high_confidence.py 一致，
# 否则 test 划分的视频会混进时序头的训练集。
_SPLIT_SEED = 20260922
_SPLIT_ANNOTATORS = {
    "DataSet1": ("Annotator2", "Annotator3"),
    "DataSet4": ("Annotator1", "Annotator2"),
}
_ALL_ANNOTATORS = ("Annotator1", "Annotator2", "Annotator3", "Annotator4")
_LABELS = (Step.OTHER, *STEP_ORDER)  # 与 EXP_CLASS_NAMES 的通道顺序一致
_NUM_CLASSES = len(EXP_CLASS_NAMES)


# ---------------------------------------------------------------------------
# 数据：划分、标注、特征缓存
# ---------------------------------------------------------------------------
def _split_videos(dataset_dirs: list[Path]) -> list[tuple[str, str, Path, str]]:
    """返回 ``(数据集名, 视频名, 数据集目录, 划分)``，规则与 exp.pt 训练集相同。"""
    names = sorted(path.name for path in dataset_dirs)
    if names != sorted(_SPLIT_ANNOTATORS):
        raise DataError(
            f"需要同时提供 DataSet1 与 DataSet4，实际 {names}",
            hint="exp.pt 的 train/val/test 是在这两个数据集合起来后划分的，缺一个就无法复现划分。",
        )
    videos = []
    for root in dataset_dirs:
        first, second = _SPLIT_ANNOTATORS[root.name]
        for video in (root / "Videos").glob("*.mp4"):
            if all((root / "Annotations" / a / f"{video.stem}.csv").exists() for a in (first, second)):
                videos.append((root.name, video.stem, root))
    if not videos:
        raise DataError("没有找到带双标注的 PSKUS 视频", hint="检查 --pskus-dir 是否指向 DataSet1 / DataSet4 目录。")
    videos.sort(key=lambda item: hashlib.sha256(f"{_SPLIT_SEED}:{item[0]}/{item[1]}".encode()).hexdigest())
    train_end = round(len(videos) * 0.70)
    val_end = train_end + round(len(videos) * 0.15)
    return [
        (source, stem, root, "train" if index < train_end else "val" if index < val_end else "test")
        for index, (source, stem, root) in enumerate(videos)
    ]


def _read_annotations(root: Path, stem: str, frame_indices: list[int]) -> np.ndarray:
    """每位标注者在抽样帧上的类别；``-1`` 表示该标注者缺失或该帧超出标注范围。

    is_washing=0 与 movement_code=7（关水龙头）都归入 0 类 other，与 exp.pt 一致。
    """
    labels = np.full((len(_ALL_ANNOTATORS), len(frame_indices)), -1, dtype=np.int8)
    for row_index, annotator in enumerate(_ALL_ANNOTATORS):
        path = root / "Annotations" / annotator / f"{stem}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for column, frame_index in enumerate(frame_indices):
            if frame_index >= len(rows):
                continue
            row = rows[frame_index]
            code = int(row["movement_code"])
            if row["is_washing"] != "1" or code in (0, 7):
                labels[row_index, column] = 0
            elif 1 <= code <= 6:
                labels[row_index, column] = code
    return labels


def _extract_video(extractor: ExpFeatureExtractor, video: Path, fps: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise DataError(f"无法打开视频：{video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(source_fps / fps))
    frames: list[np.ndarray] = []
    indices: list[int] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)
            indices.append(index)
        index += 1
    capture.release()
    if not frames:
        raise DataError(f"视频没有可读取的画面：{video}")
    features, probabilities = [], []
    for start in range(0, len(frames), 64):
        batch_features, batch_probabilities = extractor.forward_bgr(frames[start : start + 64])
        features.append(batch_features)
        probabilities.append(batch_probabilities)
    return np.concatenate(features), np.concatenate(probabilities), indices


def _load_dataset(
    videos: list[tuple[str, str, Path, str]],
    *,
    exp_path: Path,
    exp_sha: str,
    cache_dir: Path,
    fps: float,
    device: str,
    runtime_dir: Path,
) -> dict[str, list[dict]]:
    extractor: ExpFeatureExtractor | None = None
    data: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    started = time.time()
    for number, (source, stem, root, split) in enumerate(videos, start=1):
        cache = cache_dir / f"{source}__{stem}.npz"
        item = None
        if cache.exists():
            stored = np.load(cache)
            if str(stored["exp_sha256"]) == exp_sha and float(stored["fps"]) == fps:
                item = {key: stored[key] for key in ("features", "probs", "annotations")}
        if item is None:
            if extractor is None:
                extractor = ExpFeatureExtractor(exp_path, device=device, cache_dir=runtime_dir)
            features, probabilities, indices = _extract_video(extractor, root / "Videos" / f"{stem}.mp4", fps)
            item = {
                "features": features.astype(np.float16),
                "probs": probabilities,
                "annotations": _read_annotations(root, stem, indices),
            }
            np.savez_compressed(cache, exp_sha256=exp_sha, fps=fps, **item)
            print(
                f"  特征 {number}/{len(videos)} {source}/{stem}（{split}，{len(indices)} 帧，"
                f"已用 {time.time() - started:.0f}s）",
                flush=True,
            )
        annotations = item["annotations"].astype(np.int64)
        votes = np.zeros((annotations.shape[1], _NUM_CLASSES), dtype=np.float32)
        for row in annotations:
            valid = row >= 0
            votes[np.flatnonzero(valid), row[valid]] += 1.0
        counts = votes.sum(axis=1, keepdims=True)
        data[split].append(
            {
                "key": f"{source}/{stem}",
                "features": item["features"].astype(np.float32),
                "probs": item["probs"].astype(np.float32),
                "annotations": annotations,
                "target": votes / np.maximum(counts, 1.0),
                "mask": (counts[:, 0] > 0).astype(np.float32),
            }
        )
    return data


# ---------------------------------------------------------------------------
# 评估：与摄像头会话相同的解码与规则报告
# ---------------------------------------------------------------------------
def _baseline_probs(probs: np.ndarray, window: int) -> np.ndarray:
    """现有页面的做法：exp.pt 概率对过去 ``window`` 帧求平均。"""
    return np.stack([probs[max(0, t - window + 1) : t + 1].mean(axis=0) for t in range(len(probs))])


@torch.no_grad()
def _head_probs(head: ExpTemporalHead, features: np.ndarray) -> np.ndarray:
    head.eval()
    logits, _ = head(torch.from_numpy(features)[None])
    return torch.softmax(logits[0], dim=-1).numpy()


def _who_sequence(labels: list[Step], confidences: list[float] | None, cfg: AssessConfig, fps: float) -> list[int]:
    report = build_report(
        clip_id="eval", labels=labels, confidences=confidences, fps=fps, cfg=cfg, apply_smoothing=False
    )
    return [STEP_ORDER.index(step) + 1 for step in report.step_sequence if step in STEP_ORDER]


def _edit_distance(left: list[int], right: list[int]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        previous, row[0] = row[:], i
        for j, b in enumerate(right, start=1):
            row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (a != b))
    return row[-1]


def _has_inversion(sequence: list[int]) -> bool:
    return any(later < earlier for i, earlier in enumerate(sequence) for later in sequence[i + 1 :])


def _evaluate(videos: list[dict], probs_of, cfg: AssessConfig, fps: float) -> dict[str, float]:
    """``probs_of(video) -> [T, 7]``；指标均与人工标注逐段视频对比。

    * frame_acc：所有标注者一致的帧上的逐帧准确率；
    * seq_edit：预测的步骤序列与最接近的那位标注者序列的编辑距离之和 / 标注序列长度之和；
    * seq_exact：预测序列与至少一位标注者完全相同的视频占比；
    * false_order：预测报“乱序”而所有标注者序列都没有乱序的视频数。
    """
    correct = total = edits = length = exact = false_order = 0
    for video in videos:
        probs = probs_of(video)
        indices = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        labels = [
            _LABELS[i] if c >= cfg.min_confidence else Step.UNKNOWN
            for i, c in zip(indices, confidences, strict=True)
        ]
        predicted = _who_sequence(labels, confidences.tolist(), cfg, fps)
        references = [
            _who_sequence([_LABELS[max(int(c), 0)] for c in row], None, cfg, fps)
            for row in video["annotations"]
            if (row >= 0).any()
        ]
        edits += min(_edit_distance(predicted, ref) for ref in references)
        length += max(max(len(ref) for ref in references), 1)
        exact += any(predicted == ref for ref in references)
        false_order += _has_inversion(predicted) and not any(_has_inversion(ref) for ref in references)
        for t, label in enumerate(labels):
            column = {int(c) for c in video["annotations"][:, t] if c >= 0}
            if len(column) == 1:
                total += 1
                correct += label is _LABELS[column.pop()]
    return {
        "frame_acc": round(correct / max(total, 1), 4),
        "seq_edit": round(edits / max(length, 1), 4),
        "seq_exact": round(exact / max(len(videos), 1), 4),
        "false_order": int(false_order),
        "videos": len(videos),
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def _batch(videos: list[dict], rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """每段视频随机截取 12—40 秒（5 fps 下 60—200 帧），右侧补零并给出掩码。"""
    crops = []
    for video in videos:
        length = len(video["features"])
        size = min(length, rng.randint(60, 200))
        start = rng.randint(0, length - size)
        crops.append(slice(start, start + size))
    longest = max(c.stop - c.start for c in crops)
    features = torch.zeros(len(videos), longest, videos[0]["features"].shape[1])
    target = torch.zeros(len(videos), longest, _NUM_CLASSES)
    mask = torch.zeros(len(videos), longest)
    for row, (video, crop) in enumerate(zip(videos, crops, strict=True)):
        size = crop.stop - crop.start
        features[row, :size] = torch.from_numpy(video["features"][crop])
        target[row, :size] = torch.from_numpy(video["target"][crop])
        mask[row, :size] = torch.from_numpy(video["mask"][crop])
    return features, target, mask


def _train(
    data: dict[str, list[dict]], *, epochs: int, seed: int, cfg: AssessConfig, fps: float
) -> tuple[ExpTemporalHead, int, dict[str, float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, val = list(data["train"]), data["val"]
    stacked = np.concatenate([video["features"] for video in train])
    head = ExpTemporalHead(feature_dim=stacked.shape[1])
    head.feature_mean.copy_(torch.from_numpy(stacked.mean(axis=0)))
    head.feature_std.copy_(torch.from_numpy(stacked.std(axis=0) + 1e-3))
    del stacked
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

    best_score, best_state, best_epoch, best_metrics = float("inf"), None, -1, {}
    for epoch in range(1, epochs + 1):
        head.train()
        rng.shuffle(train)
        losses = []
        for start in range(0, len(train), 8):
            features, target, mask = _batch(train[start : start + 8], rng)
            # exp.pt 见过 train 划分的画面，这些帧的特征比 val/test 更“干净”；
            # 加噪声防止时序头过度相信单帧特征，从而学会依赖上下文。
            features = features + 0.5 * head.feature_std * torch.randn_like(features)
            log_probs = torch.log_softmax(head(features)[0], dim=-1)
            loss = -((target * log_probs).sum(dim=-1) * mask).sum() / mask.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = _evaluate(val, lambda video: _head_probs(head, video["features"]), cfg, fps)
        score = metrics["seq_edit"] - metrics["frame_acc"]
        marker = ""
        if score < best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            marker = "  ← 当前最佳"
        print(f"  第 {epoch:2d}/{epochs} 轮 loss={np.mean(losses):.3f} val={metrics}{marker}", flush=True)
    assert best_state is not None
    head.load_state_dict(best_state)
    return head.eval(), best_epoch, best_metrics


def _default_pskus_dirs() -> list[Path]:
    parent = PROJECT_ROOT.parent
    candidates = [parent / "pskus_dataset1_raw" / "DataSet1", parent / "pskus_dataset4_raw" / "DataSet4"]
    return [path for path in candidates if path.is_dir()]


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--pskus-dir", action="append", default=None, help="PSKUS 的 DataSet1 / DataSet4 目录（各给一次）")
    parser.add_argument("--exp", default=str(PROJECT_ROOT / "exp.pt"), help="exp.pt 路径（默认项目根目录）")
    parser.add_argument("--output", default=str(PROJECT_ROOT / DEFAULT_TEMPORAL_HEAD), help="时序头输出路径")
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数（按 val 指标保留最佳一轮）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    plain, override_items = split_argv(sys.argv[1:])
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    setup_logging(args.log_level, force=True)
    overrides = parse_overrides(override_items)
    overrides["runtime"] = {"run_name": "exp_demo", **overrides.get("runtime", {})}
    rc = load_config_from_args(args, overrides)

    try:
        dataset_dirs = [Path(p).expanduser().resolve() for p in args.pskus_dir] if args.pskus_dir else _default_pskus_dirs()
        exp_path = Path(args.exp).expanduser().resolve()
        if not exp_path.is_file():
            raise DataError(f"找不到 exp.pt：{exp_path}")
        fps = float(rc.dataset.prep.fps)
        cfg = rc.assess
        exp_sha = file_sha256(exp_path)
        videos = _split_videos(dataset_dirs)
        counts = {split: sum(1 for *_rest, s in videos if s == split) for split in ("train", "val", "test")}
        print(f"视频划分（与 exp.pt 训练集一致）：{counts}", flush=True)

        print("1/3 提取 exp.pt 特征（有缓存则跳过）…", flush=True)
        data = _load_dataset(
            videos,
            exp_path=exp_path,
            exp_sha=exp_sha,
            cache_dir=ensure_dir(rc.resolve_out_dir() / "temporal_head" / "features"),
            fps=fps,
            device=str(resolve_device(rc.runtime.device)),
            runtime_dir=ensure_dir(rc.resolve_out_dir() / "demo_runtime"),
        )

        print("2/3 训练 GRU 时序头（CPU）…", flush=True)
        head, best_epoch, _ = _train(data, epochs=args.epochs, seed=args.seed, cfg=cfg, fps=fps)

        print("3/3 对比：现有做法（exp.pt + 过去帧概率平均） vs 时序头", flush=True)
        window = max(1, cfg.smooth_window)
        results: dict[str, dict[str, float]] = {}
        for split in ("val", "test"):
            results[f"baseline_{split}"] = _evaluate(
                data[split], lambda video: _baseline_probs(video["probs"], window), cfg, fps
            )
            results[f"head_{split}"] = _evaluate(
                data[split], lambda video: _head_probs(head, video["features"]), cfg, fps
            )
        print(f"{'':<16}{'逐帧准确率↑':>10}{'序列编辑率↓':>12}{'序列全对↑':>10}{'乱序误报↓':>10}")
        for name, metrics in results.items():
            print(
                f"{name:<16}{metrics['frame_acc']:>12.3f}{metrics['seq_edit']:>14.3f}"
                f"{metrics['seq_exact']:>12.3f}{metrics['false_order']:>10d}/{metrics['videos']}"
            )
        output = save_temporal_head(
            args.output, head, exp_sha256=exp_sha, fps=fps, metrics={"best_epoch": best_epoch, **results}
        )
        print(f"时序头已保存：{output}（第 {best_epoch} 轮）")
        print("下一步：python scripts/run_camera.py --demo-exp ，页面载入视频即会使用时序头。")
    except HandwashError as exc:
        print(f"时序头训练失败：{exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
