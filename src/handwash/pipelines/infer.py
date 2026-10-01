"""推理流程：一段视频（或一个 clip）-> 逐帧动作预测。

三种模式的差别（必须在报告里写清楚用的是哪种）
------------------------------------------------
``frame``  逐帧独立分类（最快，也是"单帧基线"）
``clip``   按时序窗口推理，窗口内上下文共享（GRU/TCN 用）
``hybrid`` 逐帧与窗口结果按 ``fuse_probabilities`` 加权融合

输出
----
``ClipPrediction``（见 core.schema）与可选的 ``predictions.jsonl``：
每行一帧的 ``clip_id / frame_index / label / confidence / probs``。

设计约束：本模块**不做**业务判定（漏步/顺序/时长），那些全在 ``core.protocol``。
推理只负责"给出每帧的概率分布"。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from handwash.core.config import ResolvedConfig
from handwash.core.labels import get_label_space
from handwash.core.schema import Clip, ClipPrediction, FramePrediction
from handwash.core.temporal import window_starts
from handwash.errors import DataError, EvaluationError
from handwash.io.utils import write_jsonl
from handwash.io.video import extract_frames, probe_video, write_video
from handwash.logging import get_logger
from handwash.models.voting import fuse_probabilities, sliding_window_probs
from handwash.pipelines.common import resolve_device

__all__ = ["InferenceOutput", "extract_clip_from_video", "predict_clip", "predict_video", "save_predictions"]

log = get_logger(__name__)


class InferenceOutput:
    """推理结果包装：方便 assess 与 CLI 共用。"""

    def __init__(self, prediction: ClipPrediction, probs: np.ndarray, fps: float) -> None:
        self.prediction = prediction
        self.probs = probs  # (T, C)
        self.fps = fps

    @property
    def clip_id(self) -> str:
        return self.prediction.clip_id

    def labels(self) -> list:
        return [frame.label for frame in self.prediction.frames]

    def confidences(self) -> list[float]:
        return [frame.confidence for frame in self.prediction.frames]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "fps": self.fps,
            "num_frames": self.prediction.num_frames,
            "model_name": self.prediction.model_name,
            "labels": [step.value for step in self.labels()],
            "confidences": [round(c, 6) for c in self.confidences()],
        }


def extract_clip_from_video(path: str | Path, *, sample_fps: float, max_frames: int | None = None) -> Clip:
    """把视频解码成 ``Clip``（RGB uint8，时间顺序）。"""
    meta = probe_video(path)
    _indices, stamps, frames = extract_frames(path, sample_fps=sample_fps, max_frames=max_frames)
    if not frames:
        raise DataError(f"视频没有解出任何帧：{path}")
    positive_deltas = [b - a for a, b in zip(stamps[:-1], stamps[1:], strict=True) if b > a]
    if positive_deltas:
        effective_fps = 1.0 / float(np.median(positive_deltas))
    elif meta.fps > 0 and sample_fps > 0:
        stride = max(1, int(round(meta.fps / sample_fps)))
        effective_fps = meta.fps / stride
    else:
        effective_fps = meta.fps or sample_fps or 1.0
    log.info(
        "解码完成：%s（原始 %.1ffps / %d 帧 -> 抽取 %d 帧 @ %.1ffps）",
        Path(path).name, meta.fps or 0.0, meta.frame_count, len(frames), effective_fps,
    )
    return Clip(clip_id=Path(path).stem, frames=tuple(frames), fps=float(effective_fps), source=str(path))


def _preprocess(frames: Sequence[np.ndarray], transform) -> torch.Tensor:
    """把帧数组变成模型输入张量。"""
    processed = [np.asarray(transform(np.asarray(frame))) for frame in frames]
    return torch.as_tensor(np.stack(processed, axis=0), dtype=torch.float32)


@torch.no_grad()
def predict_clip(
    rc: ResolvedConfig,
    model: torch.nn.Module,
    clip: Clip,
    *,
    device: torch.device | None = None,
    batch_size: int | None = None,
    apply_smoothing: bool = True,
) -> InferenceOutput:
    """对一段 ``Clip`` 做推理，返回逐帧概率与标签。"""
    from handwash.data.transforms import build_transforms

    device = device or resolve_device(rc.runtime.device)
    batch_size = rc.infer.batch_size if batch_size is None else int(batch_size)
    if batch_size < 1:
        raise DataError(f"推理 batch_size 必须为正整数，实际 {batch_size}")
    space = get_label_space(rc.label_space)
    transform = build_transforms(rc.model, mode="infer", seed=rc.runtime.seed)

    frames = list(clip.frames)
    if not frames:
        raise DataError(f"clip={clip.clip_id} 没有帧")

    mode = rc.infer.mode
    window = max(1, rc.model.temporal.window)
    effective_window = min(window, len(frames))

    if mode in ("clip", "hybrid") and effective_window > 1:
        clip_probs = _window_probs(
            model,
            transform,
            frames,
            window=effective_window,
            batch_size=batch_size,
            device=device,
            tta=rc.infer.tta,
            stride=min(rc.model.temporal.stride, effective_window),
        )
    else:
        clip_probs = None
        if mode == "clip" and window > 1:
            log.warning(
                "片段只有 %d 帧，无法使用时序上下文；退回逐帧推理", len(frames)
            )

    frame_probs = None
    if mode in ("frame", "hybrid") or clip_probs is None:
        frame_probs = _frame_probs(
            model, transform, frames, batch_size=batch_size, device=device, tta=rc.infer.tta
        )

    if mode == "hybrid" and clip_probs is not None:
        assert frame_probs is not None
        probs = fuse_probabilities(frame_probs, clip_probs, temporal_weight=0.5)
    elif mode == "clip" and clip_probs is not None:
        probs = clip_probs
    else:
        assert frame_probs is not None
        probs = frame_probs

    if apply_smoothing and rc.infer.temporal_apply and rc.infer.smooth_window > 1:
        probs = sliding_window_probs(probs, window=rc.infer.smooth_window)

    labels_idx = probs.argmax(axis=1)
    frame_predictions = tuple(
        FramePrediction(
            frame_index=i,
            label=space.to_label(int(labels_idx[i])),
            confidence=float(probs[i, labels_idx[i]]),
            probs=probs[i].astype(np.float32),
            timestamp_s=i / clip.fps,
        )
        for i in range(len(frames))
    )
    prediction = ClipPrediction(
        clip_id=clip.clip_id,
        frames=frame_predictions,
        label_sequence=tuple(fp.label for fp in frame_predictions),
        model_name=getattr(model, "arch_name", "unknown"),
    )
    return InferenceOutput(prediction, probs, clip.fps)


@torch.no_grad()
def _frame_probs(
    model: torch.nn.Module,
    transform,
    frames: Sequence[np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    """逐帧推理 -> ``(T, C)`` 概率。"""
    outputs: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        tensor = _preprocess(chunk, transform).to(device)
        probs = _batch_probabilities(model, tensor, tta=tta)[:, 0, :]
        outputs.append(probs.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def _window_probs(
    model: torch.nn.Module,
    transform,
    frames: Sequence[np.ndarray],
    *,
    window: int,
    batch_size: int,
    device: torch.device,
    tta: bool = False,
    stride: int | None = None,
) -> np.ndarray:
    """滑窗推理 -> ``(T, C)``。

    重叠窗口对同一帧给出多个预测时取**算术平均**（而不是取最大）：
    平均更稳，也避免"某一窗口偶然高置信"主导结果。
    所有帧都会被覆盖；末尾不足一个完整窗口时保留较短窗口。
    """
    n = len(frames)
    stride = window if stride is None else int(stride)
    starts = window_starts(n, window, stride)

    acc_counts = np.zeros(n, dtype=np.float64)  # 每帧被多少个窗口覆盖
    prob_sum: np.ndarray | None = None
    counts = acc_counts

    batches: dict[int, list[torch.Tensor]] = {}
    span_batches: dict[int, list[tuple[int, int]]] = {}
    windows_per_batch = max(1, batch_size // window)
    for start in starts:
        span_frames = frames[start : start + window]
        length = len(span_frames)
        batches.setdefault(length, []).append(_preprocess(span_frames, transform))
        span_batches.setdefault(length, []).append((start, start + length))
        if len(batches[length]) >= windows_per_batch:
            prob_sum, counts = _accumulate(
                model, batches.pop(length), span_batches.pop(length), prob_sum, counts, device, tta=tta
            )
    for length in sorted(batches):
        prob_sum, counts = _accumulate(
            model, batches[length], span_batches[length], prob_sum, counts, device, tta=tta
        )

    if prob_sum is None:
        raise EvaluationError("滑窗推理没有产生任何窗口，请检查 window 与帧数")
    return prob_sum / np.maximum(counts, 1.0)[:, None]


@torch.no_grad()
def _accumulate(
    model: torch.nn.Module,
    batch: Sequence[torch.Tensor],
    spans: Sequence[tuple[int, int]],
    prob_sum: np.ndarray | None,
    counts: np.ndarray,
    device: torch.device,
    *,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """把一个 batch 的窗口概率累加到逐帧累加器上。"""
    stacked = torch.stack(list(batch), dim=0).to(device)  # (B, T, 3, H, W)
    probs = _batch_probabilities(model, stacked, tta=tta).detach().cpu().numpy()
    if prob_sum is None:
        prob_sum = np.zeros((counts.shape[0], probs.shape[-1]), dtype=np.float64)
    for b, (start, end) in enumerate(spans):
        prob_sum[start:end] += probs[b]
        counts[start:end] += 1.0
    return prob_sum, counts


@torch.no_grad()
def _batch_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    tta: bool,
) -> torch.Tensor:
    """Average original and horizontal-flip probabilities when TTA is enabled."""
    probabilities = torch.softmax(model.logits(images), dim=-1)
    if not tta:
        return probabilities
    flipped = torch.flip(images, dims=(-1,))
    flipped_probabilities = torch.softmax(model.logits(flipped), dim=-1)
    return (probabilities + flipped_probabilities) * 0.5


def predict_video(
    rc: ResolvedConfig,
    model: torch.nn.Module,
    video_path: str | Path,
    *,
    device: torch.device | None = None,
) -> InferenceOutput:
    """从视频文件直接推理（自动解码 + 抽帧）。"""
    clip = extract_clip_from_video(video_path, sample_fps=rc.dataset.prep.fps)
    return predict_clip(rc, model, clip, device=device)


def save_predictions(output: InferenceOutput, path: str | Path) -> Path:
    """把逐帧预测写进 JSONL（人工复核失败案例用）。"""
    rows = [
        {
            "clip_id": output.clip_id,
            "frame_index": frame.frame_index,
            "timestamp_s": round(frame.timestamp_s or 0.0, 4),
            "label": frame.label.value,
            "confidence": round(frame.confidence, 6),
        }
        for frame in output.prediction.frames
    ]
    return write_jsonl(path, rows)


def save_overlay_video(
    rc: ResolvedConfig,
    video_path: str | Path,
    output: InferenceOutput,
    path: str | Path,
) -> Path:
    """Render prediction labels over the same uniformly sampled input frames."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - Pillow is a declared dependency
        raise EvaluationError("生成叠加预览需要 Pillow") from exc

    from handwash.core.labels import STEP_ZH

    _, _, frames = extract_frames(video_path, sample_fps=rc.dataset.prep.fps)
    if len(frames) != output.prediction.num_frames:
        raise EvaluationError(
            f"叠加视频帧数 {len(frames)} 与预测帧数 {output.prediction.num_frames} 不一致",
            hint="视频在推理后发生变化，或解码器在两次读取时返回了不同的帧数。",
        )
    drawn: list[np.ndarray] = []
    for frame, label, confidence in zip(frames, output.labels(), output.confidences(), strict=True):
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        text = f"{label.order_index or '-'} {STEP_ZH.get(label, label.value)}  {confidence:.2f}"
        draw.rectangle([0, 0, image.width, 26], fill=(0, 0, 0, 160))
        draw.text((6, 6), text, fill=(255, 255, 255, 255))
        drawn.append(np.asarray(image))
    return write_video(drawn, path, fps=output.fps)
