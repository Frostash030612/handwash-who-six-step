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
from handwash.errors import DataError, EvaluationError
from handwash.io.utils import append_jsonl
from handwash.io.video import extract_frames, probe_video
from handwash.logging import get_logger
from handwash.models.voting import fuse_probabilities, sliding_window_probs
from handwash.pipelines.common import resolve_device

__all__ = ["InferenceOutput", "predict_clip", "predict_video", "extract_clip_from_video", "save_predictions"]

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
    indices, stamps, frames = extract_frames(path, sample_fps=sample_fps, max_frames=max_frames)
    if not frames:
        raise DataError(f"视频没有解出任何帧：{path}")
    effective_fps = sample_fps if sample_fps > 0 else (meta.fps or 1.0)
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
) -> InferenceOutput:
    """对一段 ``Clip`` 做推理，返回逐帧概率与标签。"""
    from handwash.data.transforms import build_transforms

    device = device or resolve_device(rc.runtime.device)
    batch_size = batch_size or rc.infer.batch_size
    space = get_label_space(rc.label_space)
    transform = build_transforms(rc.model, mode="infer", seed=rc.runtime.seed)

    frames = list(clip.frames)
    if not frames:
        raise DataError(f"clip={clip.clip_id} 没有帧")

    mode = rc.infer.mode if rc.infer.mode != "frame" else rc.train.mode
    window = max(1, rc.model.temporal.window)

    if mode in ("clip", "hybrid") and window > 1 and len(frames) >= window:
        clip_probs = _window_probs(model, transform, frames, window=window, batch_size=batch_size, device=device)
    else:
        clip_probs = None
        if mode == "clip":
            log.warning(
                "帧数（%d）少于时序窗口（%d），自动退回逐帧推理", len(frames), window
            )

    frame_probs = _frame_probs(model, transform, frames, batch_size=batch_size, device=device)

    if mode == "hybrid" and clip_probs is not None:
        probs = fuse_probabilities(frame_probs, clip_probs, temporal_weight=0.5)
    elif mode == "clip" and clip_probs is not None:
        probs = clip_probs
    else:
        probs = frame_probs

    if rc.infer.temporal_apply and rc.infer.smooth_window > 1:
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
) -> np.ndarray:
    """逐帧推理 -> ``(T, C)`` 概率。"""
    outputs: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        tensor = _preprocess(chunk, transform).to(device)
        logits = model.logits(tensor)  # (B, 1, C)
        probs = torch.softmax(logits[:, 0, :], dim=-1)
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
) -> np.ndarray:
    """滑窗推理 -> ``(T, C)``。

    重叠窗口对同一帧给出多个预测时取**算术平均**（而不是取最大）：
    平均更稳，也避免"某一窗口偶然高置信"主导结果。
    每个窗口至少覆盖一次全部帧（末尾不足处回退对齐到序列尾部）。
    """
    n = len(frames)
    starts = list(range(0, max(1, n - window + 1), max(1, window // 2)))
    if not starts or starts[-1] + window < n:
        starts.append(max(0, n - window))

    acc_counts = np.zeros(n, dtype=np.float64)  # 每帧被多少个窗口覆盖
    prob_sum: np.ndarray | None = None
    counts = acc_counts

    batch: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []
    for start in starts:
        span_frames = frames[start : start + window]
        if len(span_frames) < window:
            continue
        batch.append(_preprocess(span_frames, transform))
        spans.append((start, start + window))
        if len(batch) >= max(1, batch_size // window):
            prob_sum, counts = _accumulate(model, batch, spans, prob_sum, counts, device)
            batch, spans = [], []
    if batch:
        prob_sum, counts = _accumulate(model, batch, spans, prob_sum, counts, device)

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
) -> tuple[np.ndarray, np.ndarray]:
    """把一个 batch 的窗口概率累加到逐帧累加器上。"""
    stacked = torch.stack(list(batch), dim=0).to(device)  # (B, T, 3, H, W)
    logits = model.logits(stacked)  # (B, T, C)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    if prob_sum is None:
        prob_sum = np.zeros((counts.shape[0], probs.shape[-1]), dtype=np.float64)
    for b, (start, end) in enumerate(spans):
        prob_sum[start:end] += probs[b]
        counts[start:end] += 1.0
    return prob_sum, counts


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
    return append_jsonl(path, rows)
