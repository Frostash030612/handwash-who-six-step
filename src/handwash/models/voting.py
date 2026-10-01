"""时序概率平滑与帧级/窗口级融合。"""

from __future__ import annotations

import numpy as np


def sliding_window_probs(probs: np.ndarray, *, window: int) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"probs 应为 (T,C)，实际 {values.shape}")
    if window <= 1 or len(values) <= 1:
        return values.astype(np.float32, copy=True)
    radius = window // 2
    padded = np.pad(values, ((radius, window - radius - 1), (0, 0)), mode="edge")
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)))
    smoothed = (cumulative[window:] - cumulative[:-window]) / window
    totals = smoothed.sum(axis=1, keepdims=True)
    return (smoothed / np.maximum(totals, 1e-12)).astype(np.float32)


def fuse_probabilities(
    frame_probs: np.ndarray,
    clip_probs: np.ndarray,
    *,
    temporal_weight: float = 0.5,
) -> np.ndarray:
    frame = np.asarray(frame_probs, dtype=np.float64)
    clip = np.asarray(clip_probs, dtype=np.float64)
    if frame.shape != clip.shape:
        raise ValueError(f"帧级与时序概率形状不一致：{frame.shape} vs {clip.shape}")
    if not 0.0 <= temporal_weight <= 1.0:
        raise ValueError("temporal_weight 必须在 [0,1]")
    fused = (1.0 - temporal_weight) * frame + temporal_weight * clip
    return (fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
