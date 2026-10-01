"""基线的时序后处理：多数投票 / 滑动窗口 / 概率滑动平均。

这三者就是选题文档里"加入时序信息以减少单帧误判"的**最简形式**，
也是论文 baseline（MobileNetV2 + 后处理）的对应实现。
它们与 ``core.protocol.smooth_labels`` 的区别：

    * 本模块工作在**概率/特征**层面（有置信度可用，可做加权）；
    * ``core.protocol.smooth_labels`` 工作在**标签**层面（判定用，无概率）。

两者都必须实现，且报告里要说明用的是哪一种 —— 否则无法解释指标差异。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from handwash.core.labels import Step
from handwash.errors import EvaluationError

__all__ = [
    "majority_vote",
    "sliding_window_probs",
    "fuse_probabilities",
    "temporal_vote_accuracy_note",
]


def majority_vote(labels: Sequence[Step], *, window: int = 9) -> list[Step]:
    """多数投票（无概率版）。

    与 ``core.protocol.smooth_labels`` 的区别：本函数不做置信度过滤，
    纯粹按标签计数，用于"没有概率输出"的场景（例如读取别人的标注结果）。
    """
    if window < 1:
        raise EvaluationError(f"window 必须 >= 1，实际 {window}")
    if window % 2 == 0:
        window += 1
    seq = list(labels)
    if not seq:
        return []
    half = window // 2
    out: list[Step] = []
    for i in range(len(seq)):
        lo, hi = max(0, i - half), min(len(seq), i + half + 1)
        counts = Counter(seq[lo:hi])
        best = max(counts.values())
        tied = [lab for lab, c in counts.items() if c == best]
        # 平票保持时间连贯：优先继承上一帧
        chosen = out[-1] if (out and out[-1] in tied) else tied[0]
        out.append(chosen)
    return out


def sliding_window_probs(probs: np.ndarray, *, window: int = 9) -> np.ndarray:
    """对逐帧概率做滑动平均，返回同形状数组。

    ``probs``: ``(T, C)``，每行和为 1。窗口边界自动收缩（不做零填充，
    否则会人为压低边界处的置信度）。
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise EvaluationError(f"probs 应为 (T, C) 二维数组，实际 shape={arr.shape}")
    if window < 1:
        raise EvaluationError(f"window 必须 >= 1，实际 {window}")
    if window == 1:
        return arr.copy()

    half = window // 2
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        lo, hi = max(0, i - half), min(arr.shape[0], i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    # 重新归一化，避免浮点误差累积导致 argmax 不稳
    sums = out.sum(axis=1, keepdims=True)
    return out / np.maximum(sums, 1e-12)


def fuse_probabilities(
    frame_probs: np.ndarray,
    temporal_probs: np.ndarray,
    *,
    temporal_weight: float = 0.5,
) -> np.ndarray:
    """融合"帧级模型"与"时序模型"的概率（用于 ``train.mode=hybrid``）。

    ``temporal_weight`` 越大越信任时序模型。0 等价于纯帧级，1 等价于纯时序。
    这是报告中"时序处理带来多少提升"的直接实验旋钮。
    """
    a = np.asarray(frame_probs, dtype=np.float64)
    b = np.asarray(temporal_probs, dtype=np.float64)
    if a.shape != b.shape:
        raise EvaluationError(f"两组概率形状不一致：{a.shape} vs {b.shape}")
    if not 0.0 <= temporal_weight <= 1.0:
        raise EvaluationError(f"temporal_weight 必须在 [0,1]，实际 {temporal_weight}")
    fused = (1.0 - temporal_weight) * a + temporal_weight * b
    sums = fused.sum(axis=1, keepdims=True)
    return fused / np.maximum(sums, 1e-12)


def temporal_vote_accuracy_note() -> str:
    """写进报告的固定说明，避免各组员对"时序处理"的理解不一致。"""
    return (
        "时序处理共实现三种：① 多数投票/滑动窗口（后处理，无需训练）；"
        "② 概率滑动平均（后处理，使用置信度）；③ GRU/TCN 时序头（需训练）。"
        "报告中必须写明使用哪一种，并在同一划分上对比，否则提升幅度不可比。"
    )
