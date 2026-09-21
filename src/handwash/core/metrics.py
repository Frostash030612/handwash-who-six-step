"""评估指标：Accuracy / Macro-F1 / Weighted-F1 / 各类别 P-R / 混淆矩阵 / Bootstrap CI。

为什么不用 ``sklearn.metrics`` 直接算？
    * 契约层禁止依赖 sklearn（它拖慢导入且版本行为有差异）；
    * 我们需要"哪些类别在标签空间里但测试集没出现过"这类领域逻辑，
      sklearn 的 ``labels=`` 参数处理起来很绕。
实现只有 numpy，行为完全可测。

所有函数约定：
    y_true / y_pred 为一维整型数组（通道下标），长度为样本数；
    ``num_classes`` 显式传入，保证混淆矩阵形状稳定（不会因某类缺失而缩水）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from handwash.errors import EvaluationError

__all__ = [
    "EPS",
    "confusion_matrix",
    "accuracy",
    "precision_recall_f1",
    "macro_f1",
    "weighted_f1",
    "per_class_report",
    "bootstrap_ci",
    "expected_calibration_error",
    "top_k_accuracy",
]

EPS = 1e-12


def _validate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_pred).astype(np.int64).ravel()
    if t.shape != p.shape:
        raise EvaluationError(f"y_true 与 y_pred 长度不一致：{t.shape} vs {p.shape}")
    if t.size == 0:
        raise EvaluationError("评估集为空：y_true 长度为 0")
    if num_classes < 1:
        raise EvaluationError(f"num_classes 必须 >= 1，实际 {num_classes}")
    if t.min() < 0 or t.max() >= num_classes or p.min() < 0 or p.max() >= num_classes:
        raise EvaluationError(
            f"标签越界：取值范围 [{min(t.min(), p.min())}, {max(t.max(), p.max())}]，"
            f"但 num_classes={num_classes}",
            hint="通常是标签映射写错，或模型输出通道数与标签空间不一致。",
        )
    return t, p


def confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
    normalize: Literal["true", "pred", "all"] | bool | None = None,
) -> np.ndarray:
    """混淆矩阵，``cm[i, j]`` = 真实 i 被判为 j 的样本数。

    ``normalize`` 取值：
        * ``None`` / ``False``：原始计数
        * ``"true"`` / ``True``：按行归一化（召回视角）
        * ``"pred"``：按列归一化（精确率视角）
        * ``"all"``：按总数归一化

    非法取值会**报错**而不是静默按某一列归一化 —— 静默切换视角会让
    报告里的混淆矩阵被误读，比直接失败危险得多。
    """
    t, p = _validate(y_true, y_pred, num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)

    if normalize is None or normalize is False:
        return cm

    if normalize is True or normalize == "true":
        axis: int | None = 1
    elif normalize == "pred":
        axis = 0
    elif normalize == "all":
        axis = None
    else:
        raise EvaluationError(
            f"normalize 取值非法：{normalize!r}",
            hint='允许：None / True / "true"（按行，召回视角）/ "pred"（按列，精确率视角）/ "all"。',
        )

    if axis is None:
        return cm / max(int(cm.sum()), 1)
    denom = cm.sum(axis=axis, keepdims=True)
    return cm / np.maximum(denom, EPS)


def accuracy(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray, *,
             num_classes: int | None = None) -> float:
    """总体准确率。``num_classes`` 仅用于越界检查，可为 None。"""
    t = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_pred).astype(np.int64).ravel()
    if num_classes is not None:
        t, p = _validate(y_true, y_pred, num_classes)
    elif t.shape != p.shape or t.size == 0:
        raise EvaluationError("y_true / y_pred 形状不一致或为空")
    return float((t == p).mean())


def precision_recall_f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """逐类别 precision / recall / f1 / support。

    约定（与 sklearn 一致）：分母为 0 时该指标记 0，避免 nan 污染宏平均。
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)  # 预测为该类的总数
    true_sum = cm.sum(axis=1).astype(np.float64)  # 真实为该类的总数

    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, true_sum


def macro_f1(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray, *,
             num_classes: int, ignore_absent: bool = True) -> float:
    """宏平均 F1。

    ``ignore_absent=True``：测试集中从未出现的类别不计入平均
    （小组项目里很常见：Kaggle 小型测试集可能完全没有某一步）。
    """
    _, _, f1, support = precision_recall_f1(y_true, y_pred, num_classes=num_classes)
    mask = support > 0 if ignore_absent else np.ones_like(support, dtype=bool)
    if not mask.any():
        return 0.0
    return float(f1[mask].mean())


def weighted_f1(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray, *,
                num_classes: int) -> float:
    """按 support 加权的 F1（类别不平衡时更贴近"整体表现"）。"""
    _, _, f1, support = precision_recall_f1(y_true, y_pred, num_classes=num_classes)
    total = support.sum()
    if total <= 0:
        return 0.0
    return float((f1 * support).sum() / total)


def per_class_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
    names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """逐类别指标字典，键为类别名（默认 "class_0" ...）。"""
    if names is None:
        names = [f"class_{i}" for i in range(num_classes)]
    if len(names) != num_classes:
        raise EvaluationError(f"names 长度 {len(names)} 与 num_classes={num_classes} 不一致")

    precision, recall, f1, support = precision_recall_f1(y_true, y_pred, num_classes=num_classes)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    report: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        report[str(name)] = {
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
            "predicted": int(cm[:, i].sum()),
            "correct": int(cm[i, i]),
        }
    return report


def top_k_accuracy(
    probs: np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    *,
    k: int = 3,
) -> float:
    """Top-k 准确率。``probs`` 形状 ``(N, C)``。"""
    p = np.asarray(probs, dtype=np.float64)
    t = np.asarray(y_true).astype(np.int64).ravel()
    if p.ndim != 2 or p.shape[0] != t.size:
        raise EvaluationError(f"probs 形状 {p.shape} 与 y_true 长度 {t.size} 不匹配")
    if k > p.shape[1]:
        raise EvaluationError(f"k={k} 超过类别数 {p.shape[1]}")
    topk = np.argsort(-p, axis=1)[:, :k]
    return float((topk == t[:, None]).any(axis=1).mean())


def expected_calibration_error(
    probs: np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    *,
    num_bins: int = 10,
) -> float:
    """ECE：置信度是否可信（报告里讨论"模型什么时候会错"很有用）。"""
    p = np.asarray(probs, dtype=np.float64)
    t = np.asarray(y_true).astype(np.int64).ravel()
    if p.ndim != 2 or p.shape[0] != t.size:
        raise EvaluationError(f"probs 形状 {p.shape} 与 y_true 长度 {t.size} 不匹配")
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == t).astype(np.float64)

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def bootstrap_ci(
    metric_fn,
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """自助法置信区间，返回 ``(点估计, 下界, 上界)``。

    用途：证明"改进模型比基线好"不是抽样波动（报告里必须给区间，不能只给单点）。
    """
    t = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_pred).astype(np.int64).ravel()
    if t.size != p.size or t.size == 0:
        raise EvaluationError("y_true / y_pred 形状不一致或为空")
    if num_samples < 1:
        raise EvaluationError("num_samples 必须 >= 1")

    point = float(metric_fn(t, p))
    rng = np.random.default_rng(seed)
    n = t.size
    stats = np.empty(num_samples, dtype=np.float64)
    for i in range(num_samples):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(metric_fn(t[idx], p[idx]))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return point, lo, hi


def summarize(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
    names: Sequence[str] | None = None,
) -> Mapping[str, object]:
    """一次性产出评估所需的全部标量指标（pipeline 直接落盘用）。"""
    return {
        "num_samples": int(np.asarray(y_true).ravel().size),
        "accuracy": round(accuracy(y_true, y_pred, num_classes=num_classes), 6),
        "macro_f1": round(macro_f1(y_true, y_pred, num_classes=num_classes), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, num_classes=num_classes), 6),
        "per_class": per_class_report(y_true, y_pred, num_classes=num_classes, names=names),
    }
