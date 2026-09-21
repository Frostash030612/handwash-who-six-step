"""评估指标契约测试（src/handwash/core/metrics.py）。

为什么逐个指标都要钉死：报告里的数字是这门课的主要交付物之一。如果宏平均 F1 把
"测试集里根本没出现的类别"也算进去，或者混淆矩阵因为某类缺失而缩水，两组成员的
结果就没法比较，也无法复现论文里的表格。
"""

from __future__ import annotations

import numpy as np
import pytest

from handwash.core.metrics import (
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    per_class_report,
    precision_recall_f1,
    summarize,
    top_k_accuracy,
    weighted_f1,
)
from handwash.errors import EvaluationError

pytestmark = pytest.mark.unit

#: 一个全部预测正确的 3 类小例子。
Y_TRUE_PERFECT = np.array([0, 1, 2, 0, 1, 2])

#: 一个维度已人工核算过的小例子（见下方各用例的注释）。
Y_TRUE_SMALL = np.array([0, 1, 2, 0, 1, 2])
Y_PRED_SMALL = np.array([0, 1, 1, 0, 2, 2])


# --- accuracy ---------------------------------------------------------------
def test_accuracy_is_one_for_perfect_prediction() -> None:
    """全对必须严格等于 1.0（浮点比较用精确等值，指标本身就是这么定义的）。"""
    assert accuracy(Y_TRUE_PERFECT, Y_TRUE_PERFECT, num_classes=3) == 1.0


def test_accuracy_matches_hand_computed_value() -> None:
    """Y_PRED_SMALL 有 4/6 正确（第 3、5 个样本判错），因此是 2/3。"""
    assert accuracy(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3) == pytest.approx(4 / 6)


def test_accuracy_without_num_classes_still_checks_length() -> None:
    """num_classes 只是越界检查开关，长度不一致无论如何都必须报错。"""
    with pytest.raises(EvaluationError):
        accuracy([0, 1], [0], num_classes=None)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([0, 1, 2], [0, 1]), ([0, 1], [0, 1, 2]), ([], [])],
)
def test_length_mismatch_or_empty_raises(y_true: list[int], y_pred: list[int]) -> None:
    """长度不一致通常是切片/掩码写错，越早失败越省时间。"""
    with pytest.raises(EvaluationError):
        accuracy(y_true, y_pred, num_classes=3)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([0, 1, 3], [0, 1, 2]), ([-1, 1, 2], [0, 1, 2]), ([0, 1, 2], [0, 1, 5])],
)
def test_out_of_range_labels_raise(y_true: list[int], y_pred: list[int]) -> None:
    """越界标签几乎总是"模型通道数 != 标签空间"的信号，绝不能静默裁掉。"""
    with pytest.raises(EvaluationError, match="越界"):
        accuracy(y_true, y_pred, num_classes=3)


@pytest.mark.parametrize("num_classes", [0, -1])
def test_non_positive_num_classes_raises(num_classes: int) -> None:
    """类别数为 0 会让混淆矩阵形状非法。"""
    with pytest.raises(EvaluationError):
        accuracy([0], [0], num_classes=num_classes)


# --- 混淆矩阵 ---------------------------------------------------------------
def test_confusion_matrix_counts_are_exact() -> None:
    """cm[i, j] = 真实 i 被判为 j 的样本数；这个矩阵是所有派生指标的源头。"""
    cm = confusion_matrix(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3)
    assert cm.tolist() == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]


def test_confusion_matrix_rows_sum_to_support() -> None:
    """每一行之和必须等于该真实类别的样本数，否则指标口径就错了。"""
    for labels in ([0, 0, 1, 2, 2, 2], [1, 1, 1], [0, 2, 2, 1, 0]):
        cm = confusion_matrix(labels, [0] * len(labels), num_classes=4)
        np.testing.assert_array_equal(cm.sum(axis=1), np.bincount(labels, minlength=4))


def test_confusion_matrix_shape_follows_num_classes_not_observed_labels() -> None:
    """形状只由标签空间决定：某类缺失时也要保留那一行，否则报告会缩水。"""
    cm = confusion_matrix([0, 0], [0, 0], num_classes=5)
    assert cm.shape == (5, 5)
    assert cm.sum() == 2


def test_confusion_matrix_normalize_true_rows_sum_to_one() -> None:
    """按真实类别归一化后每行和为 1（召回视角），用于画归一化混淆矩阵。"""
    normalized = confusion_matrix(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3, normalize="true")
    np.testing.assert_allclose(normalized.sum(axis=1), np.ones(3))


def test_confusion_matrix_normalize_pred_columns_sum_to_one() -> None:
    """按预测类别归一化后每列和为 1（精确率视角）。"""
    normalized = confusion_matrix(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3, normalize="pred")
    np.testing.assert_allclose(normalized.sum(axis=0), np.ones(3))


def test_confusion_matrix_normalize_pred_keeps_unpredicted_column_finite() -> None:
    """某类从未被预测时该列为全 0；必须保持有限值，不能出现 nan。"""
    normalized = confusion_matrix([0, 0], [0, 0], num_classes=2, normalize="pred")
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized[:, 1], np.zeros(2))


def test_confusion_matrix_normalize_handles_empty_column_without_nan() -> None:
    """没有样本被预测成的类别会产生 0/0；必须用 EPS 兜住，不能污染整张图。"""
    normalized = confusion_matrix([0, 0], [0, 0], num_classes=2, normalize="true")
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized[1], np.zeros(2))


# --- 逐类别 precision / recall / f1 -----------------------------------------
def test_precision_recall_f1_matches_hand_computation() -> None:
    """人工核算第 1 类：预测 2 次、命中 1 次 -> P=0.5；真实 2 次、命中 1 次 -> R=0.5。"""
    precision, recall, f1, support = precision_recall_f1(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3)
    np.testing.assert_allclose(precision, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(recall, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(f1, [1.0, 0.5, 0.5])
    np.testing.assert_array_equal(support, [2, 2, 2])


def test_precision_recall_f1_uses_zero_for_absent_class_instead_of_nan() -> None:
    """测试集没出现的类别记 0 而不是 nan —— nan 会顺着宏平均污染最终数字。"""
    precision, recall, f1, support = precision_recall_f1([0, 0], [0, 0], num_classes=3)
    assert support[2] == 0
    assert precision[2] == 0.0
    assert recall[2] == 0.0
    assert f1[2] == 0.0
    assert np.isfinite([precision, recall, f1]).all()


# --- macro / weighted F1 ----------------------------------------------------
def test_macro_f1_is_one_for_perfect_prediction() -> None:
    """全对时宏平均 F1 与准确率一样必须是 1.0。"""
    assert macro_f1(Y_TRUE_PERFECT, Y_TRUE_PERFECT, num_classes=3) == pytest.approx(1.0)


def test_macro_f1_ignores_absent_class_by_default() -> None:
    """默认只对"真实出现过的类别"取平均：小型 Kaggle 测试集常常整步缺失。"""
    present_only = macro_f1(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3)
    assert present_only == pytest.approx((1.0 + 0.5 + 0.5) / 3)


def test_macro_f1_includes_absent_class_when_explicitly_requested() -> None:
    """显式要求把缺失类别算 0 分时必须拉低总分（跨数据集报告会用这个口径）。"""
    strict = macro_f1(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=5, ignore_absent=False)
    assert strict == pytest.approx((1.0 + 0.5 + 0.5) / 5)
    assert strict < macro_f1(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=5)


def test_weighted_f1_weights_by_support() -> None:
    """类别不平衡时按 support 加权，更贴近"整体表现"。"""
    y_true = [0, 0, 0, 0, 1]
    assert weighted_f1(y_true, [0, 0, 0, 0, 1], num_classes=2) == pytest.approx(1.0)

    # 全部预测为 0 时：0 类 P=0.8/R=1.0 -> F1=0.8889（权重 4），1 类 F1=0（权重 1）
    # 加权 F1 = (0.8889*4 + 0.0*1) / 5 = 0.7111
    assert weighted_f1(y_true, [0, 0, 0, 0, 0], num_classes=2) == pytest.approx(0.7111111, abs=1e-6)


def test_weighted_f1_returns_zero_when_no_support() -> None:
    """空测试集不应该让加权 F1 变成 nan。"""
    assert weighted_f1([0], [0], num_classes=1) == pytest.approx(1.0)
    assert macro_f1([0, 1], [0, 1], num_classes=2, ignore_absent=True) == pytest.approx(1.0)


# --- per_class_report / summarize -------------------------------------------
def test_per_class_report_contains_counters_and_names() -> None:
    """per_class 是论文表格的直接来源，字段名与含义必须稳定。"""
    report = per_class_report(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3, names=["a", "b", "c"])
    assert list(report) == ["a", "b", "c"]
    assert report["a"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2,
        "predicted": 2,
        "correct": 2,
    }
    assert report["c"]["correct"] == 1
    assert report["c"]["predicted"] == 2


def test_per_class_report_rejects_mismatched_names() -> None:
    """类别名数量对不上会让报告把指标贴到错误的步骤上。"""
    with pytest.raises(EvaluationError, match="names"):
        per_class_report(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3, names=["a", "b"])


def test_per_class_report_default_names_follow_index_order() -> None:
    """未给名字时用 class_i，保证输出结构永远一致。"""
    report = per_class_report(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3)
    assert list(report) == ["class_0", "class_1", "class_2"]


def test_summarize_bundles_scalars_and_per_class() -> None:
    """pipeline 一次性落盘用的汇总结构，键名不能随意改。"""
    summary = summarize(Y_TRUE_SMALL, Y_PRED_SMALL, num_classes=3)
    assert summary["num_samples"] == 6
    assert summary["accuracy"] == pytest.approx(4 / 6)
    assert summary["macro_f1"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert set(summary["per_class"]) == {"class_0", "class_1", "class_2"}


# --- top-k / ECE ------------------------------------------------------------
def test_top_k_accuracy_is_at_least_accuracy() -> None:
    """top-k 是准确率的上界：k 变大只会让更多样本算命中。"""
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
            [0.4, 0.35, 0.25],
        ]
    )
    y_true = np.array([0, 2, 2, 2])
    top1 = top_k_accuracy(probs, y_true, k=1)
    top3 = top_k_accuracy(probs, y_true, k=3)
    assert top1 == pytest.approx(accuracy(y_true, np.argmax(probs, axis=1), num_classes=3))
    assert top3 >= top1
    assert top3 == pytest.approx(1.0)


def test_top_k_accuracy_rejects_mismatched_shapes_and_too_large_k() -> None:
    """形状不匹配与 k 超界都属于调用错误，必须显式报错。"""
    probs = np.eye(3)
    with pytest.raises(EvaluationError):
        top_k_accuracy(probs, [0, 1], k=2)
    with pytest.raises(EvaluationError, match="超过类别数"):
        top_k_accuracy(probs, [0, 1, 2], k=4)


def test_top_k_accuracy_rejects_one_dimensional_probs() -> None:
    """probs 必须是 (N, C) 的分布矩阵，一维数组应被拒绝。"""
    with pytest.raises(EvaluationError):
        top_k_accuracy(np.array([0.5, 0.5]), [0], k=1)


def test_ece_is_zero_for_perfectly_calibrated_confident_predictions() -> None:
    """置信度 1.0 且全部正确时，ECE 必须为 0：说明"说 100% 就是 100%"。"""
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y_true = np.array([0, 1, 1])
    assert expected_calibration_error(probs, y_true, num_bins=10) == pytest.approx(0.0)


def test_ece_is_positive_when_model_is_overconfident() -> None:
    """自信却答错时 ECE 必须大于 0，否则这个指标没有诊断价值。"""
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    y_true = np.array([0, 1])
    assert expected_calibration_error(probs, y_true, num_bins=10) > 0.0


def test_ece_rejects_mismatched_shapes() -> None:
    """probs 行数与标签数不一致说明推理与标注错位。"""
    with pytest.raises(EvaluationError):
        expected_calibration_error(np.eye(3), [0, 1], num_bins=5)


# --- bootstrap CI -----------------------------------------------------------
def test_bootstrap_ci_brackets_point_estimate_with_fixed_seed() -> None:
    """报告必须给区间而不是单点：证明改进不是抽样波动。"""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])

    def metric(t: np.ndarray, p: np.ndarray) -> float:
        return accuracy(t, p)

    point, low, high = bootstrap_ci(metric, y_true, y_pred, num_samples=200, seed=7)
    assert point == pytest.approx(accuracy(y_true, y_pred, num_classes=3))
    assert low <= point <= high


def test_bootstrap_ci_is_reproducible_with_same_seed() -> None:
    """同一种子必须给出同一区间，否则报告里的置信区间每次都变，无法复现。"""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 1, 3, 0, 1, 2, 0, 0, 1, 2, 3, 3, 1, 2, 3])

    def metric(t: np.ndarray, p: np.ndarray) -> float:
        return macro_f1(t, p, num_classes=4)

    first = bootstrap_ci(metric, y_true, y_pred, num_samples=100, seed=123)
    second = bootstrap_ci(metric, y_true, y_pred, num_samples=100, seed=123)
    assert first == second
    assert first[1] < first[0] < first[2]  # 数据本身有错分，区间必须是真区间


def test_bootstrap_ci_differs_for_different_seed() -> None:
    """换种子应该给出不同的区间，说明重采样确实发生了（否则 CI 是假的）。"""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 1, 3, 0, 1, 2, 0, 0, 1, 2, 3, 3, 1, 2, 3])

    def metric(t: np.ndarray, p: np.ndarray) -> float:
        return macro_f1(t, p, num_classes=4)

    first = bootstrap_ci(metric, y_true, y_pred, num_samples=200, seed=1)
    second = bootstrap_ci(metric, y_true, y_pred, num_samples=200, seed=2)
    assert first[0] == second[0]  # 点估计与种子无关
    assert (first[1], first[2]) != (second[1], second[2])


@pytest.mark.parametrize(("num_samples", "seed"), [(0, 0), (-1, 0)])
def test_bootstrap_ci_rejects_non_positive_num_samples(num_samples: int, seed: int) -> None:
    """重采样次数必须为正，否则分位数无定义。"""
    with pytest.raises(EvaluationError, match="num_samples"):
        bootstrap_ci(lambda t, p: accuracy(t, p), [0, 1], [0, 1], num_samples=num_samples, seed=seed)


def test_bootstrap_ci_rejects_empty_input() -> None:
    """空输入无法给出置信区间。"""
    with pytest.raises(EvaluationError):
        bootstrap_ci(lambda t, p: accuracy(t, p), [], [], num_samples=10)
