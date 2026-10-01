"""评估流程：内部测试 + 跨场景测试 + 混淆矩阵与逐类别报表。

一次 ``handwash evaluate`` 产出
--------------------------------
::

    outputs/<run_name>/eval/
      eval_val.json            各 split 的 EvalResult（见 core.schema）
      eval_test.json
      eval_external_<name>.json   跨场景（例如 METC）
      confusion_matrix_<split>.png
      per_class_<split>.csv    便于直接贴进报告表格
      predictions_<split>.jsonl  逐帧预测（可复核失败案例）

指标口径固定在 ``core.metrics``：Accuracy / Macro-F1 / Weighted-F1 / 逐类 P-R /
混淆矩阵。报告里**必须**同时给 Macro-F1 与混淆矩阵 —— 只看 Accuracy 会在
类别不平衡时得出错误结论。
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from handwash.core.config import ResolvedConfig
from handwash.core.labels import Step, get_label_space
from handwash.core.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_report,
    weighted_f1,
)
from handwash.core.schema import Clip, EvalResult, FrameRecord
from handwash.core.seeding import seed_everything
from handwash.errors import EvaluationError
from handwash.io.utils import write_csv, write_json, write_jsonl
from handwash.logging import get_logger
from handwash.models.checkpoint import load_checkpoint
from handwash.paths import BEST_CHECKPOINT_NAME, checkpoint_path, ensure_dir
from handwash.pipelines.common import label_names, resolve_device, resolve_split_records

__all__ = ["EvalBundle", "evaluate", "evaluate_single_split"]

log = get_logger(__name__)


class EvalBundle:
    """多 split 评估结果集合。"""

    def __init__(self) -> None:
        self.results: dict[str, EvalResult] = {}

    def add(self, result: EvalResult) -> None:
        self.results[result.split] = result

    def to_dict(self) -> dict[str, Any]:
        return {name: res.to_dict() for name, res in self.results.items()}

    def summary_table(self) -> str:
        """控制台表格：一眼看出哪个 split 掉了多少。"""
        lines = [f"{'split':<18}{'samples':>9}{'acc':>9}{'macroF1':>10}{'weightedF1':>12}"]
        lines.append("-" * 58)
        for name, res in self.results.items():
            lines.append(
                f"{name:<18}{res.num_samples:>9}{res.accuracy:>9.4f}"
                f"{res.macro_f1:>10.4f}{res.weighted_f1:>12.4f}"
            )
        return "\n".join(lines)


def _load_model_from_checkpoint(rc: ResolvedConfig, checkpoint: str | Path, *, device: torch.device):
    """加载模型 + checkpoint。

    这里刻意做**双重校验**（类别数 + 标签空间）：只要有一个不一致就报错，
    因为"用错标签空间的权重评估"会得到看似合理但完全错误的指标。
    """
    from handwash.models.factory import build_model

    space = get_label_space(rc.label_space)
    # 用 checkpoint 里的 arch/mode 覆盖配置，避免"配置写了 A、权重是 B"
    payload = load_checkpoint(
        checkpoint,
        map_location=device,
        expect_num_classes=len(space),
        expect_label_space=rc.label_space,
    )
    metadata = payload
    expected = {
        "arch": rc.model.arch,
        "image_size": rc.model.image_size,
        "normalize": rc.model.normalize,
        "mode": rc.train.mode,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise EvaluationError(
                f"checkpoint {key} 不匹配：配置为 {value!r}，权重记录为 {metadata.get(key)!r}",
                hint="评估时加载训练该 checkpoint 时保存的配置，尤其要保持 arch、mode、image_size 和 normalize 一致。",
            )
    if metadata.get("temporal") != rc.model.temporal.to_dict():
        raise EvaluationError(
            "checkpoint temporal 配置与当前配置不匹配",
            hint="使用训练该 checkpoint 时的 model.temporal 配置。",
        )
    model = build_model(rc, pretrained=False).to(device)
    try:
        model.load_state_dict(payload["state"], strict=True)
    except RuntimeError as exc:
        raise EvaluationError(
            f"checkpoint 参数与模型结构不匹配：{checkpoint}",
            hint=str(exc),
        ) from exc
    model.eval()
    return model, payload


@torch.no_grad()
def evaluate_single_split(
    rc: ResolvedConfig,
    model: torch.nn.Module,
    *,
    split: str,
    device: torch.device,
    checkpoint_name: str = "unknown",
    dataset_name: str | None = None,
    result_split: str | None = None,
) -> tuple[EvalResult, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """评估单个 split，返回 ``(EvalResult, y_true, y_pred, 逐帧明细)``。"""
    records = resolve_split_records(rc, split=split, dataset_name=dataset_name)
    target_label_space = rc.label_space
    from handwash.paths import DATA_PROCESSED_DIRNAME, resolve_relative

    if dataset_name is not None:
        target_spec = rc.dataset_spec(dataset_name)
        target_label_space = str(target_spec.get("label_space") or dataset_name)
        image_root = resolve_relative(target_spec.get("frames_dir") or f"{DATA_PROCESSED_DIRNAME}/{dataset_name}/frames")
    else:
        source_spec = rc.dataset_spec()
        image_root = resolve_relative(
            source_spec.get("frames_dir")
            or f"{DATA_PROCESSED_DIRNAME}/{rc.dataset.name}/frames"
        )
    space = get_label_space(target_label_space)
    source_space = get_label_space(rc.label_space)
    source_to_target, metric_labels = _label_projection(source_space, space)
    num_classes = len(metric_labels)

    model.eval()
    from PIL import Image

    from handwash.pipelines.infer import predict_clip

    grouped: dict[str, list[FrameRecord]] = {}
    for record in records:
        grouped.setdefault(record.clip_id, []).append(record)

    y_true_parts: list[int] = []
    y_pred_parts: list[int] = []
    details: list[dict[str, Any]] = []
    for clip_id, clip_records in sorted(grouped.items()):
        ordered = sorted(clip_records, key=lambda record: record.frame_index)
        images: list[np.ndarray] = []
        for record in ordered:
            image_path = Path(record.image_path)
            if not image_path.is_absolute():
                image_path = image_root / image_path
            try:
                with Image.open(image_path) as handle:
                    images.append(np.asarray(handle.convert("RGB")).copy())
            except (OSError, ValueError) as exc:
                raise EvaluationError(
                    f"无法读取评估图像：{image_path}",
                    hint="检查 manifest 与 frames_dir 是否来自同一份 prepare 产物。",
                ) from exc

        timed_stamps = [
            (index, float(record.timestamp_s))
            for index, record in enumerate(ordered)
            if record.timestamp_s is not None
        ]
        timestamp_deltas = [
            (right_time - left_time) / (right_position - left_position)
            for (left_position, left_time), (right_position, right_time) in zip(
                timed_stamps[:-1], timed_stamps[1:], strict=True
            )
            if right_position > left_position and right_time > left_time
        ]
        fps = (
            1.0 / float(np.median(timestamp_deltas))
            if timestamp_deltas
            else float(rc.dataset.prep.fps)
        )
        if not np.isfinite(fps) or fps <= 0:
            raise EvaluationError(f"clip={clip_id} 的有效帧率无效：{fps}")
        prediction = predict_clip(
            rc,
            model,
            Clip(clip_id=clip_id, frames=tuple(images), fps=fps),
            device=device,
            batch_size=rc.infer.batch_size,
        )
        source_probs = np.asarray(prediction.probs, dtype=np.float64)
        if source_probs.shape != (len(ordered), len(source_to_target)):
            raise EvaluationError(
                f"clip={clip_id} 模型概率形状 {source_probs.shape} 与期望 "
                f"({len(ordered)}, {len(source_to_target)}) 不一致"
            )
        probs = np.zeros((len(ordered), num_classes), dtype=np.float64)
        for source_index, target_index in enumerate(source_to_target):
            probs[:, target_index] += source_probs[:, source_index]
        preds = probs.argmax(axis=1)

        for record, row_probs, pred_idx in zip(ordered, probs, preds, strict=True):
            true_idx = space.to_index(record.label)
            pred_idx = int(pred_idx)
            y_true_parts.append(true_idx)
            y_pred_parts.append(pred_idx)
            details.append(
                {
                    "clip_id": clip_id,
                    "frame_index": record.frame_index,
                    "true": space.to_label(true_idx).value,
                    "pred": metric_labels[pred_idx].value,
                    "confidence": round(float(row_probs[pred_idx]), 6),
                }
            )

    if not details:
        raise EvaluationError(f"split={split} 没有可评估的样本")

    y_true = np.asarray(y_true_parts, dtype=np.int64)
    y_pred = np.asarray(y_pred_parts, dtype=np.int64)
    names = [step.value for step in metric_labels]
    confidence_intervals = None
    if rc.eval.bootstrap_ci:
        confidence_intervals = _clip_bootstrap_intervals(
            y_true,
            y_pred,
            [str(item["clip_id"]) for item in details],
            num_classes=num_classes,
            num_samples=rc.eval.bootstrap_samples,
            seed=rc.runtime.seed + zlib.crc32(str(result_split or split).encode("utf-8")),
        )
    result = EvalResult(
        split=result_split or split,
        model_name=checkpoint_name,
        num_samples=int(y_true.size),
        accuracy=round(accuracy(y_true, y_pred, num_classes=num_classes), 6),
        macro_f1=round(macro_f1(y_true, y_pred, num_classes=num_classes), 6),
        weighted_f1=round(weighted_f1(y_true, y_pred, num_classes=num_classes), 6),
        per_class=per_class_report(y_true, y_pred, num_classes=num_classes, names=names),
        confusion_matrix=tuple(
            tuple(int(x) for x in row) for row in confusion_matrix(y_true, y_pred, num_classes=num_classes)
        ),
        labels=tuple(names),
        confidence_intervals=confidence_intervals,
    )
    log.info(
        "split=%s：%d 帧 | acc=%.4f macroF1=%.4f weightedF1=%.4f",
        split, result.num_samples, result.accuracy, result.macro_f1, result.weighted_f1,
    )
    return result, y_true, y_pred, details


def evaluate(
    rc: ResolvedConfig,
    *,
    checkpoint: str | Path | None = None,
    splits: Sequence[str] | None = None,
) -> EvalBundle:
    """评估入口。

    ``checkpoint=None`` 时自动找 ``<out_dir>/models/best.pt``。
    """
    seed_everything(rc.runtime.seed, deterministic=rc.runtime.deterministic)
    device = resolve_device(rc.runtime.device)
    run_dir = ensure_dir(rc.resolve_out_dir())
    eval_dir = ensure_dir(run_dir / "eval")

    ckpt = Path(checkpoint) if checkpoint else checkpoint_path(run_dir, BEST_CHECKPOINT_NAME)
    if not ckpt.exists():
        raise EvaluationError(
            f"找不到 checkpoint：{ckpt}",
            hint="先训练，或用 --checkpoint 指定权重路径。",
        )

    model, _payload = _load_model_from_checkpoint(rc, ckpt, device=device)
    wanted = list(splits or rc.eval.splits)

    bundle = EvalBundle()
    for split in wanted:
        target_datasets = rc.eval.extra_datasets if split == "external" else ()
        target_datasets = target_datasets or (None,)
        for dataset_name in target_datasets:
            output_split = f"{split}_{dataset_name}" if dataset_name else split
            result, _, _, details = evaluate_single_split(
                rc,
                model,
                split=split,
                device=device,
                checkpoint_name=str(ckpt),
                dataset_name=dataset_name,
                result_split=output_split,
            )
            bundle.add(result)
            write_json(eval_dir / f"eval_{output_split}.json", result.to_dict())

            names = list(result.labels)
            write_csv(
                eval_dir / f"per_class_{output_split}.csv",
                [
                    {"label": name, **stats}
                    for name, stats in result.per_class.items()
                ],
                fieldnames=["label", "precision", "recall", "f1", "support", "predicted", "correct"],
            )
            if rc.eval.save_confusion_matrix:
                _save_confusion_figure(
                    np.asarray(result.confusion_matrix),
                    names,
                    eval_dir / f"confusion_matrix_{output_split}.png",
                    output_split,
                )
            if rc.eval.save_predictions:
                write_jsonl(eval_dir / f"predictions_{output_split}.jsonl", details)

    write_json(eval_dir / "eval_summary.json", bundle.to_dict())
    log.info("评估完成：\n%s", bundle.summary_table())
    return bundle


def _label_projection(source_space, target_space) -> tuple[list[int], tuple[Step, ...]]:
    """Map model outputs into the target dataset's label space.

    Source-only classes are combined into target ``other`` so cross-domain
    metrics compare equivalent labels and preserve all probability mass. If the
    target has no ``other`` class, add one to the metric labels so predictions
    of source-only classes count as false positives instead of aborting eval.
    """
    target_indices = {step: index for index, step in enumerate(target_space.labels)}
    metric_labels = list(target_space.labels)
    source_only = [step for step in source_space.labels if step not in target_indices]
    other_index = target_indices.get(Step.OTHER)
    if source_only and other_index is None:
        other_index = len(metric_labels)
        metric_labels.append(Step.OTHER)
    projection: list[int] = []
    for step in source_space.labels:
        index = target_indices.get(step, other_index)
        if index is None:
            raise EvaluationError(
                f"无法把模型类别 {step.value!r} 映射到目标标签空间 {target_space.name!r}",
                hint="目标空间需包含同名类别，或包含 `other` 以承接目标数据集没有的类别。",
            )
        projection.append(index)
    return projection, tuple(metric_labels)


def _clip_bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clip_ids: Sequence[str],
    *,
    num_classes: int,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap whole clips to preserve correlation among neighboring frames."""
    if len(clip_ids) != len(y_true) or len(y_pred) != len(y_true):
        raise EvaluationError("clip bootstrap 输入长度不一致")
    grouped: dict[str, list[int]] = {}
    for index, clip_id in enumerate(clip_ids):
        grouped.setdefault(clip_id, []).append(index)
    clips = sorted(grouped)
    result: dict[str, Any] = {
        "unit": "clip",
        "num_clips": len(clips),
        "num_samples": num_samples,
    }
    if len(clips) < 2:
        result.update({"available": False, "reason": "至少需要 2 段视频才能估计区间"})
        return result

    rng = np.random.default_rng(seed)
    accuracy_values = np.empty(num_samples, dtype=np.float64)
    f1_values = np.empty(num_samples, dtype=np.float64)
    grouped_true = {clip: y_true[grouped[clip]] for clip in clips}
    grouped_pred = {clip: y_pred[grouped[clip]] for clip in clips}
    for draw in range(num_samples):
        picked = rng.choice(clips, size=len(clips), replace=True)
        sample_true = np.concatenate([grouped_true[clip] for clip in picked])
        sample_pred = np.concatenate([grouped_pred[clip] for clip in picked])
        accuracy_values[draw] = accuracy(sample_true, sample_pred, num_classes=num_classes)
        f1_values[draw] = macro_f1(sample_true, sample_pred, num_classes=num_classes)

    result["available"] = True
    result["accuracy"] = {
        "low": float(np.quantile(accuracy_values, 0.025)),
        "high": float(np.quantile(accuracy_values, 0.975)),
    }
    result["macro_f1"] = {
        "low": float(np.quantile(f1_values, 0.025)),
        "high": float(np.quantile(f1_values, 0.975)),
    }
    return result


def _save_confusion_figure(cm: np.ndarray, names: Sequence[str], path: Path, split: str) -> Path | None:
    """画混淆矩阵（按行归一化）。

    matplotlib 是可选依赖：没装就跳过热图，但指标 JSON 仍然完整。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无显示环境（服务器/CI）必须用 Agg
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - 环境缺 matplotlib
        log.warning("未安装 matplotlib，跳过混淆矩阵图片（指标 JSON 不受影响）")
        return None

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.7), max(5, len(names) * 0.6)))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalized) - {split}")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log.info("混淆矩阵已保存：%s", path)
    return path


def label_axis(rc: ResolvedConfig) -> list[str]:
    """混淆矩阵坐标轴标签（报告里直接引用）。"""
    return label_names(rc)
