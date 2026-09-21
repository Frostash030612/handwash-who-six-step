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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from handwash.core.config import ResolvedConfig
from handwash.core.labels import get_label_space
from handwash.core.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_report,
    weighted_f1,
)
from handwash.core.schema import EvalResult
from handwash.core.seeding import seed_everything
from handwash.errors import EvaluationError
from handwash.io.utils import append_jsonl, write_csv, write_json
from handwash.logging import get_logger
from handwash.models.checkpoint import load_checkpoint
from handwash.paths import BEST_CHECKPOINT_NAME, checkpoint_path, ensure_dir
from handwash.pipelines.common import (
    build_manifest_loader,
    label_names,
    resolve_device,
    resolve_split_records,
)

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
    model = build_model(rc).to(device)
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    if missing:
        log.warning("加载 checkpoint：缺少 %d 个参数（前 3 个：%s）", len(missing), missing[:3])
    if unexpected:
        log.warning("加载 checkpoint：多余 %d 个参数（前 3 个：%s）", len(unexpected), unexpected[:3])
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
) -> tuple[EvalResult, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """评估单个 split，返回 ``(EvalResult, y_true, y_pred, 逐帧明细)``。"""
    records = resolve_split_records(rc, split=split)
    for_clip = rc.train.mode in ("clip", "hybrid")
    loader = build_manifest_loader(rc, records, split=split, shuffle=False, for_clip_mode=for_clip)
    space = get_label_space(rc.label_space)
    num_classes = len(space)

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    details: list[dict[str, Any]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model.logits(images)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().numpy()  # (B, T)
        targets = batch["label_index"].numpy()  # (B, T)
        clip_ids = batch["clip_id"]
        frame_indices = batch.get("frame_index")
        if frame_indices is not None:
            frame_indices = frame_indices.numpy()  # (B, T)

        for b in range(targets.shape[0]):
            for t in range(targets.shape[1]):
                true_idx = int(targets[b, t])
                pred_idx = int(preds[b, t])
                y_true_parts.append(np.asarray([true_idx]))
                y_pred_parts.append(np.asarray([pred_idx]))
                details.append(
                    {
                        "clip_id": clip_ids[b],
                        "frame_index": int(frame_indices[b, t]) if frame_indices is not None else t,
                        "true": space.to_label(true_idx).value,
                        "pred": space.to_label(pred_idx).value,
                        "confidence": round(float(probs[b, t, pred_idx].item()), 6),
                    }
                )

    if not details:
        raise EvaluationError(f"split={split} 没有可评估的样本")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    names = [s.value for s in space.labels]
    result = EvalResult(
        split=split,
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

    model, payload = _load_model_from_checkpoint(rc, ckpt, device=device)
    wanted = list(splits or rc.eval.splits)

    bundle = EvalBundle()
    for split in wanted:
        result, y_true, y_pred, details = evaluate_single_split(
            rc, model, split=split, device=device, checkpoint_name=str(ckpt)
        )
        bundle.add(result)
        write_json(eval_dir / f"eval_{split}.json", result.to_dict())

        names = list(result.labels)
        write_csv(
            eval_dir / f"per_class_{split}.csv",
            [
                {"label": name, **{k: v for k, v in stats.items()}}
                for name, stats in result.per_class.items()
            ],
            fieldnames=["label", "precision", "recall", "f1", "support", "predicted", "correct"],
        )
        if rc.eval.save_confusion_matrix:
            _save_confusion_figure(
                np.asarray(result.confusion_matrix), names, eval_dir / f"confusion_matrix_{split}.png", split
            )
        if rc.eval.save_predictions:
            append_jsonl(eval_dir / f"predictions_{split}.jsonl", details)

    write_json(eval_dir / "eval_summary.json", bundle.to_dict())
    log.info("评估完成：\n%s", bundle.summary_table())
    return bundle


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
