"""对本地视频批量复核 exp.pt 与 GRU 演示模型的六步报告。

这条路径复用网页的模型、过去帧概率平均和完整性规则，但按固定采样率离线
处理文件；它用于定位失败案例，不代表摄像头实时性能或独立测试集准确率。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from handwash.core.config import ResolvedConfig
from handwash.core.labels import Step
from handwash.core.protocol import build_report
from handwash.core.schema import ProtocolReport
from handwash.errors import DataError
from handwash.io.utils import list_videos, write_csv, write_json
from handwash.io.video import iter_frames, probe_video
from handwash.logging import get_logger
from handwash.paths import ensure_dir, resolve_relative
from handwash.pipelines.assess import save_report
from handwash.pipelines.exp_demo import ExpDemoClassifier, file_sha256

__all__ = ["audit_demo_videos"]

log = get_logger(__name__)

_SUMMARY_FIELDS = (
    "video",
    "source_group",
    "variant",
    "sampled_frames",
    "video_duration_s",
    "detected_steps",
    "missing_steps",
    "is_complete",
    "is_in_order",
    "short_step_warnings",
    "total_duration_warning",
    "wash_duration_s",
    "overall_score",
)


def _causal_mean(probabilities: np.ndarray, window: int) -> np.ndarray:
    """与网页一致：每一帧只平均当前及过去最多 ``window`` 帧。"""
    return np.stack(
        [
            probabilities[max(0, index - window + 1) : index + 1].mean(axis=0)
            for index in range(len(probabilities))
        ]
    )


def _sample_video(classifier: ExpDemoClassifier, video: Path) -> tuple[np.ndarray, np.ndarray, int]:
    features: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    count = 0

    def flush() -> None:
        if batch:
            batch_features, batch_probabilities = classifier.extractor.forward_bgr(batch)
            features.append(batch_features)
            probabilities.append(batch_probabilities)
            batch.clear()

    for _frame_index, _time_s, rgb in iter_frames(video, sample_fps=classifier.rc.dataset.prep.fps):
        # io.iter_frames 输出 RGB；Ultralytics 的 numpy 输入遵循 OpenCV BGR 约定。
        batch.append(np.ascontiguousarray(rgb[:, :, ::-1]))
        count += 1
        if len(batch) >= classifier.rc.infer.batch_size:
            flush()
    flush()
    if not count:
        raise DataError(f"视频没有可解码的采样帧：{video}")
    return np.concatenate(features), np.concatenate(probabilities), count


def _report_from_probabilities(
    classifier: ExpDemoClassifier,
    video: Path,
    probabilities: np.ndarray,
    *,
    model_name: str,
) -> ProtocolReport:
    confidence = probabilities.max(axis=1)
    labels = [
        classifier.space.to_label(int(index))
        if float(score) >= classifier.rc.assess.min_confidence
        else Step.UNKNOWN
        for index, score in zip(probabilities.argmax(axis=1), confidence, strict=True)
    ]
    return build_report(
        clip_id=video.stem,
        labels=labels,
        confidences=confidence.tolist(),
        fps=classifier.rc.dataset.prep.fps,
        cfg=classifier.rc.assess,
        model_name=model_name,
        apply_smoothing=False,
    )


def _summary_row(
    video: Path, root: Path, variant: str, frames: int, duration_s: float, report: ProtocolReport
) -> dict[str, object]:
    relative = video.relative_to(root)
    missing = [v.step for v in report.violations if v.kind == "missing" and v.step is not None]
    short_steps = [v for v in report.violations if v.kind == "insufficient_duration" and v.step is not None]
    total_warning = any(v.kind == "insufficient_duration" and v.step is None for v in report.violations)
    return {
        "video": relative.as_posix(),
        "source_group": relative.parent.as_posix(),
        "variant": variant,
        "sampled_frames": frames,
        "video_duration_s": round(duration_s, 2),
        "detected_steps": ";".join(str(s.step.order_index) for s in report.statistics if s.detected),
        "missing_steps": ";".join(str(step.order_index) for step in missing),
        "is_complete": report.is_complete,
        "is_in_order": report.is_in_order,
        "short_step_warnings": len(short_steps),
        "total_duration_warning": total_warning,
        "wash_duration_s": round(report.total_wash_duration_s, 2),
        "overall_score": round(report.overall_score, 4),
    }


def audit_demo_videos(
    rc: ResolvedConfig,
    *,
    folder: str | Path,
    checkpoint: str | Path = "exp.pt",
    temporal_head: str | Path = "exp_temporal_head.pt",
    output_dir: str | Path | None = None,
) -> Path:
    """保存逐段报告、汇总表和复现实验所需的权重/配置指纹。"""
    root = resolve_relative(folder).resolve()
    if not root.is_dir():
        raise DataError(f"本地核验视频目录不存在：{root}")
    videos = list_videos(root)
    if not videos:
        raise DataError(f"本地核验目录里没有视频：{root}")

    exp_path = resolve_relative(checkpoint)
    head_path = resolve_relative(temporal_head)
    if not exp_path.is_file() or not head_path.is_file():
        raise DataError(f"核验需要 exp.pt 与时序头权重：{exp_path} / {head_path}")
    classifier = ExpDemoClassifier(rc, exp_path, temporal_head=head_path)
    target = ensure_dir(
        resolve_relative(output_dir) if output_dir is not None else rc.resolve_out_dir() / "demo_audit"
    )
    rows: list[dict[str, object]] = []

    for video in videos:
        log.info("本地视频核验：%s", video.relative_to(root))
        features, frame_probs, count = _sample_video(classifier, video)
        smoothed_probs = _causal_mean(frame_probs, max(1, rc.assess.smooth_window))
        assert classifier.temporal_head is not None
        with torch.inference_mode():
            logits, _ = classifier.temporal_head(torch.from_numpy(features)[None])
            gru_probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        duration_s = probe_video(video).duration_s
        reports = {
            "frame_smoothed": _report_from_probabilities(
                classifier, video, smoothed_probs, model_name="exp.pt + past-frame probability mean"
            ),
            "gru": _report_from_probabilities(
                classifier, video, gru_probs, model_name="exp.pt + GRU temporal head"
            ),
        }
        relative_stem = video.relative_to(root).with_suffix("")
        for variant, report in reports.items():
            save_report(report, target / relative_stem, stem=variant)
            rows.append(_summary_row(video, root, variant, count, duration_s, report))

    summary = write_csv(target / "summary.csv", rows, fieldnames=_SUMMARY_FIELDS)
    write_json(
        target / "metadata.json",
        {
            "purpose": "local qualitative audit; not an independent accuracy or live-latency estimate",
            "input_root": str(root),
            "video_count": len(videos),
            "sample_fps": rc.dataset.prep.fps,
            "model_input_size": classifier.image_size,
            "class_labels": [step.value for step in classifier.space.labels],
            "config_hash": rc.config_hash,
            "config_sources": list(rc.sources),
            "checkpoint": str(exp_path),
            "checkpoint_sha256": file_sha256(exp_path),
            "temporal_head": str(head_path),
            "temporal_head_sha256": file_sha256(head_path),
            "variants": ["frame_smoothed", "gru"],
        },
    )
    return summary
