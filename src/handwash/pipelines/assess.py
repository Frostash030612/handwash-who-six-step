"""完整性评估流程：逐帧预测 -> WHO 六步报告。

这是选题文档里"系统最终输出的东西"：
    是否完成六步 / 顺序是否正确 / 每步做了多久 / 哪一步明显不足 / 综合得分。

本模块只做"编排"：
    推理（infer）-> 规则判定（core.protocol）-> 落盘（JSON / Markdown）
判定逻辑本身**一行都不在这里**，这样规则变更只改 core/protocol.py 一处。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from handwash.core.config import ResolvedConfig
from handwash.core.labels import STEP_ZH, Step
from handwash.core.protocol import build_report
from handwash.core.schema import ProtocolReport
from handwash.io.utils import list_videos, write_json, write_text
from handwash.logging import get_logger
from handwash.paths import ensure_dir
from handwash.pipelines.common import resolve_device
from handwash.pipelines.infer import (
    InferenceOutput,
    extract_clip_from_video,
    predict_clip,
    save_predictions,
)

__all__ = ["assess_video", "assess_videos", "report_to_markdown", "save_report"]

log = get_logger(__name__)

_SEVERITY_ZH = {"info": "提示", "warning": "警告", "error": "错误"}
_KIND_ZH = {
    "missing": "漏步",
    "out_of_order": "顺序异常",
    "insufficient_duration": "时长不足",
    "repeated": "步骤重复",
    "extra_activity": "额外动作",
    "missing_faucet_event": "水龙头事件缺失",
}


def assess_clip(
    rc: ResolvedConfig,
    model: torch.nn.Module,
    clip,
    *,
    device: torch.device | None = None,
) -> tuple[ProtocolReport, InferenceOutput]:
    """对一段已解码的 ``Clip`` 做推理 + 判定。"""
    # Protocol assessment owns its smoothing parameters; inference smoothing is
    # disabled here so the same predictions are never smoothed twice.
    output = predict_clip(
        rc,
        model,
        clip,
        device=device or resolve_device(rc.runtime.device),
        apply_smoothing=False,
    )
    labels = output.labels()
    report = build_report(
        clip_id=output.clip_id,
        labels=labels,
        confidences=output.confidences(),
        fps=output.fps,
        cfg=rc.assess,
        model_name=output.prediction.model_name,
    )
    return report, output


def assess_video(
    rc: ResolvedConfig,
    model: torch.nn.Module,
    video_path: str | Path,
    *,
    device: torch.device | None = None,
    save_dir: str | Path | None = None,
) -> ProtocolReport:
    """对单个视频文件做完整性评估（最常用的入口，也是 demo 的入口）。"""
    dev = device or resolve_device(rc.runtime.device)
    clip = extract_clip_from_video(video_path, sample_fps=rc.dataset.prep.fps)
    report, output = assess_clip(rc, model, clip, device=dev)

    target = Path(save_dir) if save_dir else ensure_dir(rc.resolve_out_dir() / "assess")
    save_report(report, target, stem=Path(video_path).stem)
    if rc.infer.save_frame_predictions:
        save_predictions(output, target / f"{Path(video_path).stem}_frames.jsonl")
    return report


def assess_videos(
    rc: ResolvedConfig,
    model: torch.nn.Module,
    paths: Sequence[str | Path],
    *,
    device: torch.device | None = None,
    save_dir: str | Path | None = None,
) -> list[ProtocolReport]:
    """批量评估（例如组员自采的 4—6 段视频全部跑一遍，用于报告里的对比表）。"""
    dev = device or resolve_device(rc.runtime.device)
    target = Path(save_dir) if save_dir else ensure_dir(rc.resolve_out_dir() / "assess")
    reports: list[ProtocolReport] = []
    for path in paths:
        log.info("评估：%s", path)
        try:
            clip = extract_clip_from_video(path, sample_fps=rc.dataset.prep.fps)
            report, output = assess_clip(rc, model, clip, device=dev)
            reports.append(report)
            save_report(report, target, stem=Path(path).stem)
            if rc.infer.save_frame_predictions:
                save_predictions(output, target / f"{Path(path).stem}_frames.jsonl")
        except Exception as exc:
            log.error("视频处理失败，已跳过：%s（%s）", path, exc)

    if reports:
        write_json(target / "assess_summary.json", [r.to_dict() for r in reports])
        log.info("批量评估完成：%d/%d 段成功", len(reports), len(paths))
    return reports


def assess_folder(
    rc: ResolvedConfig,
    model: torch.nn.Module,
    folder: str | Path,
    *,
    device: torch.device | None = None,
) -> list[ProtocolReport]:
    """评估一个目录下的全部视频。"""
    videos = list_videos(folder)
    if not videos:
        log.warning("目录下没有视频文件：%s", folder)
    return assess_videos(rc, model, videos, device=device)


# ============================================================================
# 输出（JSON 给程序看，Markdown 给人看）
# ============================================================================
def save_report(report: ProtocolReport, directory: str | Path, *, stem: str | None = None) -> tuple[Path, Path]:
    """同时写 ``<stem>.json`` 与 ``<stem>.md``，返回两个路径。"""
    target = ensure_dir(directory)
    name = stem or report.clip_id
    json_path = write_json(target / f"{name}.json", report.to_dict())
    md_path = target / f"{name}.md"
    write_text(md_path, report_to_markdown(report))
    log.info("报告已保存：%s / %s", json_path.name, md_path.name)
    return json_path, md_path


def report_to_markdown(report: ProtocolReport) -> str:
    """把报告渲染成中文 Markdown（可直接贴进作业文档 / demo 界面）。"""
    lines: list[str] = []
    verdict = "流程完整" if report.is_complete else "存在漏步"
    order = "顺序正确" if report.is_in_order else "顺序异常"
    lines.append(f"# 洗手完整性评估报告：{report.clip_id}")
    lines.append("")
    lines.append(f"- 结论：**{verdict}**，**{order}**")
    lines.append(f"- 综合得分：**{report.overall_score:.2f}** / 1.00")
    lines.append(f"- 搓洗总时长：{report.total_wash_duration_s:.1f} s")
    lines.append(f"- 视频总时长（抽帧后）：{report.total_duration_s:.1f} s")
    lines.append(f"- 模型：`{report.model_name}`")
    lines.append("")

    lines.append("## 各步骤明细")
    lines.append("")
    lines.append("| 步骤 | 动作（中文） | 是否检出 | 时长(s) | 占比 | 平均置信度 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for stat in report.statistics:
        lines.append(
            f"| 第 {stat.step_no} 步 | {STEP_ZH.get(stat.step, stat.step.value)} | "
            f"{'是' if stat.detected else '**否**'} | {stat.duration_s:.1f} | "
            f"{stat.ratio:.1%} | {stat.mean_confidence:.2f} |"
        )
    lines.append("")

    if report.timeline:
        # 与顺序/重复判定同一口径：只看六步，中途的 other/unknown 停顿不拆开同一步。
        order: list[Step] = []
        for seg in report.timeline:
            if seg.step.order_index and (not order or order[-1] is not seg.step):
                order.append(seg.step)
        lines.append("## 识别时间轴")
        lines.append("")
        lines.append("- 识别顺序：" + (" → ".join(f"第 {s.order_index} 步" for s in order) or "未识别到六步动作"))
        lines.append("")
        lines.append("| 时间(s) | 识别结果 | 平均置信度 |")
        lines.append("| --- | --- | --- |")
        for seg in report.timeline:
            name = STEP_ZH.get(seg.step, seg.step.value)
            label = f"第 {seg.step.order_index} 步 · {name}" if seg.step.order_index else name
            lines.append(f"| {seg.start_s:.1f}–{seg.end_s:.1f} | {label} | {seg.mean_confidence:.2f} |")
        lines.append("")

    lines.append("## 发现的问题")
    lines.append("")
    if not report.violations:
        lines.append("未发现漏步、乱序或时长不足。")
    else:
        lines.append("| 类型 | 严重度 | 说明 |")
        lines.append("| --- | --- | --- |")
        for violation in report.violations:
            lines.append(
                f"| {_KIND_ZH.get(violation.kind, violation.kind)} | "
                f"{_SEVERITY_ZH.get(violation.severity, violation.severity)} | {violation.detail} |"
            )
    lines.append("")

    if report.notes:
        lines.append("## 补充说明")
        lines.append("")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## 判定口径")
    lines.append("")
    lines.append(
        "阈值来自配置文件的 `assess` 段（见 docs/PROTOCOL.md）。"
        "总时长参考 WHO 建议的 40—60 秒；单步时长按“总搓洗时间 ÷ 应做步骤数”的比例判定，"
        "因为 WHO 并未要求六步平均分配时间。"
    )
    return "\n".join(lines) + "\n"


def detected_step_sequence(report: ProtocolReport) -> list[Step]:
    """便捷函数：报告里检出过的 WHO 六步（按时间顺序去重）。"""
    seen: list[Step] = []
    for step in report.step_sequence:
        if step.order_index and step not in seen:
            seen.append(step)
    return seen
