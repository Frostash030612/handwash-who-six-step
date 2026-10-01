"""WHO 洗手流程完整性判定 —— 本项目的业务核心。

把"逐帧分类结果"翻译成"这份洗手流程规范吗"，回答选题文档中的研究问题 3、4：
    * 是否漏步（missing）
    * 是否乱序（out_of_order）
    * 动作时间是否明显不足（insufficient_duration）
    * 是否重复步骤（repeated）

设计原则（CONTRIBUTING.md R15）
--------------------------------
1. **纯函数**：输入标签序列 + 时间轴 + 阈值，输出 ``ProtocolReport``，
   不碰文件、不碰模型，因此可以被单元测试完整覆盖。
2. **阈值全部来自 ``AssessConfig``**：本文件里不允许出现任何魔数。
3. **判定与统计分离**：先做时序平滑与分段（``segment`` / ``smooth_labels``），
   再做覆盖/顺序/时长判定。改规则时只动对应函数。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from handwash.core.config import AssessConfig
from handwash.core.labels import CANONICAL_STEPS, NON_WASH_STEPS, STEP_ORDER, STEP_ZH, Step
from handwash.core.schema import ProtocolReport, ProtocolViolation, StepStatistic
from handwash.errors import ProtocolError

__all__ = [
    "Segment",
    "build_report",
    "check_coverage",
    "check_durations",
    "check_order",
    "check_repeats",
    "collapse_repeats",
    "segment",
    "smooth_labels",
]


class Segment:
    """一段连续的同标签动作（半开区间 [start, end)）。"""

    __slots__ = ("end", "label", "mean_confidence", "start")

    def __init__(self, label: Step, start: int, end: int, mean_confidence: float = 1.0) -> None:
        if end <= start:
            raise ProtocolError(f"非法片段：start={start}, end={end}（要求 end > start）")
        self.label = label
        self.start = start
        self.end = end
        self.mean_confidence = mean_confidence

    @property
    def num_frames(self) -> int:
        return self.end - self.start

    def duration(self, fps: float) -> float:
        """本片段时长（秒）。fps 必须显式传入 —— Segment 故意不持有时间信息。"""
        if fps <= 0:
            raise ProtocolError(f"fps 必须为正，实际 {fps}")
        return self.num_frames / fps

    def to_tuple(self) -> tuple[int, int, Step]:
        # 注意：对外暴露的是**闭区间** end-1，与 schema.ClipPrediction.segments 一致
        return (self.start, self.end - 1, self.label)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Segment({self.label.value}, {self.start}:{self.end})"


# ============================================================================
# 1) 时序处理
# ============================================================================
def smooth_labels(
    labels: Sequence[Step],
    confidences: Sequence[float] | None = None,
    *,
    window: int = 9,
    min_confidence: float = 0.0,
) -> tuple[list[Step], list[float]]:
    """多数投票滑动窗口平滑：消除"单帧跳变"造成的假分段。

    规则细节（必须写清楚，否则两个人实现出来的结果会对不上）：
        * 窗口为奇数且以当前帧为中心；边界处窗口自动收缩（不做 padding 造票）。
        * 平票时：优先保持"上一帧结果"（时间连贯性），否则取置信度之和更高的类别。
        * 置信度低于 ``min_confidence`` 的帧**不参与投票**（视为不可信观测）。
          若整个窗口都不可信，则沿用上一帧标签（首帧则取该帧原标签）。

    Returns
    -------
    (smoothed_labels, smoothed_confidences)
        平滑后的标签序列，以及每帧对应多数票类别的平均置信度。
    """
    if window < 1:
        raise ProtocolError(f"smooth_window 必须 >= 1，实际 {window}")
    if window % 2 == 0:
        window += 1  # 自动修正为奇数，而不是报错打断实验
    labels = list(labels)
    n = len(labels)
    if n == 0:
        return [], []
    confs = [1.0] * n if confidences is None else list(confidences)
    if len(confs) != n:
        raise ProtocolError(f"confidences 长度 {len(confs)} 与 labels 长度 {n} 不一致")

    half = window // 2
    out_labels: list[Step] = []
    out_confs: list[float] = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        votes: Counter[Step] = Counter()
        conf_sum: dict[Step, float] = {}
        for j in range(lo, hi):
            if confs[j] < min_confidence:
                continue
            votes[labels[j]] += 1
            conf_sum[labels[j]] = conf_sum.get(labels[j], 0.0) + confs[j]

        if not votes:
            # 窗口内没有可信观测 -> 沿用上一帧，保持时间连贯
            prev = out_labels[-1] if out_labels else labels[i]
            out_labels.append(prev)
            out_confs.append(confs[i])
            continue

        best_count = max(votes.values())
        tied = [lab for lab, c in votes.items() if c == best_count]
        if len(tied) == 1:
            chosen = tied[0]
        elif out_labels and out_labels[-1] in tied:
            chosen = out_labels[-1]
        else:
            chosen = max(tied, key=lambda lab: (conf_sum.get(lab, 0.0), -STEP_ORDER.index(lab) if lab in STEP_ORDER else 0))
        out_labels.append(chosen)
        out_confs.append(conf_sum[chosen] / max(votes[chosen], 1))

    return out_labels, out_confs


def segment(
    labels: Sequence[Step],
    confidences: Sequence[float] | None = None,
    *,
    fps: float = 1.0,
    min_segment_frames: int = 1,
    min_segment_s: float = 0.0,
) -> list[Segment]:
    """把标签序列切成连续同标签片段，并过滤掉过短的片段。

    过滤策略：短片段**不是直接丢弃**，而是并入相邻的较长片段（`_merge_short`），
    因为直接删除会在时间轴上留洞，导致时长统计偏差。
    """
    if fps <= 0:
        raise ProtocolError(f"fps 必须为正，实际 {fps}")
    labels = list(labels)
    if not labels:
        return []
    confs = [1.0] * len(labels) if confidences is None else list(confidences)

    raw: list[Segment] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            mean_conf = float(np.mean(confs[start:i]))
            raw.append(Segment(labels[start], start, i, mean_conf))
            start = i

    kept = [s for s in raw if s.num_frames >= min_segment_frames and s.duration(fps) >= min_segment_s]
    if not kept:
        # 全都太短：退回"整段一个主标签"，保证后续统计不出现空结果
        counts = Counter(labels)
        dominant = max(counts, key=lambda lab: (counts[lab], -_order_key(lab)))
        return [Segment(dominant, 0, len(labels), float(np.mean(confs)))]

    return _absorb_short_segments(raw, kept, labels, fps, min_segment_frames, min_segment_s)


def _order_key(step: Step) -> int:
    return STEP_ORDER.index(step) if step in STEP_ORDER else len(STEP_ORDER)


def _absorb_short_segments(
    raw: list[Segment],
    kept: list[Segment],
    labels: list[Step],
    fps: float,
    min_frames: int,
    min_seconds: float,
) -> list[Segment]:
    """把被过滤掉的短片段并入其前一/后一有效片段，保持时间轴连续无缝。"""
    kept_keys = {(s.start, s.end) for s in kept}
    result: list[Segment] = [Segment(s.label, s.start, s.end, s.mean_confidence) for s in kept]

    for seg in raw:
        if (seg.start, seg.end) in kept_keys:
            continue
        # 找时间上最近的有效片段（先看前一个，再看后一个）
        prev = next((s for s in reversed(result) if s.end <= seg.start), None)
        nxt = next((s for s in result if s.start >= seg.end), None)
        if prev is not None and nxt is not None:
            gap_prev = seg.start - prev.end
            gap_next = nxt.start - seg.end
            target = prev if gap_prev <= gap_next else nxt
        else:
            target = prev or nxt
        if target is None:  # pragma: no cover - 理论上不可能
            continue
        target.start = min(target.start, seg.start)
        target.end = max(target.end, seg.end)

    result.sort(key=lambda s: s.start)
    # 合并后可能出现相邻同标签片段，重新压平一次
    return _flatten(result)


def _flatten(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    out: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        last = out[-1]
        if seg.label == last.label:
            merged_conf = (
                last.mean_confidence * last.num_frames + seg.mean_confidence * seg.num_frames
            ) / (last.num_frames + seg.num_frames)
            last.end = max(last.end, seg.end)
            last.mean_confidence = merged_conf
        else:
            out.append(seg)
    return out


def collapse_repeats(labels: Sequence[Step]) -> list[Step]:
    """去掉连续重复，得到"动作序列"（用于顺序判定）。

    例：``[1,1,2,1,1,3]`` -> ``[1,2,1,3]``，其中第二次出现的 1 属于"重复步骤"。
    """
    out: list[Step] = []
    for lab in labels:
        if not out or out[-1] != lab:
            out.append(lab)
    return out


# ============================================================================
# 2) 三类判定
# ============================================================================
def check_coverage(
    detected_steps: Iterable[Step],
    *,
    required: Sequence[Step] = CANONICAL_STEPS,
    tolerance: int = 0,
) -> tuple[bool, list[Step]]:
    """覆盖判定：返回 ``(是否完整, 缺失步骤列表)``。"""
    present = set(detected_steps)
    missing = [s for s in required if s not in present]
    return len(missing) <= tolerance, missing


def check_order(
    sequence: Sequence[Step],
) -> list[tuple[Step, Step]]:
    """顺序判定：返回所有**逆序对** (前, 后)，即后出现的步骤序号小于前面的。

    只比较 WHO 六步（忽略 faucet/other/unknown），且按"动作序列"比较
    （调用前应先 ``collapse_repeats``）。

    注意：真实洗手可能出现 1→2→1（回头补做），本项目把它视为乱序并在报告里
    单独说明，而不是静默忽略——因为"顺序异常"本身就是作业要检出的目标。
    """
    who_seq: list[Step] = [s for s in sequence if s in STEP_ORDER]
    inversions: list[tuple[Step, Step]] = []
    for i in range(len(who_seq)):
        for j in range(i + 1, len(who_seq)):
            if STEP_ORDER.index(who_seq[j]) < STEP_ORDER.index(who_seq[i]):
                inversions.append((who_seq[i], who_seq[j]))
    return inversions


def check_durations(
    statistics: Mapping[Step, float],
    *,
    total_wash_s: float,
    cfg: AssessConfig,
    num_expected_steps: int = len(CANONICAL_STEPS),
) -> list[tuple[Step, str]]:
    """时长判定：返回 ``(步骤, 原因说明)`` 列表，表示该步时间明显不足。

    判定口径（必须写清楚，否则组员会争论"到底算不算够"）：
        * ``duration_check == "seconds"``：要求该步时长 >= ``cfg.min_step_duration_s``。
        * ``duration_check == "ratio"``  ：要求该步时长 >=
          ``cfg.step_duration_ratio`` × (总搓洗时间 / 应做步骤数)。
          即"相对平均份额的最低比例"，比硬性秒数更公平（WHO 未规定均分）。
        * ``duration_check == "none"``   ：不判定。
    """
    problems: list[tuple[Step, str]] = []
    if cfg.duration_check == "none" or num_expected_steps <= 0:
        return problems
    fair_share = total_wash_s / num_expected_steps if num_expected_steps else 0.0
    threshold_ratio = fair_share * cfg.step_duration_ratio

    for step in CANONICAL_STEPS:
        duration = float(statistics.get(step, 0.0))
        if cfg.duration_check == "seconds":
            if duration < cfg.min_step_duration_s:
                problems.append(
                    (step, f"仅 {duration:.1f}s，低于最低要求 {cfg.min_step_duration_s:.1f}s")
                )
        else:
            if duration < threshold_ratio:
                problems.append(
                    (
                        step,
                        f"仅 {duration:.1f}s，低于平均份额 {fair_share:.1f}s 的 "
                        f"{cfg.step_duration_ratio:.0%}（阈值 {threshold_ratio:.1f}s）",
                    )
                )
    return problems


def check_repeats(sequence: Sequence[Step]) -> list[tuple[Step, int]]:
    """重复判定：返回 ``(步骤, 出现次数)`` 列表（按 WHO 六步统计）。"""
    counts = Counter(s for s in sequence if s in STEP_ORDER)
    return [(step, n) for step, n in counts.items() if n > 1]


# ============================================================================
# 3) 汇总
# ============================================================================
def build_report(
    *,
    clip_id: str,
    labels: Sequence[Step],
    confidences: Sequence[float] | None = None,
    fps: float,
    cfg: AssessConfig,
    model_name: str = "unknown",
    apply_smoothing: bool = True,
) -> ProtocolReport:
    """主入口：标签序列 -> ``ProtocolReport``。

    Parameters
    ----------
    labels:
        逐帧规范标签（时间顺序）。
    confidences:
        逐帧置信度；None 表示全部按 1.0 处理（例如人工标注的参考序列）。
    fps:
        抽帧后的有效帧率（**不是**原始视频 fps；用抽帧后的序列就传抽帧 fps）。
    cfg:
        判定阈值（来自 ``AssessConfig``）。
    """
    if fps <= 0:
        raise ProtocolError(f"fps 必须为正，实际 {fps}")
    labels = list(labels)
    if not labels:
        raise ProtocolError(f"视频 {clip_id} 的标签序列为空，无法评估")
    if confidences is not None and len(confidences) != len(labels):
        raise ProtocolError(
            f"confidences 长度 {len(confidences)} 与 labels 长度 {len(labels)} 不一致"
        )

    if apply_smoothing:
        smoothed, smoothed_conf = smooth_labels(
            labels, confidences, window=cfg.smooth_window, min_confidence=cfg.min_confidence
        )
    else:
        smoothed = labels
        smoothed_conf = [1.0] * len(labels) if confidences is None else list(confidences)

    segments = segment(
        smoothed,
        smoothed_conf,
        fps=fps,
        min_segment_frames=cfg.min_segment_frames,
        min_segment_s=cfg.min_segment_s,
    )

    # --- 逐步骤统计 -------------------------------------------------------
    duration_by_step: dict[Step, float] = dict.fromkeys(STEP_ORDER, 0.0)
    frames_by_step: dict[Step, int] = dict.fromkeys(STEP_ORDER, 0)
    conf_by_step: dict[Step, list[float]] = {s: [] for s in STEP_ORDER}
    for seg, conf in zip(segments, _segment_mean_confs(segments), strict=True):
        if seg.label in STEP_ORDER:
            duration_by_step[seg.label] += seg.duration(fps)
            frames_by_step[seg.label] += seg.num_frames
            conf_by_step[seg.label].extend([conf] * seg.num_frames)

    total_wash_s = float(sum(duration_by_step.values()))
    statistics: list[StepStatistic] = []
    for step in CANONICAL_STEPS:
        duration = duration_by_step[step]
        statistics.append(
            StepStatistic(
                step=step,
                detected=duration >= cfg.min_segment_s,
                duration_s=duration,
                ratio=(duration / total_wash_s) if total_wash_s > 0 else 0.0,
                num_frames=frames_by_step[step],
                mean_confidence=(
                    float(np.mean(conf_by_step[step])) if conf_by_step[step] else 0.0
                ),
            )
        )

    # --- 三类判定 ---------------------------------------------------------
    violations: list[ProtocolViolation] = []
    detected_steps = [st.step for st in statistics if st.detected]

    is_complete, missing = check_coverage(detected_steps, tolerance=cfg.missing_tolerance)
    for step in missing:
        violations.append(
            ProtocolViolation(
                kind="missing",
                step=step,
                detail=f"未检出第 {step.order_index} 步（{STEP_ZH[step]}）",
                severity="error",
            )
        )

    action_sequence = collapse_repeats([s.label for s in segments])
    inversions = check_order(action_sequence) if cfg.order_check else []
    for first, second in inversions:
        violations.append(
            ProtocolViolation(
                kind="out_of_order",
                step=second,
                detail=(
                    f"顺序异常：第 {second.order_index} 步（{STEP_ZH[second]}）出现在 "
                    f"第 {first.order_index} 步（{STEP_ZH[first]}）之前"
                ),
                severity="warning",
            )
        )

    for step, reason in check_durations(
        duration_by_step, total_wash_s=total_wash_s, cfg=cfg
    ):
        violations.append(
            ProtocolViolation(
                kind="insufficient_duration",
                step=step,
                detail=f"第 {step.order_index} 步（{STEP_ZH[step]}）时长不足：{reason}",
                severity="warning",
            )
        )

    if not cfg.allow_repeats:
        for step, count in check_repeats(action_sequence):
            violations.append(
                ProtocolViolation(
                    kind="repeated",
                    step=step,
                    detail=f"第 {step.order_index} 步（{STEP_ZH[step]}）重复出现 {count} 次",
                    severity="info",
                )
            )

    if cfg.require_faucet_events:
        observed = set(smoothed)
        for event in (Step.FAUCET_ON, Step.FAUCET_OFF):
            if event not in observed:
                violations.append(
                    ProtocolViolation(
                        kind="missing_faucet_event",
                        step=event,
                        detail=f"未检出水龙头事件：{STEP_ZH[event]}",
                        severity="warning",
                    )
                )

    if total_wash_s < cfg.min_total_duration_s:
        violations.append(
            ProtocolViolation(
                kind="insufficient_duration",
                step=None,
                detail=(
                    f"整套搓洗总时长 {total_wash_s:.1f}s，低于 WHO 建议的 "
                    f"{cfg.min_total_duration_s:.0f}s 下限"
                ),
                severity="error",
            )
        )

    total_frames = len(smoothed)
    report = ProtocolReport(
        clip_id=clip_id,
        is_complete=is_complete,
        is_in_order=not inversions,
        step_sequence=tuple(s.label for s in segments),
        statistics=tuple(statistics),
        violations=tuple(violations),
        total_wash_duration_s=total_wash_s,
        total_duration_s=total_frames / fps,
        overall_score=_overall_score(
            statistics=statistics,
            is_complete=is_complete,
            is_in_order=not inversions,
            total_wash_s=total_wash_s,
            cfg=cfg,
        ),
        model_name=model_name,
        notes=tuple(_notes(detected_steps, action_sequence)),
    )
    return report


def _segment_mean_confs(segments: Sequence[Segment]) -> list[float]:
    return [float(s.mean_confidence) for s in segments]


def _overall_score(
    *,
    statistics: Sequence[StepStatistic],
    is_complete: bool,
    is_in_order: bool,
    total_wash_s: float,
    cfg: AssessConfig,
) -> float:
    """综合得分 = 0.5×覆盖率 + 0.3×顺序正确 + 0.2×时长充足率。

    权重固定在函数内（属于"评分口径"，写在 CHANGELOG 与 docs/PROTOCOL.md），
    不放进 config：避免各组员用不同权重得到不可比的分数。
    """
    covered = sum(1 for st in statistics if st.detected) / max(len(CANONICAL_STEPS), 1)
    duration_ratio = min(1.0, total_wash_s / max(cfg.reference_total_duration_s, 1e-6))
    score = 0.5 * covered + 0.3 * (1.0 if is_in_order else 0.0) + 0.2 * duration_ratio
    if not is_complete:
        score = min(score, 1.0)
    return round(float(score), 4)


def _notes(detected_steps: Sequence[Step], action_sequence: Sequence[Step]) -> list[str]:
    """人类可读的补充说明，供报告与演示界面直接展示。"""
    notes: list[str] = []
    non_wash_seen = [s for s in action_sequence if s in NON_WASH_STEPS]
    if non_wash_seen:
        notes.append(
            "流程中检出非搓洗动作：" + "、".join(STEP_ZH.get(s, s.value) for s in dict.fromkeys(non_wash_seen))
        )
    if not detected_steps:
        notes.append("未检出任何有效的 WHO 六步动作，请确认机位与抽帧参数是否合适。")
    return notes
