"""WHO 完整性判定契约测试（src/handwash/core/protocol.py）。

这是本项目的业务核心：把逐帧分类结果翻译成"这份洗手流程规范吗"。测试的重点不是
覆盖率数字，而是**两个人照着同一份文档实现必须得到同一结论** —— 所以每个阈值
口径（平滑窗口、最短片段、时长比例、顺序判定）都单独钉一个用例。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import pairwise

import pytest

from handwash.core.config import AssessConfig
from handwash.core.labels import CANONICAL_STEPS, STEP_ZH, Step
from handwash.core.protocol import (
    Segment,
    build_report,
    check_coverage,
    check_durations,
    check_order,
    check_repeats,
    collapse_repeats,
    segment,
    smooth_labels,
)
from handwash.errors import ProtocolError

pytestmark = pytest.mark.unit

#: 测试统一帧率：1 帧 = 0.2 秒。
FPS = 5.0

#: 每步帧数（=10 秒），远大于默认 min_segment_frames=5 与 min_segment_s=1.0。
STEP_FRAMES = 50
STEP_SECONDS = STEP_FRAMES / FPS

A, B, C = Step.STEP_1, Step.STEP_2, Step.STEP_3
D, E, F = Step.STEP_4, Step.STEP_5, Step.STEP_6
ALL_STEPS = (A, B, C, D, E, F)


def seq(*specs: tuple[Step, int]) -> list[Step]:
    """把 ``(步骤, 帧数)`` 展开成逐帧标签序列。

    为什么要有这个 helper：``build_report`` 的输入是逐帧标签，直接手写 300 个
    元素会让"到底哪一步短了"这种测试意图完全看不出来。
    """
    return [step for step, frames in specs for _ in range(frames)]


def full_sequence(frames: int = STEP_FRAMES) -> list[Step]:
    """一段完全规范的六步序列（顺序正确、每步时长充足）。"""
    return seq(*[(step, frames) for step in ALL_STEPS])


def kinds(report) -> list[str]:
    """提取违规类型列表，便于用 ``in`` / ``not in`` 做断言。"""
    return [violation.kind for violation in report.violations]


def steps_of_kind(report, kind: str) -> list[Step | None]:
    """提取某一类违规涉及的步骤。"""
    return [violation.step for violation in report.violations if violation.kind == kind]


# --- Segment ----------------------------------------------------------------
def test_segment_is_half_open_and_exposes_closed_tuple() -> None:
    """内部用半开区间 [start, end) 便于切片，对外用闭区间便于和 ClipPrediction 对齐。

    这两者混用是分段代码最常见的 off-by-one 来源，因此两边都要锁死。
    """
    seg = Segment(A, 2, 7, 0.8)
    assert (seg.start, seg.end) == (2, 7)
    assert seg.num_frames == 5
    assert seg.to_tuple() == (2, 6, A)
    assert seg.mean_confidence == pytest.approx(0.8)


def test_segment_duration_requires_positive_fps() -> None:
    """Segment 故意不持有时间信息，fps 必须显式传入；非法 fps 不允许算出一个负时长。"""
    seg = Segment(A, 0, 10)
    assert seg.duration(FPS) == pytest.approx(2.0)
    with pytest.raises(ProtocolError, match="fps"):
        seg.duration(0.0)
    with pytest.raises(ProtocolError, match="fps"):
        seg.duration(-1.0)


def test_segment_rejects_empty_or_reversed_range() -> None:
    """空片段会让后续统计出现 0 帧的"步骤"，属于实现错误，必须当场报错。"""
    for start, end in ((5, 5), (5, 4)):
        with pytest.raises(ProtocolError, match="非法片段"):
            Segment(A, start, end)


# --- smooth_labels ----------------------------------------------------------
def test_smooth_labels_removes_isolated_single_frame_flip() -> None:
    """单帧跳变是模型抖动而非真的换动作，不平滑会凭空多出两步（漏步/乱序误报）。"""
    noisy = [A] * 5 + [B] + [A] * 5
    smoothed, _ = smooth_labels(noisy, window=5)
    assert smoothed == [A] * 11


def test_smooth_labels_keeps_genuine_long_change() -> None:
    """真正的动作切换不能被平滑掉，否则会漏判整步。"""
    smoothed, _ = smooth_labels([A] * 6 + [B] * 6, window=5)
    assert smoothed == [A] * 6 + [B] * 6


def test_smooth_labels_rounds_even_window_up_instead_of_failing() -> None:
    """偶数窗口自动修正为奇数：实验中途改参数不应该直接打断训练/评估流程。"""
    smoothed, _ = smooth_labels([A, A, B], window=4)
    assert len(smoothed) == 3
    assert smoothed == [A, A, A]


@pytest.mark.parametrize("window", [0, -1, -9])
def test_smooth_labels_rejects_non_positive_window(window: int) -> None:
    """窗口 <= 0 没有数学意义，必须报错而不是返回原序列。"""
    with pytest.raises(ProtocolError, match="smooth_window"):
        smooth_labels([A, B], window=window)


def test_smooth_labels_returns_empty_for_empty_input() -> None:
    """空输入必须在平滑阶段就返回空，而不是抛异常打断上层流水线。"""
    assert smooth_labels([], window=5) == ([], [])


def test_smooth_labels_low_confidence_frames_do_not_vote() -> None:
    """低于 min_confidence 的帧视为不可信观测：它不能把邻居的标签改坏。"""
    labels = [A] * 5 + [B] + [A] * 5
    confidences = [0.9] * 5 + [0.05] + [0.9] * 5
    smoothed, _ = smooth_labels(labels, confidences, window=5, min_confidence=0.4)
    assert smoothed == [A] * 11


def test_smooth_labels_holds_previous_label_when_whole_window_untrusted() -> None:
    """整个窗口都不可信时沿用上一帧，保证时间连贯而不是随机跳。"""
    smoothed, confidences = smooth_labels([A, B, C], [0.0, 0.0, 0.0], window=3, min_confidence=0.5)
    assert smoothed == [A, A, A]
    assert confidences == [0.0, 0.0, 0.0]


def test_smooth_labels_returns_mean_confidence_of_winning_votes() -> None:
    """输出置信度是"胜出类别的平均置信度"，报告里的可信度指标依赖它。"""
    labels = [A, A, A]
    smoothed, confidences = smooth_labels(labels, [0.2, 0.4, 0.6], window=3)
    assert smoothed == labels
    assert confidences[1] == pytest.approx(0.4)


def test_smooth_labels_confidences_length_must_match_labels() -> None:
    """长度不一致说明置信度与帧错位，必须报错而不是截断。"""
    with pytest.raises(ProtocolError, match="confidences 长度"):
        smooth_labels([A, A], [1.0])


def test_smooth_labels_defaults_confidences_to_one() -> None:
    """不提供置信度时（例如人工参考序列）全部按 1.0 处理。"""
    _, confidences = smooth_labels([A, B], window=1)
    assert confidences == [1.0, 1.0]


# --- segment ----------------------------------------------------------------
def test_segment_merges_consecutive_equal_labels() -> None:
    """分段必须按"连续同标签"切，重复的动作序列另外用 collapse_repeats 处理。"""
    segments = segment([A] * 3 + [B] * 4 + [C] * 2, fps=FPS, min_segment_frames=1, min_segment_s=0.0)
    assert [(s.label, s.start, s.end) for s in segments] == [(A, 0, 3), (B, 3, 7), (C, 7, 9)]


def test_segment_absorbs_short_segment_without_leaving_gaps() -> None:
    """短片段必须被并入邻居而不是删除 —— 删掉会在时间轴上留洞，时长统计会偏。"""
    labels = [A] * 10 + [B] * 2 + [A] * 10
    segments = segment(labels, fps=FPS, min_segment_frames=5, min_segment_s=1.0)
    assert segments[0].start == 0
    assert segments[-1].end == len(labels)
    covered = {i for s in segments for i in range(s.start, s.end)}
    assert covered == set(range(len(labels)))


def test_segment_spans_cover_whole_timeline_for_gappy_input() -> None:
    """任意输入下，所有片段必须是首尾相接、无重叠、无空洞的划分。"""
    labels = seq((A, 20), (B, 2), (C, 20), (D, 1), (E, 20))
    segments = segment(labels, fps=FPS, min_segment_frames=5, min_segment_s=1.0)
    assert segments[0].start == 0
    assert segments[-1].end == len(labels)
    for left, right in pairwise(segments):
        assert left.end == right.start
    covered = {i for s in segments for i in range(s.start, s.end)}
    assert covered == set(range(len(labels)))


def test_segment_returns_single_dominant_segment_when_everything_is_short() -> None:
    """全部片段都过短时退回"整段一个主标签"，避免上层拿到空分段后除零。"""
    labels = [A, A, B, C, C]
    segments = segment(labels, fps=1.0, min_segment_frames=10, min_segment_s=10.0)
    assert len(segments) == 1
    assert segments[0].label == A
    assert (segments[0].start, segments[0].end) == (0, len(labels))


def test_segment_respects_min_segment_seconds() -> None:
    """除了帧数下限，还要看真实时长：高抽帧率下 5 帧可能只有 0.05 秒。"""
    labels = [A] * 5 + [B] * 40
    segments = segment(labels, fps=100.0, min_segment_frames=1, min_segment_s=0.5)
    assert [s.label for s in segments] == [B]


def test_segment_empty_input_returns_empty_list() -> None:
    """空序列返回空分段，由调用方决定是否报错（build_report 会报）。"""
    assert segment([], fps=FPS) == []


@pytest.mark.parametrize("fps", [0.0, -5.0])
def test_segment_rejects_non_positive_fps(fps: float) -> None:
    """fps <= 0 会让时长变成负数或除零，必须显式拦住。"""
    with pytest.raises(ProtocolError, match="fps"):
        segment([A, B], fps=fps)


def test_segment_mean_confidence_averages_frames() -> None:
    """片段置信度取帧均值，用于报告里"这一步判得可不可信"。"""
    segments = segment([A, A, A, A], [0.2, 0.4, 0.6, 0.8], fps=FPS, min_segment_frames=1, min_segment_s=0.0)
    assert segments[0].mean_confidence == pytest.approx(0.5)


# --- collapse_repeats -------------------------------------------------------
def test_collapse_repeats_keeps_returns_to_earlier_step() -> None:
    """只压掉**连续**重复；1→2→1 中的第二个 1 是有意义的"回头补做"，不能删。"""
    assert collapse_repeats([1, 1, 2, 1, 1, 3]) == [1, 2, 1, 3]


def test_collapse_repeats_on_steps_and_edge_cases() -> None:
    """Step 序列与边界输入（空、单元素、全同）都必须稳定。"""
    assert collapse_repeats([A, A, B, A, A, C]) == [A, B, A, C]
    assert collapse_repeats([]) == []
    assert collapse_repeats([A]) == [A]
    assert collapse_repeats([A] * 7) == [A]


# --- check_coverage ---------------------------------------------------------
def test_check_coverage_reports_missing_steps_in_who_order() -> None:
    """缺失列表必须按 WHO 顺序给出，报告里才能稳定地逐条列出。"""
    complete, missing = check_coverage([A, B, C, E, F])
    assert complete is False
    assert missing == [D]

    complete_all, missing_none = check_coverage(ALL_STEPS)
    assert complete_all is True
    assert missing_none == []


def test_check_coverage_tolerance_allows_configured_number_of_misses() -> None:
    """missing_tolerance 是"允许漏几步仍算基本完整"的开关，必须真的生效。"""
    assert check_coverage([A, B, C, D], tolerance=0)[0] is False
    assert check_coverage([A, B, C, D], tolerance=2)[0] is True
    assert check_coverage([A, B, C, D], tolerance=2)[1] == [E, F]


def test_check_coverage_ignores_non_wash_activity() -> None:
    """开关水龙头、过渡动作不属于六步，不能拿来凑覆盖率。"""
    complete, missing = check_coverage([Step.FAUCET_ON, Step.WASHING_HANDS, A])
    assert complete is False
    assert len(missing) == 5


# --- check_order ------------------------------------------------------------
def test_check_order_returns_empty_for_strictly_ascending_who_sequence() -> None:
    """规范流程必须零逆序对，否则报告会把标准动作误判为乱序。"""
    assert check_order(list(CANONICAL_STEPS)) == []


def test_check_order_returns_inversion_pairs() -> None:
    """逆序对记录"前面那步、后面那步"，报告要据此解释顺序异常。"""
    assert check_order([A, C, B]) == [(C, B)]
    assert check_order([A, B, F, D, E]) == [(F, D), (F, E)]


def test_check_order_ignores_non_who_labels() -> None:
    """faucet/other/unknown 不参与顺序约束，否则开关水龙头会被判成乱序。"""
    assert check_order([Step.FAUCET_ON, A, Step.OTHER, B, Step.FAUCET_OFF, C]) == []
    assert check_order([]) == []


def test_check_order_detects_return_to_earlier_step() -> None:
    """1→2→1 属于"回头补做"，本项目显式判定为乱序并写进报告，而不是静默忽略。"""
    assert check_order([A, B, A]) == [(B, A)]


# --- check_durations --------------------------------------------------------
def test_check_durations_seconds_mode_uses_absolute_threshold() -> None:
    """seconds 口径：每步必须 >= min_step_duration_s（阈值 3 秒，第 1 步给了 5 秒）。"""
    cfg = AssessConfig(duration_check="seconds", min_step_duration_s=3.0)
    problems = check_durations({A: 5.0}, total_wash_s=60.0, cfg=cfg)
    assert [step for step, _ in problems] == [B, C, D, E, F]
    assert all("低于最低要求" in reason for _, reason in problems)


def test_check_durations_ratio_mode_compares_to_fair_share() -> None:
    """ratio 口径：阈值 = 总搓洗时长 / 应做步数 × step_duration_ratio。

    用 10 秒总时长、6 步、比例 0.4 构造：平均份额 1.667 秒、阈值 0.667 秒，
    因此只有 0.5 秒的那一步会被判为时长不足。阈值必须由这个公式算出来，
    避免测试和实现各写一套"凭感觉"的数字。
    """
    cfg = AssessConfig(duration_check="ratio", step_duration_ratio=0.4)
    durations = {A: 3.0, B: 2.0, C: 2.0, D: 2.0, E: 0.5, F: 0.5}
    total = sum(durations.values())
    fair_share = total / len(ALL_STEPS)
    threshold = fair_share * cfg.step_duration_ratio

    problems = check_durations(durations, total_wash_s=total, cfg=cfg)

    expected = [step for step, value in durations.items() if value < threshold]
    assert expected == [E, F]
    assert [step for step, _ in problems] == expected
    assert all("平均份额" in reason for _, reason in problems)


def test_check_durations_ratio_mode_passes_a_perfectly_even_flow() -> None:
    """六步均分的流程不该被 ratio 口径误判为时长不足。"""
    cfg = AssessConfig(duration_check="ratio", step_duration_ratio=0.4)
    even = check_durations(dict.fromkeys(ALL_STEPS, 10.0), total_wash_s=60.0, cfg=cfg)
    assert even == []


def test_check_durations_none_mode_skips_check_entirely() -> None:
    """duration_check=none 用于"只判漏步"的对照实验，必须真的不做判定。"""
    cfg = AssessConfig(duration_check="none")
    assert check_durations({}, total_wash_s=0.0, cfg=cfg) == []


def test_check_durations_zero_expected_steps_is_noop() -> None:
    """应做步数为 0 时不能除零。"""
    cfg = AssessConfig(duration_check="ratio")
    assert check_durations({A: 0.0}, total_wash_s=60.0, cfg=cfg, num_expected_steps=0) == []


# --- check_repeats ----------------------------------------------------------
def test_check_repeats_counts_only_who_steps_above_one() -> None:
    """重复次数 >= 2 才算重复；辅助动作出现多次不算"重复步骤"。"""
    assert check_repeats([A, A, B, C]) == [(A, 2)]
    assert check_repeats([A, B, C, D, E, F]) == []
    assert check_repeats([Step.FAUCET_ON] * 5 + [A]) == []


# --- build_report -----------------------------------------------------------
def test_build_report_accepts_a_fully_compliant_sequence() -> None:
    """完全规范的流程必须判为完整、有序、无违规，综合得分为 1.0。"""
    report = build_report(clip_id="good", labels=full_sequence(), fps=FPS, cfg=AssessConfig())
    assert report.clip_id == "good"
    assert report.is_complete is True
    assert report.is_in_order is True
    assert "missing" not in kinds(report)
    assert report.violations == ()
    assert report.overall_score == pytest.approx(1.0)
    assert report.total_wash_duration_s == pytest.approx(6 * STEP_SECONDS)


def test_build_report_statistics_match_the_given_timeline() -> None:
    """每一步的帧数/时长/占比必须与输入时间轴一致，这是报告的数字基础。"""
    report = build_report(clip_id="stats", labels=full_sequence(), fps=FPS, cfg=AssessConfig())
    assert [stat.step for stat in report.statistics] == list(CANONICAL_STEPS)
    for stat in report.statistics:
        assert stat.num_frames == STEP_FRAMES
        assert stat.duration_s == pytest.approx(STEP_SECONDS)
        assert stat.detected is True
        assert stat.ratio == pytest.approx(1 / 6, abs=1e-6)
        assert stat.mean_confidence == pytest.approx(1.0)


def test_build_report_flags_the_missing_step_with_its_step_value() -> None:
    """漏步必须精确指出是哪一步 —— 报告最重要的回答之一。"""
    labels = seq(*[(step, STEP_FRAMES) for step in ALL_STEPS if step is not D])
    report = build_report(clip_id="missing4", labels=labels, fps=FPS, cfg=AssessConfig())
    assert report.is_complete is False
    assert steps_of_kind(report, "missing") == [D]
    assert STEP_ZH[D] in report.violations[0].detail


def test_build_report_flags_swapped_pair_as_out_of_order() -> None:
    """顺序异常要能被检出，且 reported step 是"跑到前面去的那一步"。"""
    labels = seq((A, STEP_FRAMES), (C, STEP_FRAMES), (B, STEP_FRAMES), (D, STEP_FRAMES),
                 (E, STEP_FRAMES), (F, STEP_FRAMES))
    report = build_report(clip_id="swapped", labels=labels, fps=FPS, cfg=AssessConfig())
    assert report.is_complete is True
    assert report.is_in_order is False
    assert steps_of_kind(report, "out_of_order") == [B]
    assert report.overall_score < 1.0


def test_build_report_ignores_order_when_order_check_disabled() -> None:
    """order_check=False 用于"只评估覆盖率"的消融实验。"""
    labels = seq((A, STEP_FRAMES), (C, STEP_FRAMES), (B, STEP_FRAMES), (D, STEP_FRAMES),
                 (E, STEP_FRAMES), (F, STEP_FRAMES))
    report = build_report(clip_id="noorder", labels=labels, fps=FPS, cfg=AssessConfig(order_check=False))
    assert report.is_in_order is True
    assert "out_of_order" not in kinds(report)


def test_build_report_flags_insufficient_step_duration() -> None:
    """某一步明显偏短要单独指出：这是"做了但没做到位"的证据。"""
    short_frames = 10  # 2 秒，远低于 60 秒流程的平均份额阈值 4 秒
    labels = seq((A, 100), (B, 100), (C, 100), (D, short_frames), (E, 100), (F, 100))
    report = build_report(clip_id="short", labels=labels, fps=FPS, cfg=AssessConfig())
    assert D in steps_of_kind(report, "insufficient_duration")
    assert report.statistics[3].duration_s == pytest.approx(short_frames / FPS)


def test_build_report_flags_total_duration_below_who_recommendation() -> None:
    """WHO 建议整套流程 40 秒以上；总时长不足时给一条不针对具体步骤的违规。"""
    labels = seq(*[(step, 20) for step in ALL_STEPS])  # 每步 4 秒，共 24 秒
    report = build_report(clip_id="tooshort", labels=labels, fps=FPS, cfg=AssessConfig())
    totals = [v for v in report.violations if v.kind == "insufficient_duration" and v.step is None]
    assert len(totals) == 1
    assert report.total_wash_duration_s == pytest.approx(24.0)


def test_build_report_flags_repeated_step_when_repeats_disallowed() -> None:
    """默认不允许重复步骤：回到第 1 步既算乱序也算重复，报告要同时说明。"""
    labels = full_sequence() + seq((A, STEP_FRAMES))
    report = build_report(clip_id="repeat", labels=labels, fps=FPS, cfg=AssessConfig(allow_repeats=False))
    assert steps_of_kind(report, "repeated") == [A]
    assert A in steps_of_kind(report, "out_of_order")


def test_pause_inside_a_step_is_not_a_repeat() -> None:
    """第 5 步中途停顿 2 秒（other / unknown）再继续，不是"重做第 5 步"。

    真实视频里手会短暂移出画面或停下取洗手液；只有中间插入另一个 WHO 步骤
    才算重复，这与顺序判定忽略 other/unknown 的口径一致。
    """
    for pause in (Step.OTHER, Step.UNKNOWN):
        labels = seq((A, 50), (B, 50), (C, 50), (D, 50), (E, 25), (pause, 10), (E, 25), (F, 50))
        report = build_report(clip_id="pause", labels=labels, fps=FPS, cfg=AssessConfig(), apply_smoothing=False)
        assert "repeated" not in kinds(report)
        assert report.is_in_order is True


def test_same_inversion_is_reported_once_across_pauses() -> None:
    """2 → 停顿 → 2 → 1 只是一次"回到第 1 步"，不应因停顿被报成两条乱序。"""
    labels = seq((A, 50), (B, 25), (Step.OTHER, 10), (B, 25), (A, 50), (C, 50), (D, 50), (E, 50), (F, 50))
    report = build_report(clip_id="dup", labels=labels, fps=FPS, cfg=AssessConfig(), apply_smoothing=False)
    assert steps_of_kind(report, "out_of_order") == [A]
    assert steps_of_kind(report, "repeated") == [A]


def test_report_timeline_matches_segments_used_for_judgement() -> None:
    """页面时间轴直接画 report.timeline，它必须与判定所用的分段一致且首尾相接。"""
    labels = seq((Step.OTHER, 10), (A, 50), (B, 50), (A, 20), (C, 50))
    report = build_report(clip_id="tl", labels=labels, fps=FPS, cfg=AssessConfig(), apply_smoothing=False)
    assert tuple(seg.step for seg in report.timeline) == report.step_sequence
    assert report.timeline[0].start_s == pytest.approx(0.0)
    assert report.timeline[-1].end_s == pytest.approx(len(labels) / FPS)
    for left, right in zip(report.timeline, report.timeline[1:]):
        assert left.end_s == pytest.approx(right.start_s)
    payload = report.to_dict()["timeline"]
    assert payload[1] == {"step": A.value, "step_no": 1, "start_s": 2.0, "end_s": 12.0, "mean_confidence": 1.0}


def test_build_report_allows_repeats_when_configured() -> None:
    """allow_repeats=True 时不得再产生 repeated 违规（阈值全部来自 AssessConfig）。"""
    labels = full_sequence() + seq((A, STEP_FRAMES))
    report = build_report(clip_id="repeat-ok", labels=labels, fps=FPS, cfg=AssessConfig(allow_repeats=True))
    assert "repeated" not in kinds(report)


def test_build_report_ignores_single_frame_noise() -> None:
    """一帧误判不应该凭空多出一次"重复步骤"：平滑与"短片段吸收"共同保证这一点。"""
    labels = full_sequence()
    noisy = labels[:100] + [A] + labels[100:]  # 在第 3 步中间插一帧第 1 步
    report = build_report(clip_id="noisy", labels=noisy, fps=FPS, cfg=AssessConfig())
    assert "repeated" not in kinds(report)
    assert report.is_in_order is True
    assert report.is_complete is True


def test_single_frame_noise_is_also_absorbed_without_smoothing() -> None:
    """即使关掉平滑，单帧噪声也会被 segment 的短片段吸收规则消化掉。

    把这个行为写成测试的原因是：它是刻意的双保险（CONTRIBUTING R15），不是巧合；
    如果有人把 segment 改成"直接丢弃过短片段"，时间轴就会出现空洞，这条会先红。
    """
    labels = full_sequence()
    noisy = labels[:100] + [A] + labels[100:]
    raw = build_report(clip_id="raw", labels=noisy, fps=FPS, cfg=AssessConfig(), apply_smoothing=False)
    smoothed = build_report(clip_id="smooth", labels=noisy, fps=FPS, cfg=AssessConfig())
    assert "repeated" not in kinds(raw)
    assert raw.step_sequence == smoothed.step_sequence
    assert raw.total_wash_duration_s == pytest.approx(smoothed.total_wash_duration_s)


def test_build_report_records_non_wash_activity_in_notes() -> None:
    """报告要能直接展示"这一段里有开关水龙头"，供演示界面使用。"""
    labels = seq((Step.FAUCET_ON, 10)) + full_sequence()
    report = build_report(clip_id="faucet", labels=labels, fps=FPS, cfg=AssessConfig())
    assert any("非搓洗动作" in note for note in report.notes)
    assert report.step_sequence[0] is Step.FAUCET_ON


def test_build_report_records_model_name_for_traceability() -> None:
    """报告必须记住是哪个模型给出的结论，否则无法回溯。"""
    report = build_report(
        clip_id="trace", labels=full_sequence(), fps=FPS, cfg=AssessConfig(), model_name="yolo26n-cls"
    )
    assert report.model_name == "yolo26n-cls"


@pytest.mark.parametrize(
    ("labels", "fps"),
    [([], FPS), (full_sequence(), 0.0), (full_sequence(), -1.0)],
)
def test_build_report_rejects_unevaluable_input(labels: Sequence[Step], fps: float) -> None:
    """空序列与非法 fps 都属于"不可判定"，必须抛 ProtocolError 而不是给出假报告。"""
    with pytest.raises(ProtocolError):
        build_report(clip_id="bad", labels=labels, fps=fps, cfg=AssessConfig())


def test_build_report_to_dict_is_json_serializable() -> None:
    """报告要落盘成 JSON 并喂给演示界面，因此不能出现 Step / numpy 等不可序列化对象。"""
    report = build_report(clip_id="json", labels=full_sequence(), fps=FPS, cfg=AssessConfig())
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert "json" in text
    assert payload["clip_id"] == "json"
    assert payload["is_complete"] is True
    assert isinstance(payload["step_sequence"][0], str)  # Step 必须序列化成字符串


def test_overall_score_stays_within_unit_interval_for_every_scenario() -> None:
    """得分是给用户看的，必须始终落在 [0, 1]，否则演示页面会画出离谱的进度条。"""
    cfg = AssessConfig()
    scenarios = {
        "perfect": full_sequence(),
        "missing_three": seq(*[(step, STEP_FRAMES) for step in (A, B, C)]),
        "all_other": seq((Step.OTHER, 300)),
        "very_short": seq(*[(step, 2) for step in ALL_STEPS]),
        "reversed": seq(*[(step, STEP_FRAMES) for step in reversed(ALL_STEPS)]),
    }
    for name, labels in scenarios.items():
        report = build_report(clip_id=name, labels=labels, fps=FPS, cfg=cfg)
        assert 0.0 <= report.overall_score <= 1.0, name


def test_reversed_sequence_is_reported_as_severely_out_of_order() -> None:
    """完全倒着做的流程：顺序判定必须给出大量逆序对，且得分明显偏低。"""
    labels = seq(*[(step, STEP_FRAMES) for step in reversed(ALL_STEPS)])
    report = build_report(clip_id="reversed", labels=labels, fps=FPS, cfg=AssessConfig())
    assert report.is_in_order is False
    assert len(steps_of_kind(report, "out_of_order")) >= 5
    assert report.overall_score < 0.8


def test_confidence_below_minimum_frames_do_not_create_fake_steps() -> None:
    """低置信度帧不参与投票：否则一段"模型其实没看清"的帧会被当成真动作。"""
    labels = [A] * 20 + [B] * 20
    confidences = [0.9] * 20 + [0.1] * 20
    report = build_report(
        clip_id="lowconf",
        labels=labels,
        confidences=confidences,
        fps=FPS,
        cfg=AssessConfig(min_confidence=0.4),
    )
    # 平票时保持"上一帧标签"，因此整段都会平滑成第 1 步，后五步全部判为漏步
    assert report.statistics[1].num_frames == 0
    assert report.step_sequence == (A,)
    assert steps_of_kind(report, "missing") == [B, C, D, E, F]
    assert report.is_complete is False
