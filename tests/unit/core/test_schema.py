"""跨层数据契约测试（src/handwash/core/schema.py）。

这些 DTO 是模块之间唯一的交流方式。任何一处 to_dict/from_dict 不对称，落盘的
manifest 或评估结果就会在下一次加载时炸掉 —— 而那时往往已经跑完了几小时训练。
所以这里对每个结构都做"往返一致"检查。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from handwash.core.labels import Step
from handwash.core.schema import (
    SCHEMA_VERSION,
    Clip,
    ClipPrediction,
    ClipRecord,
    EvalResult,
    FrameIndex,
    FramePrediction,
    FrameRecord,
    ProtocolReport,
    ProtocolViolation,
    Sample,
    StepStatistic,
    as_path,
    label_sequence_to_str,
)

pytestmark = pytest.mark.unit


def _sample_frame() -> FrameRecord:
    """一个典型的 manifest 行。"""
    return FrameRecord(
        clip_id="clip_0001",
        frame_index=3,
        image_path="data/processed/kaggle/clip_0001/000003.jpg",
        label=Step.STEP_2,
        dataset="kaggle",
        split="train",
        timestamp_s=0.6,
    )


def _sample_clip_record() -> ClipRecord:
    """一个典型的一段原始视频记录。"""
    return ClipRecord(
        clip_id="clip_0001",
        dataset="kaggle",
        split="val",
        video_path="data/raw/kaggle/step2_water/demo.mp4",
        frame_count=120,
        fps=5.0,
        duration_s=24.0,
        label_sequence=(Step.STEP_1, Step.STEP_2),
        metadata={"participant": "p01"},
    )


# --- FrameRecord / ClipRecord ----------------------------------------------
def test_frame_record_round_trips_through_dict() -> None:
    """Step 必须序列化成字符串：否则 JSON/YAML 里会出现不可读的枚举对象。"""
    record = _sample_frame()
    payload = record.to_dict()
    assert payload["label"] == "step_2_palm_over_dorsum"
    assert isinstance(payload["label"], str)
    assert FrameRecord.from_dict(payload) == record
    json.dumps(payload, ensure_ascii=False)  # 必须能直接落盘


def test_frame_record_round_trips_when_timestamp_missing() -> None:
    """timestamp_s 允许为 None；CSV 里会变成空字符串，读回来仍应是 None。"""
    record = FrameRecord(
        clip_id="c", frame_index=0, image_path="a.jpg", label=Step.OTHER, dataset="d", split="test"
    )
    payload = record.to_dict()
    assert payload["timestamp_s"] is None
    assert FrameRecord.from_dict(payload) == record
    assert FrameRecord.from_dict({**payload, "timestamp_s": ""}).timestamp_s is None


def test_frame_record_from_dict_coerces_string_numbers() -> None:
    """manifest 常从 CSV 读入，数字会是字符串，转换必须显式且安全。"""
    payload = {
        "clip_id": 1,
        "frame_index": "7",
        "image_path": Path("x.jpg"),
        "label": "step_3_fingers_interlaced",
        "dataset": "metc",
        "split": "train",
        "timestamp_s": "1.4",
    }
    record = FrameRecord.from_dict(payload)
    assert record.clip_id == "1"
    assert record.frame_index == 7
    assert record.image_path == "x.jpg"
    assert record.timestamp_s == pytest.approx(1.4)
    assert record.label is Step.STEP_3


def test_clip_record_round_trips_and_serializes_step_sequence() -> None:
    """label_sequence 是"这段视频做过哪些步骤"的唯一记录，必须能往返。"""
    record = _sample_clip_record()
    payload = record.to_dict()
    assert payload["label_sequence"] == ["step_1_palm_to_palm", "step_2_palm_over_dorsum"]
    assert ClipRecord.from_dict(payload) == record
    json.dumps(payload, ensure_ascii=False)


def test_clip_record_optional_fields_and_defaults() -> None:
    """frame_count/fps/duration_s 允许缺省；缺失时要还原成 None 而不是 0。"""
    record = ClipRecord(clip_id="c", dataset="d", split="train", video_path="v.mp4", frame_count=10)
    payload = record.to_dict()
    assert payload["fps"] is None
    assert payload["duration_s"] is None
    assert payload["label_sequence"] == []
    assert payload["metadata"] == {}
    assert ClipRecord.from_dict({**payload, "fps": "", "duration_s": ""}).fps is None


def test_clip_record_original_video_is_the_grouping_key() -> None:
    """防数据泄漏的分组键默认取原始视频路径，两个属性不能分叉。"""
    record = _sample_clip_record()
    assert record.original_video == record.video_path


def test_frame_record_is_frozen() -> None:
    """DTO 必须不可变：共享引用被就地改写是极难排查的一类 bug。"""
    record = _sample_frame()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
        record.frame_index = 99  # type: ignore[misc]


def test_frame_index_container_holds_raw_annotation() -> None:
    """原始标注（含可选时间戳与路径）与 manifest 行分开保存，避免互相污染。"""
    index = FrameIndex(clip_id="c", frame_index=0, label=Step.STEP_6, timestamp_s=0.2, path="a.jpg")
    assert index.label is Step.STEP_6
    assert index.timestamp_s == pytest.approx(0.2)


# --- Sample / Clip ----------------------------------------------------------
def test_sample_accepts_three_dimensional_chw_image() -> None:
    """CHW 三维是模型的输入约定，shape 必须原样保留。"""
    image = np.zeros((3, 224, 224), dtype=np.float32)
    sample = Sample(image=image, label_index=1, clip_id="c", frame_index=5)
    assert sample.image.shape == (3, 224, 224)
    assert sample.label_index == 1


@pytest.mark.parametrize(
    "shape",
    [(224, 224), (224, 224, 3, 1), (3,), ()],
)
def test_sample_rejects_non_three_dimensional_image(shape: tuple[int, ...]) -> None:
    """维度写错（例如忘了 permute 成 CHW）会让后续卷积层报出难以理解的错误。"""
    with pytest.raises(ValueError, match="CHW"):
        Sample(image=np.zeros(shape, dtype=np.float32), label_index=0, clip_id="c", frame_index=0)


def test_clip_exposes_frame_count_duration_and_timestamps() -> None:
    """时间轴统一由 Clip 派生，其他模块不许自己除以 fps。"""
    frames = tuple(np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4))
    clip = Clip(clip_id="c", frames=frames, fps=2.0)
    assert clip.num_frames == 4
    assert clip.duration_s == pytest.approx(2.0)
    assert clip.timestamps_s() == (0.0, 0.5, 1.0, 1.5)


@pytest.mark.parametrize("fps", [0.0, -1.0, -30.0])
def test_clip_rejects_non_positive_fps(fps: float) -> None:
    """fps <= 0 会让时长与时间戳失去意义，必须在构造时就拒绝。"""
    with pytest.raises(ValueError, match="fps"):
        Clip(clip_id="c", frames=(), fps=fps)


def test_clip_allows_empty_frame_tuple() -> None:
    """空视频（解码失败但已降级处理）不应在构造处就炸，交由上层判断。"""
    clip = Clip(clip_id="c", frames=(), fps=25.0)
    assert clip.num_frames == 0
    assert clip.duration_s == pytest.approx(0.0)


def test_clip_default_source_is_unknown() -> None:
    """来源默认 unknown，防止把"不知道哪来的视频"混进正式结果。"""
    assert Clip(clip_id="c", frames=(), fps=1.0).source == "unknown"


# --- 预测侧 -----------------------------------------------------------------
def test_frame_prediction_accepts_boundary_confidences() -> None:
    """0 与 1 是合法置信度边界，不能因为用了开区间而被误拒。"""
    assert FramePrediction(frame_index=0, label=Step.OTHER, confidence=0.0).confidence == 0.0
    assert FramePrediction(frame_index=0, label=Step.OTHER, confidence=1.0).confidence == 1.0


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -1.0])
def test_frame_prediction_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    """越界置信度通常意味着概率没做归一化，任其流入会让阈值判定全部失真。"""
    with pytest.raises(ValueError, match="confidence"):
        FramePrediction(frame_index=0, label=Step.STEP_1, confidence=confidence)


def test_clip_prediction_auto_fills_label_sequence_from_frames() -> None:
    """只给逐帧预测时，序列应自动推导，避免调用方各自拼一遍。"""
    frames = (
        FramePrediction(frame_index=0, label=Step.STEP_1, confidence=0.9),
        FramePrediction(frame_index=1, label=Step.STEP_2, confidence=0.8),
    )
    prediction = ClipPrediction(clip_id="c", frames=frames)
    assert prediction.label_sequence == (Step.STEP_1, Step.STEP_2)
    assert prediction.num_frames == 2
    assert prediction.mean_confidence() == pytest.approx(0.85)


def test_clip_prediction_keeps_explicit_label_sequence() -> None:
    """显式传入的序列（可能是平滑后的结果）不能被逐帧标签覆盖。"""
    frames = (FramePrediction(frame_index=0, label=Step.STEP_1, confidence=0.9),)
    prediction = ClipPrediction(clip_id="c", frames=frames, label_sequence=(Step.STEP_6,))
    assert prediction.label_sequence == (Step.STEP_6,)


def test_clip_prediction_mean_confidence_of_empty_frames_is_zero() -> None:
    """没有帧时不能除零。"""
    assert ClipPrediction(clip_id="c", frames=()).mean_confidence() == 0.0


def test_clip_prediction_segments_use_inclusive_end() -> None:
    """segments 里的 end 是**闭区间**，必须与 protocol.Segment.to_tuple() 一致。"""
    frames = tuple(
        FramePrediction(frame_index=i, label=Step.STEP_1, confidence=0.5) for i in range(5)
    )
    prediction = ClipPrediction(clip_id="c", frames=frames, segments=((0, 4, Step.STEP_1),))
    assert prediction.segments[0][1] == 4
    assert prediction.num_frames == 5


# --- 报告侧 -----------------------------------------------------------------
def test_step_statistic_step_no_derives_from_step_order() -> None:
    """报告里"第几步"必须来自 Step.order_index，不能另存一份容易失同步的字段。"""
    statistic = StepStatistic(
        step=Step.STEP_4, detected=True, duration_s=8.0, ratio=0.25, num_frames=40, mean_confidence=0.9
    )
    assert statistic.step_no == 4


def test_protocol_report_to_dict_is_fully_serializable() -> None:
    """报告是最终交付物，必须能直接 json.dumps 并被前端渲染。"""
    report = ProtocolReport(
        clip_id="c",
        is_complete=False,
        is_in_order=False,
        step_sequence=(Step.STEP_1, Step.FAUCET_ON),
        statistics=(
            StepStatistic(
                step=Step.STEP_1, detected=True, duration_s=5.0, ratio=0.5, num_frames=25,
                mean_confidence=0.75,
            ),
        ),
        violations=(ProtocolViolation(kind="missing", step=Step.STEP_4, detail="未检出第 4 步"),),
        total_wash_duration_s=10.0,
        total_duration_s=12.0,
        overall_score=0.6,
        model_name="baseline",
        notes=("demo",),
    )
    payload = report.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["step_sequence"] == ["step_1_palm_to_palm", "faucet_on"]
    assert payload["violations"][0]["step"] == "step_4_backs_of_fingers"
    assert payload["statistics"][0]["step_no"] == 1
    assert payload["overall_score"] == pytest.approx(0.6)


def test_protocol_violation_allows_step_less_global_finding() -> None:
    """总时长不足这类违规不属于某一步，step 必须允许为 None 并在序列化时保持 None。"""
    report = ProtocolReport(
        clip_id="c",
        is_complete=True,
        is_in_order=True,
        step_sequence=(),
        statistics=(),
        violations=(ProtocolViolation(kind="insufficient_duration", step=None, detail="总时长不足"),),
    )
    assert report.to_dict()["violations"][0]["step"] is None


# --- 评估侧 -----------------------------------------------------------------
def _sample_eval_result() -> EvalResult:
    return EvalResult(
        split="test",
        model_name="yolo26n-cls",
        num_samples=120,
        accuracy=0.83,
        macro_f1=0.79,
        weighted_f1=0.82,
        per_class={"class_0": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 40}},
        confusion_matrix=((30, 10), (5, 35)),
        labels=("step_1_palm_to_palm", "step_2_palm_over_dorsum"),
        latency_ms_per_clip=12.5,
    )


def test_eval_result_round_trips_through_dict() -> None:
    """评估结果要落盘再被报告脚本读回画表，往返必须完全一致。"""
    result = _sample_eval_result()
    payload = result.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert EvalResult.from_dict(payload) == result


def test_eval_result_round_trips_nested_containers_as_immutable_types() -> None:
    """confusion_matrix / labels 读回后必须是 tuple，防止下游就地改写。"""
    restored = EvalResult.from_dict(_sample_eval_result().to_dict())
    assert isinstance(restored.confusion_matrix, tuple)
    assert all(isinstance(row, tuple) for row in restored.confusion_matrix)
    assert isinstance(restored.labels, tuple)
    assert restored.confusion_matrix[0] == (30, 10)


def test_eval_result_latency_can_be_absent() -> None:
    """没测延迟时不能写成 0（会污染"平均延迟"统计），必须保持 None。"""
    payload = _sample_eval_result().to_dict()
    payload["latency_ms_per_clip"] = None
    assert EvalResult.from_dict(payload).latency_ms_per_clip is None


def test_eval_result_from_dict_uses_default_schema_version_when_missing() -> None:
    """旧结果文件可能没有 schema_version，读取时应回退到当前版本而不是报错。"""
    payload = _sample_eval_result().to_dict()
    payload.pop("schema_version")
    assert EvalResult.from_dict(payload).schema_version == SCHEMA_VERSION


# --- 小工具 -----------------------------------------------------------------
def test_label_sequence_to_str_matches_to_dict_convention() -> None:
    """所有序列化路径必须使用同一套 Step -> str 规则，否则文件间会对不上。"""
    assert label_sequence_to_str([Step.STEP_1, Step.FAUCET_OFF]) == ["step_1_palm_to_palm", "faucet_off"]


def test_as_path_normalizes_str_and_keeps_path() -> None:
    """路径参数在 io 层大量出现，统一转换可以省掉一堆重复判断。"""
    assert as_path("a/b.txt") == Path("a/b.txt")
    original = Path("c.txt")
    assert as_path(original) is original
