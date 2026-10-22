"""Typed records for clips, frame samples, predictions and training state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from handwash.core.labels import Step

__all__ = [
    "SCHEMA_VERSION",
    "Clip",
    "ClipPrediction",
    "ClipRecord",
    "EvalResult",
    "FrameIndex",
    "FramePrediction",
    "FrameRecord",
    "ProtocolReport",
    "Sample",
    "Split",
    "StepStatistic",
    "TimelineSegment",
]


SCHEMA_VERSION: int = 1

Split = Literal["train", "val", "test", "external"]
SPLITS: tuple[Split, ...] = ("train", "val", "test", "external")


# ============================================================================


# ============================================================================
@dataclass(frozen=True, slots=True)
class FrameIndex:
    clip_id: str
    frame_index: int
    label: Step
    timestamp_s: float | None = None
    path: str | None = None


@dataclass(frozen=True, slots=True)
class FrameRecord:
    clip_id: str
    frame_index: int
    image_path: str
    label: Step
    dataset: str
    split: Split
    timestamp_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label.value
        return d

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> FrameRecord:
        return cls(
            clip_id=str(row["clip_id"]),
            frame_index=int(row["frame_index"]),
            image_path=str(row["image_path"]),
            label=Step(str(row["label"])),
            dataset=str(row["dataset"]),
            split=str(row["split"]),  # type: ignore[arg-type]
            timestamp_s=float(row["timestamp_s"]) if row.get("timestamp_s") not in (None, "") else None,
        )


@dataclass(frozen=True, slots=True)
class ClipRecord:
    clip_id: str
    dataset: str
    split: Split
    video_path: str
    frame_count: int
    fps: float | None = None
    duration_s: float | None = None
    label_sequence: tuple[Step, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def original_video(self) -> str:

        return self.video_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "dataset": self.dataset,
            "split": self.split,
            "video_path": self.video_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "label_sequence": [s.value for s in self.label_sequence],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ClipRecord:
        return cls(
            clip_id=str(row["clip_id"]),
            dataset=str(row["dataset"]),
            split=str(row["split"]),  # type: ignore[arg-type]
            video_path=str(row["video_path"]),
            frame_count=int(row["frame_count"]),
            fps=float(row["fps"]) if row.get("fps") not in (None, "") else None,
            duration_s=float(row["duration_s"]) if row.get("duration_s") not in (None, "") else None,
            label_sequence=tuple(Step(s) for s in row.get("label_sequence", ())),
            metadata=dict(row.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class Sample:
    image: np.ndarray
    label_index: int
    clip_id: str
    frame_index: int

    def __post_init__(self) -> None:
        if self.image.ndim != 3:
            raise ValueError(
                f"Sample.image must be a three-dimensional CHW array; received shape={self.image.shape}"
            )


@dataclass(frozen=True, slots=True)
class Clip:
    clip_id: str
    frames: tuple[np.ndarray, ...]
    fps: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"Clip.fps must be positive; received {self.fps}")

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    def timestamps_s(self) -> tuple[float, ...]:
        return tuple(i / self.fps for i in range(self.num_frames))


# ============================================================================


# ============================================================================
@dataclass(frozen=True, slots=True)
class FramePrediction:
    frame_index: int
    label: Step
    confidence: float
    probs: np.ndarray | None = None
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1]; received {self.confidence}")


@dataclass(frozen=True, slots=True)
class ClipPrediction:
    clip_id: str
    frames: tuple[FramePrediction, ...]

    segments: tuple[tuple[int, int, Step], ...] = ()
    label_sequence: tuple[Step, ...] = ()
    model_name: str = "unknown"

    def __post_init__(self) -> None:
        if not self.label_sequence and self.frames:
            object.__setattr__(self, "label_sequence", tuple(f.label for f in self.frames))

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def mean_confidence(self) -> float:
        if not self.frames:
            return 0.0
        return float(np.mean([f.confidence for f in self.frames]))


# ============================================================================


# ============================================================================
@dataclass(frozen=True, slots=True)
class StepStatistic:
    step: Step
    detected: bool
    duration_s: float
    ratio: float
    num_frames: int
    mean_confidence: float

    @property
    def step_no(self) -> int:
        return self.step.order_index


@dataclass(frozen=True, slots=True)
class ProtocolViolation:
    kind: Literal["missing", "out_of_order", "insufficient_duration", "repeated", "extra_activity"]
    step: Step | None
    detail: str
    severity: Literal["info", "warning", "error"] = "warning"


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    step: Step
    start_s: float
    end_s: float
    mean_confidence: float


@dataclass(frozen=True, slots=True)
class ProtocolReport:
    clip_id: str
    is_complete: bool
    is_in_order: bool
    step_sequence: tuple[Step, ...]
    statistics: tuple[StepStatistic, ...]
    violations: tuple[ProtocolViolation, ...] = ()
    total_wash_duration_s: float = 0.0
    total_duration_s: float = 0.0
    overall_score: float = 0.0
    model_name: str = "unknown"
    notes: tuple[str, ...] = ()
    timeline: tuple[TimelineSegment, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "is_complete": self.is_complete,
            "is_in_order": self.is_in_order,
            "step_sequence": [s.value for s in self.step_sequence],
            "statistics": [
                {
                    "step": st.step.value,
                    "step_no": st.step_no,
                    "detected": st.detected,
                    "duration_s": round(st.duration_s, 3),
                    "ratio": round(st.ratio, 4),
                    "num_frames": st.num_frames,
                    "mean_confidence": round(st.mean_confidence, 4),
                }
                for st in self.statistics
            ],
            "violations": [
                {
                    "kind": v.kind,
                    "step": v.step.value if v.step else None,
                    "detail": v.detail,
                    "severity": v.severity,
                }
                for v in self.violations
            ],
            "total_wash_duration_s": round(self.total_wash_duration_s, 3),
            "total_duration_s": round(self.total_duration_s, 3),
            "overall_score": round(self.overall_score, 4),
            "model_name": self.model_name,
            "notes": list(self.notes),
            "timeline": [
                {
                    "step": seg.step.value,
                    "step_no": seg.step.order_index,
                    "start_s": round(seg.start_s, 3),
                    "end_s": round(seg.end_s, 3),
                    "mean_confidence": round(seg.mean_confidence, 4),
                }
                for seg in self.timeline
            ],
        }


# ============================================================================


# ============================================================================
@dataclass(frozen=True, slots=True)
class EvalResult:
    split: str
    model_name: str
    num_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: Mapping[str, Mapping[str, float]]
    confusion_matrix: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]
    latency_ms_per_clip: float | None = None
    confidence_intervals: Mapping[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "model_name": self.model_name,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": {k: dict(v) for k, v in self.per_class.items()},
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "labels": list(self.labels),
            "latency_ms_per_clip": self.latency_ms_per_clip,
            "confidence_intervals": (
                dict(self.confidence_intervals) if self.confidence_intervals is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvalResult:
        return cls(
            split=str(payload["split"]),
            model_name=str(payload["model_name"]),
            num_samples=int(payload["num_samples"]),
            accuracy=float(payload["accuracy"]),
            macro_f1=float(payload["macro_f1"]),
            weighted_f1=float(payload["weighted_f1"]),
            per_class={k: dict(v) for k, v in payload["per_class"].items()},
            confusion_matrix=tuple(tuple(int(x) for x in row) for row in payload["confusion_matrix"]),
            labels=tuple(str(x) for x in payload["labels"]),
            latency_ms_per_clip=(
                float(payload["latency_ms_per_clip"])
                if payload.get("latency_ms_per_clip") is not None
                else None
            ),
            confidence_intervals=(
                dict(payload["confidence_intervals"])
                if payload.get("confidence_intervals") is not None
                else None
            ),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )


def label_sequence_to_str(sequence: Sequence[Step]) -> list[str]:

    return [s.value for s in sequence]


def as_path(value: str | Path) -> Path:

    return value if isinstance(value, Path) else Path(value)
