"""外接摄像头的逐帧 YOLO 分类与会话状态。

摄像头画面由网页采集。此模块只接收 RGB 帧和采集时间，不读取摄像头设备。
逐帧模型每次只看当前帧，实时显示平均已到达帧的概率；带时序头的分类器
（提供 ``new_stream``）在会话内保留过去帧的状态，此时不再叠加概率平均。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from handwash.core.config import ResolvedConfig
from handwash.core.labels import STEP_ORDER, STEP_ZH, LabelSpace, Step, get_label_space
from handwash.core.protocol import build_report
from handwash.core.schema import ProtocolReport
from handwash.data.transforms import build_transforms
from handwash.errors import ConfigError, DataError
from handwash.pipelines.common import resolve_device

__all__ = ["LiveClassifier", "LiveSession"]


class LiveClassifier:
    """加载后的项目分类模型；一个实例可供多个顺序帧复用。"""

    def __init__(self, rc: ResolvedConfig, model: torch.nn.Module) -> None:
        if not rc.model.arch.lower().startswith(("yolo", "yolon")):
            raise ConfigError("摄像头入口要求 YOLO 分类架构")
        if rc.model.temporal.kind != "none" or rc.train.mode != "frame" or rc.infer.mode != "frame":
            raise ConfigError(
                "摄像头入口要求逐帧分类配置",
                hint="设置 model.temporal.kind=none、train.mode=frame、infer.mode=frame。",
            )
        self.rc = rc
        self.device = resolve_device(rc.runtime.device)
        self.model = model.to(self.device).eval()
        self.transform = build_transforms(rc.model, mode="infer", seed=rc.runtime.seed)
        self.space = get_label_space(rc.label_space)
        self.model_name = str(getattr(model, "arch_name", rc.model.arch))

    @torch.inference_mode()
    def predict(self, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DataError(f"摄像头帧必须是 HWC RGB uint8，实际 {image.shape} / {image.dtype}")
        transformed = np.ascontiguousarray(self.transform(image))
        tensor = torch.as_tensor(transformed, dtype=torch.float32, device=self.device)
        logits = self.model.logits(tensor[None, None, :, :, :])
        if logits.shape != (1, 1, len(self.space)):
            raise DataError(
                f"模型输出类别数与标签空间不符：{tuple(logits.shape)}，期望 (1, 1, {len(self.space)})"
            )
        return torch.softmax(logits[0, 0], dim=-1).cpu().numpy().astype(np.float32)


@dataclass(slots=True)
class _Observation:
    time_s: float
    label: Step
    confidence: float


class _FrameClassifier(Protocol):
    rc: ResolvedConfig
    space: LabelSpace
    model_name: str

    def predict(self, frame: np.ndarray) -> np.ndarray: ...


class LiveSession:
    """一段摄像头会话：连续分类、过去帧平滑和最终规则报告。"""

    def __init__(self, classifier: _FrameClassifier, *, session_id: str) -> None:
        self.classifier = classifier
        self.session_id = session_id
        self._first_capture_s: float | None = None
        self._last_capture_s: float | None = None
        self._stream = self._open_stream()
        # 时序头已在模型内部整合过去帧；再做概率平均只会增加切换延迟。
        window = 1 if self._stream is not None else max(1, classifier.rc.assess.smooth_window)
        self._prob_window: deque[np.ndarray] = deque(maxlen=window)
        self._observations: list[_Observation] = []

    def _open_stream(self) -> Any:
        new_stream = getattr(self.classifier, "new_stream", None)
        return new_stream() if callable(new_stream) else None

    @property
    def num_frames(self) -> int:
        return len(self._observations)

    def add_frame(self, frame: np.ndarray, *, capture_time_s: float) -> dict[str, Any]:
        if not math.isfinite(capture_time_s) or capture_time_s < 0:
            raise DataError("摄像头采集时间必须是非负的有限秒数")
        if self._last_capture_s is not None and capture_time_s <= self._last_capture_s:
            raise DataError("摄像头帧的采集时间必须严格递增")
        gap = (
            self._last_capture_s is not None
            and capture_time_s - self._last_capture_s > 2.0 / self.classifier.rc.dataset.prep.fps
        )
        if gap and self._stream is not None:
            # 断流后的画面与之前的上下文不连续，时序状态从头开始。
            self._stream = self._open_stream()
        source = self._stream if self._stream is not None else self.classifier
        probabilities = source.predict(frame)
        if self._first_capture_s is None:
            self._first_capture_s = capture_time_s
        if gap:
            self._prob_window.clear()
        self._last_capture_s = capture_time_s
        self._prob_window.append(probabilities)
        averaged = np.mean(np.stack(tuple(self._prob_window)), axis=0)
        index = int(np.argmax(averaged))
        confidence = float(averaged[index])
        label = self.classifier.space.to_label(index)
        if confidence < self.classifier.rc.assess.min_confidence:
            label = Step.UNKNOWN
        elapsed = capture_time_s - self._first_capture_s
        self._observations.append(_Observation(elapsed, label, confidence))
        report = self.report()
        stats = {stat.step: stat for stat in report.statistics}
        return {
            "session_id": self.session_id,
            "frame_count": self.num_frames,
            "elapsed_s": round(elapsed, 3),
            "label": label.value,
            "label_zh": STEP_ZH[label],
            "step_no": label.order_index,
            "confidence": round(confidence, 4),
            "steps": [
                {
                    "step": step.value,
                    "step_no": step.order_index,
                    "name": STEP_ZH[step],
                    "detected": bool(stats[step].detected) if step in stats else False,
                    "duration_s": round(stats[step].duration_s, 2) if step in stats else 0.0,
                }
                for step in STEP_ORDER
            ],
            # 与最终报告同一套分段：页面画出的顺序就是报告判定所依据的顺序。
            "timeline": report.to_dict()["timeline"],
            "provisional": True,
        }

    def _uniform_observations(self) -> tuple[list[Step], list[float]]:
        """按目标采样率重建时间轴；只有邻近真实采集帧的时刻才有动作标签。"""
        if not self._observations:
            raise DataError("会话没有收到摄像头帧")
        fps = float(self.classifier.rc.dataset.prep.fps)
        interval = 1.0 / fps
        duration = self._observations[-1].time_s
        count = max(1, int(math.floor(duration * fps + 0.5)) + 1)
        labels: list[Step] = []
        confidences: list[float] = []
        source_index = 0
        for position in range(count):
            time_s = position * interval
            while (
                source_index + 1 < len(self._observations)
                and self._observations[source_index + 1].time_s <= time_s
            ):
                source_index += 1
            source = self._observations[source_index]
            if source_index + 1 < len(self._observations):
                following = self._observations[source_index + 1]
                if abs(following.time_s - time_s) < abs(source.time_s - time_s):
                    source = following
            if abs(time_s - source.time_s) > 0.5 * interval:
                labels.append(Step.UNKNOWN)
                confidences.append(0.0)
            else:
                labels.append(source.label)
                confidences.append(source.confidence)
        return labels, confidences

    def report(self) -> ProtocolReport:
        labels, confidences = self._uniform_observations()
        return build_report(
            clip_id=self.session_id,
            labels=labels,
            confidences=confidences,
            fps=self.classifier.rc.dataset.prep.fps,
            cfg=self.classifier.rc.assess,
            model_name=self.classifier.model_name,
            apply_smoothing=False,
        )

    def prediction_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": self.session_id,
                "time_s": round(item.time_s, 4),
                "label": item.label.value,
                "confidence": round(item.confidence, 6),
            }
            for item in self._observations
        ]
