"""把早期 Ultralytics 分类权重 exp.pt 接入本机摄像头演示。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from handwash.core.config import ResolvedConfig
from handwash.core.labels import STEP_ORDER, LabelSpace, Step
from handwash.errors import BackendUnavailableError, DataError, ModelError
from handwash.paths import ensure_dir
from handwash.pipelines.common import resolve_device

__all__ = ["ExpDemoClassifier"]


_EXPECTED_NAMES = (
    "0_other_or_nonstandard",
    "1_palm_to_palm",
    "2_palm_over_dorsum",
    "3_palm_to_palm_interlaced",
    "4_backs_of_fingers",
    "5_thumb_rubbing",
    "6_fingertips_to_palm",
)


class ExpDemoClassifier:
    """仅接受已核对过的七类 exp.pt，复用正式页面和会话规则。"""

    def __init__(self, rc: ResolvedConfig, checkpoint: str | Path) -> None:
        self.rc = rc
        self.space = LabelSpace("exp_demo", (Step.OTHER, *STEP_ORDER))
        self.model_name = "exp.pt（Ultralytics 演示）"
        self.device = str(resolve_device(rc.runtime.device))

        # Ultralytics 会初始化设置与绘图库缓存；将其放在本项目可写的演示目录。
        cache_dir = ensure_dir(rc.resolve_out_dir() / "demo_runtime")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise BackendUnavailableError("ultralytics", "演示版需要安装 ultralytics。") from exc

        try:
            model = YOLO(str(checkpoint))
        except Exception as exc:
            raise ModelError(f"无法加载 Ultralytics 演示权重：{checkpoint}", hint=str(exc)) from exc
        if model.task != "classify":
            raise ModelError(f"exp.pt 演示入口要求分类模型，实际任务为 {model.task!r}")
        names = tuple(str(model.names.get(index, "")) for index in range(len(model.names)))
        if names != _EXPECTED_NAMES:
            raise ModelError(
                f"exp.pt 的类别顺序与已核对的演示模型不一致：{names}",
                hint="请使用当前项目根目录的 exp.pt；其他权重应走正式模型入口。",
            )
        self.model = model
        try:
            self.image_size = int(model.overrides.get("imgsz", 320))
        except (TypeError, ValueError) as exc:
            raise ModelError("exp.pt 未记录有效的输入尺寸") from exc
        if self.image_size <= 0:
            raise ModelError("exp.pt 未记录有效的输入尺寸")

    def predict(self, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DataError(f"摄像头帧必须是 HWC RGB uint8，实际 {image.shape} / {image.dtype}")
        # Ultralytics 的 numpy 输入遵循 OpenCV BGR 约定；浏览器送来的帧是 RGB。
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        result = self.model.predict(
            source=bgr,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        if result.probs is None:
            raise ModelError("exp.pt 没有返回分类概率")
        probabilities = result.probs.data.detach().cpu().numpy().astype(np.float32)
        if probabilities.shape != (len(self.space),) or not np.isfinite(probabilities).all():
            raise ModelError("exp.pt 返回的分类概率无效")
        return probabilities
