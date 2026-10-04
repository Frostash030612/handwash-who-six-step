"""把 Ultralytics 分类权重 exp.pt 接入本机摄像头演示；可选叠加 GRU 时序头。

exp.pt 逐帧分类，只看当前一帧。手部大半移出画面时，单帧已不足以区分第 3—6 步，
模型会连续数秒给出同一个错误类别（典型是认成第 6 步），平滑窗口无法纠正。
时序头读取 exp.pt 分类层前的特征序列，用过去几秒的上下文判断当前步骤。
exp.pt 骨干保持冻结；时序头由 ``scripts/train_exp_temporal_head.py`` 单独训练。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from handwash.core.config import ResolvedConfig
from handwash.core.labels import STEP_ORDER, LabelSpace, Step
from handwash.errors import BackendUnavailableError, DataError, ModelError
from handwash.paths import ensure_dir
from handwash.pipelines.common import resolve_device

__all__ = [
    "DEFAULT_TEMPORAL_HEAD",
    "EXP_CLASS_NAMES",
    "ExpDemoClassifier",
    "ExpFeatureExtractor",
    "ExpTemporalHead",
    "file_sha256",
    "load_temporal_head",
    "save_temporal_head",
]


EXP_CLASS_NAMES = (
    "0_other_or_nonstandard",
    "1_palm_to_palm",
    "2_palm_over_dorsum",
    "3_palm_to_palm_interlaced",
    "4_backs_of_fingers",
    "5_thumb_rubbing",
    "6_fingertips_to_palm",
)
_HEAD_FORMAT = "exp-temporal-head/1"
#: 时序头默认文件（项目根目录，由 scripts/train_exp_temporal_head.py 生成）
DEFAULT_TEMPORAL_HEAD = "exp_temporal_head.pt"


def file_sha256(path: str | Path) -> str:
    """权重文件指纹：时序头只对训练它时所用的那份 exp.pt 有效。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise DataError(f"摄像头帧必须是 HWC RGB uint8，实际 {image.shape} / {image.dtype}")
    # Ultralytics 的 numpy 输入遵循 OpenCV BGR 约定；浏览器送来的帧是 RGB。
    return np.ascontiguousarray(image[:, :, ::-1])


class ExpFeatureExtractor:
    """exp.pt 的逐帧前向：同时返回 7 类概率和分类层前的特征。

    训练脚本和网页推理共用这一入口，保证两边预处理完全一致
    （Ultralytics 自带的缩放与中心裁剪，输入为 OpenCV BGR）。
    """

    def __init__(self, checkpoint: str | Path, *, device: str, cache_dir: str | Path) -> None:
        # Ultralytics 会初始化设置与绘图库缓存；将其放在本项目可写的演示目录。
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
        if names != EXP_CLASS_NAMES:
            raise ModelError(
                f"exp.pt 的类别顺序与已核对的演示模型不一致：{names}",
                hint="请使用当前项目根目录的 exp.pt；其他权重应走正式模型入口。",
            )
        try:
            self.image_size = int(model.overrides.get("imgsz", 320))
        except (TypeError, ValueError) as exc:
            raise ModelError("exp.pt 未记录有效的输入尺寸") from exc
        if self.image_size <= 0:
            raise ModelError("exp.pt 未记录有效的输入尺寸")
        self.model = model
        self.device = device
        self.checkpoint = Path(checkpoint)

        # 先推理一帧，让 Ultralytics 建好推理后端，再在分类头的 Linear 上取输入特征。
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.model.predict(source=blank, imgsz=self.image_size, device=self.device, verbose=False)
        try:
            linear = self.model.predictor.model.model.model[-1].linear
        except (AttributeError, IndexError, TypeError) as exc:
            raise ModelError("无法定位 exp.pt 的分类层", hint="Ultralytics 版本与当前适配器不兼容。") from exc
        if not isinstance(linear, nn.Linear):
            raise ModelError("exp.pt 的分类头缺少 Linear 层", hint="Ultralytics 版本与当前适配器不兼容。")
        self.feature_dim = int(linear.in_features)
        self._captured: list[torch.Tensor] = []
        linear.register_forward_hook(lambda _module, inputs, _output: self._captured.append(inputs[0].detach()))

    def forward_bgr(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """BGR 帧列表 -> ``(features[N, D], probabilities[N, 7])``。"""
        if not frames:
            raise DataError("至少需要一帧画面")
        self._captured.clear()
        results = self.model.predict(
            source=list(frames),
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if any(result.probs is None for result in results):
            raise ModelError("exp.pt 没有返回分类概率")
        probabilities = np.stack(
            [result.probs.data.detach().cpu().numpy() for result in results]
        ).astype(np.float32)
        features = torch.cat(self._captured).float().cpu().numpy() if self._captured else np.empty((0,))
        if probabilities.shape != (len(frames), len(EXP_CLASS_NAMES)) or not np.isfinite(probabilities).all():
            raise ModelError("exp.pt 返回的分类概率无效")
        if features.shape != (len(frames), self.feature_dim) or not np.isfinite(features).all():
            raise ModelError(f"exp.pt 特征形状异常：{features.shape}，期望 ({len(frames)}, {self.feature_dim})")
        return features, probabilities


class ExpTemporalHead(nn.Module):
    """单向 GRU 时序头：只用当前帧与过去帧，因此可以逐帧流式推理。"""

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden: int = 128,
        num_classes: int = len(EXP_CLASS_NAMES),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hparams = {
            "feature_dim": int(feature_dim),
            "hidden": int(hidden),
            "num_classes": int(num_classes),
            "dropout": float(dropout),
        }
        # 特征标准化参数随权重一起保存，推理时无需另带统计量文件。
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``features[B, T, D]`` -> ``(logits[B, T, C], state)``。"""
        normalized = (features - self.feature_mean) / self.feature_std
        output, state = self.rnn(self.encoder(normalized), state)
        return self.classifier(output), state


def save_temporal_head(
    path: str | Path,
    head: ExpTemporalHead,
    *,
    exp_sha256: str,
    fps: float,
    metrics: dict[str, Any],
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(
        {
            "format": _HEAD_FORMAT,
            "class_names": list(EXP_CLASS_NAMES),
            "exp_sha256": exp_sha256,
            "fps": float(fps),
            "hparams": dict(head.hparams),
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "metrics": metrics,
        },
        target,
    )
    return target


def load_temporal_head(
    path: str | Path, *, exp_checkpoint: str | Path, fps: float
) -> ExpTemporalHead:
    """加载时序头，并核对它确实是在这份 exp.pt、这个采样率上训练的。"""
    source = Path(path)
    if not source.is_file():
        raise DataError(f"找不到时序头权重：{source}", hint="先运行 scripts/train_exp_temporal_head.py。")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelError(f"无法读取时序头权重：{source}", hint=str(exc)) from exc
    if not isinstance(payload, dict) or payload.get("format") != _HEAD_FORMAT:
        raise ModelError(f"不是 exp.pt 时序头权重：{source}")
    if tuple(payload.get("class_names", ())) != EXP_CLASS_NAMES:
        raise ModelError("时序头的类别顺序与 exp.pt 不一致")
    if payload.get("exp_sha256") != file_sha256(exp_checkpoint):
        raise ModelError(
            "时序头不是在当前 exp.pt 上训练的",
            hint="exp.pt 更换后需重新运行 scripts/train_exp_temporal_head.py。",
        )
    if abs(float(payload.get("fps", 0.0)) - float(fps)) > 1e-6:
        raise ModelError(
            f"时序头训练采样率 {payload.get('fps')} fps 与当前配置 {fps} fps 不一致",
            hint="GRU 按固定帧间隔学习时间上下文，请保持 dataset.prep.fps 与训练时一致。",
        )
    head = ExpTemporalHead(**payload["hparams"])
    head.load_state_dict(payload["state_dict"])
    return head.eval()


class _TemporalStream:
    """一个摄像头会话内的 GRU 隐状态；会话之间互不影响。"""

    def __init__(self, classifier: ExpDemoClassifier) -> None:
        self._classifier = classifier
        self._state: torch.Tensor | None = None

    @torch.inference_mode()
    def predict(self, frame: np.ndarray) -> np.ndarray:
        features, _ = self._classifier.extractor.forward_bgr([_rgb_to_bgr(frame)])
        head = self._classifier.temporal_head
        assert head is not None
        logits, state = head(torch.from_numpy(features)[None], self._state)
        probabilities = torch.softmax(logits[0, -1], dim=-1).numpy().astype(np.float32)
        if not np.isfinite(probabilities).all():
            raise ModelError("时序头返回的分类概率无效")
        self._state = state
        return probabilities


class ExpDemoClassifier:
    """仅接受已核对过的七类 exp.pt，复用正式页面和会话规则。"""

    def __init__(
        self,
        rc: ResolvedConfig,
        checkpoint: str | Path,
        *,
        temporal_head: str | Path | None = None,
    ) -> None:
        self.rc = rc
        self.space = LabelSpace("exp_demo", (Step.OTHER, *STEP_ORDER))
        self.device = str(resolve_device(rc.runtime.device))
        cache_dir = ensure_dir(rc.resolve_out_dir() / "demo_runtime")
        self.extractor = ExpFeatureExtractor(checkpoint, device=self.device, cache_dir=cache_dir)
        self.image_size = self.extractor.image_size
        self.temporal_head = (
            None
            if temporal_head is None
            else load_temporal_head(temporal_head, exp_checkpoint=checkpoint, fps=rc.dataset.prep.fps)
        )
        if self.temporal_head is not None and self.temporal_head.hparams["feature_dim"] != self.extractor.feature_dim:
            raise ModelError("时序头的输入维度与 exp.pt 特征维度不一致")
        # 显示在英文网页上，也写入报告的 model_name 字段，因此用英文。
        self.model_name = (
            "exp.pt + GRU temporal head" if self.temporal_head is not None else "exp.pt (frame-only)"
        )

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """只看当前帧的 exp.pt 概率（未加载时序头时使用）。"""
        _, probabilities = self.extractor.forward_bgr([_rgb_to_bgr(frame)])
        return probabilities[0]

    def new_stream(self) -> _TemporalStream | None:
        """为一个新会话创建时序状态；未加载时序头时返回 None，按逐帧方式推理。"""
        return None if self.temporal_head is None else _TemporalStream(self)
