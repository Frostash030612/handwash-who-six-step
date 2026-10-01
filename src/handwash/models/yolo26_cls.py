"""YOLO26n-cls 适配器（主模型）。

来自选题文档：主模型采用 Ultralytics 的 ``yolo26n-cls``，对视频帧做动作分类。

**重要现实检查（不要跳过）**
----------------------------
Ultralytics 的模型名随时间变化。本适配器因此做了三件事：

1. ``arch="yolo26n-cls"`` 时先探测 ``ultralytics`` 是否能构建该权重；
   构建失败会抛出带**明确指引**的 ``ModelError``，而不是一个难懂的
   权重下载报错。
2. 注册了 ``yolon-cls`` 通用名，允许用配置指定任意分类权重：
   ``model.pretrained=yolo11n-cls.pt``（或 yolov8n-cls.pt）即可，
   这样即使 yolo26 权重尚未发布，实验也能立刻跑起来。
3. 提供 ``probe_yolo_availability()`` 供 ``handwash doctor`` 与报告使用。

输入约定（与其它模型不同，必须记住）
------------------------------------
Ultralytics 的 ``Classify`` 网络**自带**归一化（内部按 0-255 处理），
所以使用本适配器时，配置里应写::

    model:
      arch: yolo26n-cls
      normalize: zero_one      # 不要用 imagenet

否则等于归一化了两次，准确率会莫名偏低。``handwash doctor`` 会检查这一点。

换骨干时**只改配置**，不要改代码（CONTRIBUTING.md R18）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from handwash.core.registry import register_model
from handwash.errors import BackendUnavailableError, ModelError
from handwash.logging import get_logger
from handwash.models.base import BaseClassifier

__all__ = ["YoloClassifier", "probe_yolo_availability", "YOLO_DEFAULT_BACKBONE"]

log = get_logger(__name__)

#: 主模型权重名（选题文档指定）。若官方尚未发布，doctor 会给出替代方案。
YOLO_DEFAULT_BACKBONE = "yolo26n-cls.pt"

#: 已知可用的替代分类权重（按新旧排序），用于自动回退提示
_FALLBACK_BACKBONES: tuple[str, ...] = ("yolo11n-cls.pt", "yolov8n-cls.pt", "yolov8s-cls.pt")


def probe_yolo_availability(backbone: str = YOLO_DEFAULT_BACKBONE) -> dict[str, Any]:
    """探测 ultralytics 与指定权重是否可用（不抛异常，供 doctor 调用）。"""
    result: dict[str, Any] = {"ultralytics": False, "version": None, "backbone": backbone, "loadable": False}
    try:
        import ultralytics
    except ImportError:
        result["hint"] = "未安装 ultralytics：conda env update -f environment.yml"
        return result

    result["ultralytics"] = True
    result["version"] = getattr(ultralytics, "__version__", "unknown")
    try:
        from ultralytics import YOLO

        YOLO(backbone)
        result["loadable"] = True
    except Exception as exc:  # noqa: BLE001 - ultralytics 会抛各种异常
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["hint"] = (
            f"权重 {backbone} 目前不可用。可用替代：{list(_FALLBACK_BACKBONES)}；"
            "在配置里写 model.pretrained=<权重名> 即可沿用本适配器。"
        )
    return result


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise BackendUnavailableError(
            "ultralytics",
            "conda env update -f environment.yml（或 pip install ultralytics）",
        ) from exc
    return YOLO


def _find_last_linear(model: nn.Module) -> tuple[nn.Module, str, nn.Linear]:
    """定位最后一个 ``nn.Linear``，返回 ``(父模块, 子模块名, 该层)``。"""
    found: tuple[nn.Module, str, nn.Linear] | None = None
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                found = (parent, child_name, child)
    if found is None:
        raise ModelError(
            "无法在 YOLO 模型中找到分类线性层",
            hint="ultralytics 内部结构可能变了；请更新本适配器的 _find_last_linear。",
        )
    return found


@register_model("yolo26n-cls")
@register_model("yolon-cls")
@register_model("yolo-cls")
class YoloClassifier(BaseClassifier):
    """Ultralytics YOLO 分类模型的薄适配层。

    只做三件事，不多做：
        1. 用 ``ultralytics.YOLO`` 载入官方预训练权重当骨干；
        2. 用自己的 ``Linear(D, C)`` 替换原分类头，保证 C 与标签空间一致；
        3. 把输出统一成 ``(B, 1, C)``。

    权重保存：``models/checkpoint.py`` 只序列化本类自己的 ``head.*`` 参数
    （骨干参数来自官方权重文件，重复保存只会让 checkpoint 变成几百 MB）。
    """

    arch_name = "yolo-cls"
    _original_num_classes: int = 0

    def __init__(
        self,
        num_classes: int,
        *,
        backbone: str = YOLO_DEFAULT_BACKBONE,
        pretrained: bool | str = True,
        dropout: float = 0.2,
        **_: Any,
    ) -> None:
        super().__init__(num_classes, dropout=dropout)
        self.backbone_name = str(pretrained) if isinstance(pretrained, str) and pretrained != "auto" else str(backbone)
        self.arch_name = Path(self.backbone_name).stem

        YOLO = _require_ultralytics()
        try:
            # 传权重路径/名字即加载预训练；pretrained=False 时用官方结构但不加载权重
            self.yolo = YOLO(self.backbone_name)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(
                f"无法加载 YOLO 权重 `{self.backbone_name}`：{exc}",
                hint=(
                    "确认网络可访问 ultralytics 权重源，或把 model.pretrained 指向本地 .pt 文件；"
                    f"也可先用替代权重 {list(_FALLBACK_BACKBONES)} 跑通流程。"
                ),
            ) from exc

        self.yolo_model: nn.Module = self.yolo.model  # type: ignore[assignment]
        _, _, original_head = _find_last_linear(self.yolo_model)
        self._feature_dim = int(original_head.in_features)
        self._original_num_classes = int(original_head.out_features)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self._feature_dim, num_classes),
        )
        log.info(
            "构建 YOLO 分类模型：权重=%s，特征维度=%d，类别数=%d（官方权重原为 %d 类）",
            self.backbone_name,
            self._feature_dim,
            num_classes,
            self._original_num_classes,
        )

    # --- 前向 -------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """取骨干最后一层之前的特征，返回 ``(B, 1, D)``。

        实现方式：临时把分类线性层替换为 ``Identity``，跑完前向再还原。
        之所以不用"把权重设成单位矩阵"这种技巧：那样会改变数值尺度，
        且对带 bias 的头不成立。
        """
        parent, name, head = _find_last_linear(self.yolo_model)
        setattr(parent, name, nn.Identity())
        try:
            feats = self.yolo_model(x)
        finally:
            setattr(parent, name, head)

        if isinstance(feats, (list, tuple)):  # 某些版本返回 (logits, extras)
            feats = feats[0]
        if not isinstance(feats, torch.Tensor):  # pragma: no cover - 防御性
            raise ModelError(f"YOLO 前向返回了非张量：{type(feats).__name__}")
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)
        if feats.shape[-1] != self._feature_dim:
            raise ModelError(
                f"YOLO 特征维度与预期不符：得到 {feats.shape[-1]}，期望 {self._feature_dim}",
                hint="请检查 ultralytics 版本；临时方案：model.arch=mobilenet_v2。",
            )
        return feats.unsqueeze(1)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def backbone_modules(self) -> list[nn.Module]:
        return [self.yolo_model]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """只返回本项目自己训练的参数（``head.*``）。"""
        return {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if name.startswith("head.")
        }

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "backbone_name": self.backbone_name,
                "original_num_classes": self._original_num_classes,
                "note": "骨干权重来自 ultralytics 官方文件，checkpoint 只保存 head.*",
            }
        )
        return info
