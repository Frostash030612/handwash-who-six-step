"""时序模型：GRU / TCN / 均值池化，用于消除单帧跳变（研究问题 3）。

在整体架构里的位置
------------------
::

    视频帧 -> [骨干 CNN：YOLO26n-cls 或 MobileNetV2] -> 特征 (B, T, D)
                                                     -> [时序头] -> logits (B, T, C)
                                                     -> 平滑/分段（core.protocol）
                                                     -> 漏步与时长判定

设计要点
--------
* 时序头**逐帧输出** logits（而不是整段一个标签），因为我们的核心任务是
  "每一帧属于哪一步"，逐帧输出才能做分段与时长统计。
  双向模型会看到未来帧，但它仍然逐位置输出，只是每个位置的表示里含上下文。
  **注意**：双向模型不能用于实时/在线演示，报告里必须说明这一点。
* 支持把"纯帧级模型的预测"作为输入（``kind="mean_pool"`` 等价于时间平滑基线）。
* 支持从已训练好的帧级 checkpoint 初始化骨干（两级训练）：
  这是本项目最省算力的迁移路径，也是论文 baseline 的做法。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from handwash.core.config import TemporalConfig
from handwash.core.registry import register_model
from handwash.errors import ModelError
from handwash.logging import get_logger
from handwash.models.base import BaseClassifier

__all__ = ["TemporalClassifier", "GRUHead", "TCNHead", "MeanPoolHead"]

log = get_logger(__name__)


# ============================================================================
# 时序头
# ============================================================================
class GRUHead(nn.Module):
    """单向/双向 GRU 逐帧分类头。"""

    def __init__(self, feature_dim: int, num_classes: int, cfg: TemporalConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(feats)
        return self.classifier(self.drop(self.norm(out)))


class _TemporalBlock(nn.Module):
    """TCN 残差块：两层因果卷积 + 权重归一化 + 残差。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # 因果卷积的左填充量
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        )
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.downsample = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        out = self.drop(out)
        out = self.relu(self.conv2(nn.functional.pad(out, (self.pad, 0))))
        out = self.drop(out)
        return self.relu(out + self.downsample(x))


class TCNHead(nn.Module):
    """膨胀因果卷积（TCN）逐帧分类头。

    相比 GRU 的优点：并行度高、感受野可控、对超参不敏感；适合作为"时序方案"的对照。
    """

    def __init__(self, feature_dim: int, num_classes: int, cfg: TemporalConfig) -> None:
        super().__init__()
        channels = int(cfg.hidden_size)
        self.input_proj = nn.Conv1d(feature_dim, channels, 1)
        self.blocks = nn.ModuleList(
            [
                _TemporalBlock(channels, cfg.kernel_size, dilation, cfg.dropout)
                for dilation in cfg.dilations
            ]
        )
        self.norm = nn.LayerNorm(channels)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)  # (B, D, T)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, T, C')
        return self.classifier(self.norm(x))


class MeanPoolHead(nn.Module):
    """无参数的纯分类头：直接对特征做线性映射（等价于"不做时序建模"）。

    存在的意义：它是"时序处理到底有没有用"这个问题的**下界基线**。
    报告里必须给出它的数字，否则无法说明 GRU/TCN 的收益。
    """

    def __init__(self, feature_dim: int, num_classes: int, cfg: TemporalConfig) -> None:
        super().__init__()
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.drop(feats))


_HEADS: dict[str, type[nn.Module]] = {
    "gru": GRUHead,
    "tcn": TCNHead,
    "mean_pool": MeanPoolHead,
}


# ============================================================================
# 组合模型
# ============================================================================
@register_model("temporal")
@register_model("gru")
@register_model("tcn")
class TemporalClassifier(BaseClassifier):
    """骨干 + 时序头 的组合模型。

    Parameters
    ----------
    backbone_arch:
        骨干注册名（``yolo26n-cls`` / ``mobilenet_v2`` / ...）。
    temporal_cfg:
        时序头配置（``ModelConfig.temporal``）。
    backbone_state:
        可选的骨干权重（两级训练：先训帧级模型，再冻结骨干训时序头）。
    """

    arch_name = "temporal"

    def __init__(
        self,
        num_classes: int,
        *,
        backbone_arch: str = "mobilenet_v2",
        temporal_cfg: TemporalConfig | None = None,
        backbone_kwargs: dict[str, Any] | None = None,
        backbone_state: dict[str, torch.Tensor] | str | Path | None = None,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__(num_classes, dropout=dropout)
        cfg = temporal_cfg or TemporalConfig()
        cfg.validate()
        if cfg.kind not in _HEADS:
            raise ModelError(
                f"未知的时序头类型：{cfg.kind!r}", hint=f"可用：{sorted(_HEADS)}"
            )

        from handwash.models.factory import build_backbone

        # 显式参数优先：从 backbone_kwargs 里摘出 pretrained 后，其余原样透传。
        # （不做这步会出现 "got multiple values for keyword argument"，或更糟——
        #  pretrained 被静默丢弃后默认变成 True，离线环境直接下载失败。）
        kwargs = dict(backbone_kwargs or {})
        backbone_pretrained = kwargs.pop("pretrained", True)
        for reserved in ("num_classes", "backbone", "dropout"):
            kwargs.pop(reserved, None)
        self.backbone = build_backbone(
            backbone_arch,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=backbone_pretrained,
            **kwargs,
        )
        self._feature_dim = int(self.backbone.feature_dim)
        self.temporal_kind = cfg.kind
        self.arch_name = f"{backbone_arch}+{cfg.kind}"

        if cfg.kind == "mean_pool":
            # 下界基线：不加任何时序建模，仅对特征做线性映射
            self.temporal = _HEADS["mean_pool"](self._feature_dim, num_classes, cfg)
        else:
            self.temporal = _HEADS[cfg.kind](self._feature_dim, num_classes, cfg)

        if freeze_backbone:
            self.backbone.freeze_backbone(True)
        if backbone_state is not None:
            self.load_backbone_state(backbone_state)

        log.info(
            "构建时序模型：骨干=%s（D=%d）+ 时序头=%s，类别数=%d",
            backbone_arch,
            self._feature_dim,
            cfg.kind,
            num_classes,
        )

    # --- 契约 -------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """输入 ``(B, T, 3, H, W)`` 或单帧 ``(B, 3, H, W)``，输出 ``(B, T, D)``。"""
        if x.ndim == 4:  # 单帧 -> 补时间维
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ModelError(
                f"时序模型输入应为 (B, T, 3, H, W) 或 (B, 3, H, W)，实际 {tuple(x.shape)}"
            )
        batch, steps = x.shape[0], x.shape[1]
        flat = x.reshape(batch * steps, *x.shape[2:])
        feats = self.backbone.embed(flat)  # (B*T, 1, D)
        return feats.reshape(batch, steps, self._feature_dim)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.embed(x))

    def backbone_modules(self) -> list[nn.Module]:
        return [self.backbone]

    # --- 两级训练辅助 -----------------------------------------------------
    def load_backbone_state(self, state: dict[str, torch.Tensor] | str | Path) -> None:
        """从 checkpoint 或 state_dict 载入骨干权重。

        兼容两种存法：
            * 完整 checkpoint 字典（含 ``state`` 键）；
            * 直接是 state_dict（键可能带 ``backbone.`` 前缀，会自动剥离）。
        """
        payload: Any = state
        if isinstance(state, (str, Path)):
            path = Path(state)
            if not path.exists():
                raise ModelError(f"骨干权重不存在：{path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state" in payload and isinstance(payload["state"], dict):
            payload = payload["state"]
        if not isinstance(payload, dict):
            raise ModelError(f"无法识别的骨干权重格式：{type(payload).__name__}")

        cleaned: dict[str, torch.Tensor] = {}
        for key, value in payload.items():
            new_key = key
            for prefix in ("backbone.", "module.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned[new_key] = value

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if missing:
            log.warning("骨干加载：缺少 %d 个参数（前 3 个：%s）", len(missing), missing[:3])
        if unexpected:
            log.warning("骨干加载：多余 %d 个参数（前 3 个：%s）", len(unexpected), unexpected[:3])
