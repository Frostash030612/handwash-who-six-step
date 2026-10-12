"""Small spatial-attention adapter for an existing YOLO classification head.

The original classifier averages every spatial location equally. This module
starts with exactly the same average and learns where to put more weight. It
keeps the pretrained convolutional backbone and seven-class order intact.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from handwash.pipelines.exp_demo import EXP_CLASS_NAMES, file_sha256

ADAPTER_FORMAT = "yolo-spatial-attention/1"


class SpatialAttentionPool(nn.Module):
    """Pool a feature map with a learned spatial distribution.

    Zero attention logits make the initial output exactly equal to global
    average pooling. A residual gate keeps the original average available
    while the adapter learns a spatial focus.
    """

    def __init__(self, channels: int, *, initial_gate_logit: float = -1.38629436) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.attention.weight)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("Expected a [batch, channels, height, width] feature map")
        batch, _channels, height, width = features.shape
        weights = self.attention(features).reshape(batch, 1, height * width)
        weights = weights.softmax(dim=-1).reshape(batch, 1, height, width)
        focused = (features * weights).sum(dim=(-2, -1), keepdim=True)
        uniform = features.mean(dim=(-2, -1), keepdim=True)
        gate = self.gate_logit.sigmoid()
        return uniform + gate * (focused - uniform)


def install_spatial_attention(yolo, *, adapter_path: str | Path | None = None):
    """Patch a YOLO classifier in memory and optionally load a bound adapter.

    No base checkpoint is modified. The saved adapter includes the SHA-256 of
    the exact YOLO weight used during training. Install it before the first
    ``predict`` call, when Ultralytics creates its inference backend.
    """

    if yolo.task != "classify":
        raise ValueError("Spatial attention requires a YOLO classification model")
    if getattr(yolo, "predictor", None) is not None:
        raise ValueError("Install spatial attention before the first YOLO.predict call")
    names = tuple(str(yolo.names.get(index, "")) for index in range(len(yolo.names)))
    if names != EXP_CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {names}")
    head = yolo.model.model[-1]
    if not isinstance(head.pool, nn.AdaptiveAvgPool2d) or not isinstance(head.linear, nn.Linear):
        raise ValueError("YOLO classification head no longer has the expected pool/linear structure")
    adapter = SpatialAttentionPool(head.linear.in_features)
    if adapter_path is not None:
        payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != ADAPTER_FORMAT
            or tuple(payload.get("class_names", ())) != EXP_CLASS_NAMES
        ):
            raise ValueError("Invalid spatial-attention adapter checkpoint")
        if payload.get("base_sha256") != file_sha256(yolo.ckpt_path):
            raise ValueError("Spatial-attention adapter does not match the YOLO checkpoint")
        adapter.load_state_dict(payload["pool_state_dict"], strict=True)
        head.linear.load_state_dict(payload["linear_state_dict"], strict=True)
    adapter = adapter.to(device=head.linear.weight.device, dtype=head.linear.weight.dtype)
    head.pool = adapter
    return adapter
