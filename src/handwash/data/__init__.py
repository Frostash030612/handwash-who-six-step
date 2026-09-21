"""数据层：把磁盘上的帧图像变成模型可以吃的张量。

分层位置：L2。允许依赖 core 与 io，禁止依赖 models / pipelines。

公开接口（pipeline 只认这几个）
--------------------------------
``FrameManifestDataset``   帧级分类数据集（torch Dataset）
``build_transforms``       训练/评估预处理流水线
``build_dataset``          按 config 造数据集的工厂
``SyntheticFrameDataset``  合成数据，用于 30 秒冒烟训练与单元测试
``write_synthetic_dataset``把合成数据落盘成标准目录结构（供端到端演练）
"""

from __future__ import annotations

__all__ = [
    "FrameManifestDataset",
    "SyntheticFrameDataset",
    "build_dataset",
    "build_transforms",
    "write_synthetic_dataset",
    "NormalizeSpec",
    "NORMALIZE_PRESETS",
]

from handwash.data.synthetic import SyntheticFrameDataset, write_synthetic_dataset
from handwash.data.transforms import NORMALIZE_PRESETS, NormalizeSpec, build_transforms

# torch 是可选依赖：没有 torch 时仍可 import 本包（合成数据生成只用 numpy/Pillow）
try:  # pragma: no cover - 取决于环境
    from handwash.data.dataset import FrameManifestDataset, build_dataset
except ImportError as exc:  # pragma: no cover
    _TORCH_ERROR = exc

    def _missing(*_args: object, **_kwargs: object):  # type: ignore[misc]
        raise ImportError(
            "本功能需要 PyTorch：conda env update -f environment.yml 后重试"
        ) from _TORCH_ERROR

    FrameManifestDataset = _missing  # type: ignore[assignment]
    build_dataset = _missing  # type: ignore[assignment]
