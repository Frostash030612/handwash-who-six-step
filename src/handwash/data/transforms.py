"""预处理与数据增强。

为什么不用 torchvision.transforms（CONTRIBUTING.md R17）
--------------------------------------------------------
1. 增强策略是本项目"可比较的实验变量"之一，必须**显式、可测、可复现**；
   黑盒 Compose 里的随机性来源难以在报告里说清楚。
2. 手写实现只用 numpy + Pillow，不引入 torchvision 版本差异带来的行为漂移。
3. 完全相同的代码同时服务 torch 数据集与 numpy 推理路径（demo 端）。

接口契约（**不要改**，改了要同步改 data/dataset.py 与 pipelines/infer.py）
--------------------------------------------------------------------------
``build_transforms(cfg, mode, seed)`` 返回一个可调用对象::

    fn(image: np.ndarray HWC uint8) -> np.ndarray CHW float32

不返回 torch.Tensor：转张量由数据集负责，这样变换层零 torch 依赖、可单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from handwash.core.config import AugmentConfig, ModelConfig
from handwash.errors import ConfigError

__all__ = ["NORMALIZE_PRESETS", "NormalizeSpec", "build_transforms", "Compose", "to_tensor_chw"]


@dataclass(frozen=True, slots=True)
class NormalizeSpec:
    """归一化参数（ImageNet 统计量是迁移学习的默认选择）。"""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.mean, dtype=np.float32).reshape(3, 1, 1),
            np.asarray(self.std, dtype=np.float32).reshape(3, 1, 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"mean": list(self.mean), "std": list(self.std)}


NORMALIZE_PRESETS: dict[str, NormalizeSpec] = {
    "imagenet": NormalizeSpec((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "zero_one": NormalizeSpec((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "minus_one_one": NormalizeSpec((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def get_normalize_spec(name: str) -> NormalizeSpec:
    key = str(name).strip().lower()
    if key not in NORMALIZE_PRESETS:
        raise ConfigError(
            f"未知的归一化方案：{name!r}", hint=f"可用：{sorted(NORMALIZE_PRESETS)}"
        )
    return NORMALIZE_PRESETS[key]


def to_tensor_chw(image: np.ndarray, spec: NormalizeSpec | None = None) -> np.ndarray:
    """HWC uint8/float -> CHW float32（可选归一化）。"""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"期望 HWC 三通道图像，实际 shape={arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:  # 视为 0-255
        arr = arr / 255.0
    tensor = np.transpose(arr, (2, 0, 1))
    if spec is not None:
        mean, std = spec.as_arrays()
        tensor = (tensor - mean) / std
    return np.ascontiguousarray(tensor, dtype=np.float32)


# ============================================================================
# 单步变换
# ============================================================================
class _Transform:
    """变换基类：``__call__(image) -> image``，可组合。"""

    name: str = "transform"

    def __call__(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError


class _Resize(_Transform):
    name = "resize"

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(np.asarray(image))
        # 短边缩放 + 中心裁剪：与分类任务常用做法一致，保证长宽比不被拉伸
        width, height = pil.size
        scale = self.size / min(width, height)
        new_size = (max(self.size, int(round(width * scale))), max(self.size, int(round(height * scale))))
        pil = pil.resize(new_size, Image.BILINEAR)
        left = (pil.size[0] - self.size) // 2
        top = (pil.size[1] - self.size) // 2
        return np.asarray(pil.crop((left, top, left + self.size, top + self.size)))


class _RandomResizedCrop(_Transform):
    name = "random_resized_crop"

    def __init__(self, size: int, scale: tuple[float, float], rng: np.random.Generator) -> None:
        self.size = int(size)
        self.scale = scale
        self.rng = rng

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(np.asarray(image))
        width, height = pil.size
        area = width * height
        for _ in range(10):  # 最多重试 10 次找合法裁剪框
            target_area = area * float(self.rng.uniform(*self.scale))
            aspect = math.exp(float(self.rng.uniform(math.log(3 / 4), math.log(4 / 3))))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                left = int(self.rng.integers(0, width - crop_w + 1))
                top = int(self.rng.integers(0, height - crop_h + 1))
                cropped = pil.crop((left, top, left + crop_w, top + crop_h))
                return np.asarray(cropped.resize((self.size, self.size), Image.BILINEAR))
        return _Resize(self.size)(image)  # 兜底：退回中心裁剪


class _RandomHorizontalFlip(_Transform):
    name = "horizontal_flip"

    def __init__(self, prob: float, rng: np.random.Generator) -> None:
        self.prob = float(prob)
        self.rng = rng

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.prob:
            return np.ascontiguousarray(np.asarray(image)[:, ::-1])
        return np.asarray(image)


class _RandomRotation(_Transform):
    name = "rotation"

    def __init__(self, degrees: float, rng: np.random.Generator) -> None:
        self.degrees = float(degrees)
        self.rng = rng

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        if self.degrees <= 0:
            return np.asarray(image)
        angle = float(self.rng.uniform(-self.degrees, self.degrees))
        pil = Image.fromarray(np.asarray(image)).rotate(angle, resample=Image.BILINEAR, expand=False)
        return np.asarray(pil)


class _ColorJitter(_Transform):
    name = "color_jitter"

    def __init__(self, strength: float, rng: np.random.Generator) -> None:
        self.strength = float(strength)
        self.rng = rng

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.strength <= 0:
            return np.asarray(image)
        arr = np.asarray(image).astype(np.float32)
        # 亮度/对比度/饱和度：三者在 [1-s, 1+s] 内独立采样
        for _ in range(3):
            factor = 1.0 + float(self.rng.uniform(-self.strength, self.strength))
            arr = arr * factor
        if self.strength > 0 and self.rng.random() < 0.3:
            gray = arr.mean(axis=2, keepdims=True)
            arr = 0.8 * arr + 0.2 * gray  # 轻微去色，模拟不同手机白平衡
        return np.clip(arr, 0, 255).astype(np.uint8)


class _GaussianBlur(_Transform):
    name = "gaussian_blur"

    def __init__(self, prob: float, rng: np.random.Generator) -> None:
        self.prob = float(prob)
        self.rng = rng

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.prob <= 0 or self.rng.random() >= self.prob:
            return np.asarray(image)
        from PIL import Image, ImageFilter

        radius = float(self.rng.uniform(0.3, 1.4))
        pil = Image.fromarray(np.asarray(image)).filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(pil)


class Compose(_Transform):
    """顺序执行若干变换。"""

    name = "compose"

    def __init__(self, steps: list[_Transform]) -> None:
        self.steps = steps

    def __call__(self, image: np.ndarray) -> np.ndarray:
        out = np.asarray(image)
        for step in self.steps:
            out = step(out)
        return out

    def __len__(self) -> int:
        return len(self.steps)

    def describe(self) -> list[str]:
        """返回变换名列表，写进实验日志（报告里要说明用了哪些增强）。"""
        return [s.name for s in self.steps]


class _NormalizeTensor(_Transform):
    """最后一步：HWC uint8 -> CHW float32（归一化）。"""

    name = "to_tensor"

    def __init__(self, spec: NormalizeSpec) -> None:
        self.spec = spec

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return to_tensor_chw(image, self.spec)


def build_transforms(
    cfg: ModelConfig,
    *,
    mode: str = "train",
    augment: AugmentConfig | None = None,
    seed: int = 42,
) -> _Transform:
    """按配置构造变换流水线。

    Parameters
    ----------
    cfg:
        模型配置（提供 ``image_size`` 与 ``normalize``）。
    mode:
        ``"train"`` 启用增强；``"eval"`` / ``"infer"`` 只做 resize + 归一化。
        **评估传 eval、推理传 infer**，两者都只做相同的 resize + normalize。
    seed:
        增强随机源种子。训练时用 ``runtime.seed + epoch`` 之类的值即可复现。
    """
    spec = get_normalize_spec(cfg.normalize)
    steps: list[_Transform] = []
    if mode not in ("train", "eval", "infer"):
        raise ConfigError(f"未知的 transforms mode：{mode!r}", hint="允许 train / eval / infer")

    rng = np.random.default_rng(seed)
    if mode == "train":
        aug = augment or AugmentConfig()
        if aug.random_resized_crop:
            steps.append(_RandomResizedCrop(cfg.image_size, tuple(aug.crop_scale), rng))
        else:
            steps.append(_Resize(cfg.image_size))
        if aug.horizontal_flip:
            steps.append(_RandomHorizontalFlip(0.5, rng))
        if aug.rotation_deg > 0:
            steps.append(_RandomRotation(aug.rotation_deg, rng))
        if aug.color_jitter > 0:
            steps.append(_ColorJitter(aug.color_jitter, rng))
        if aug.gaussian_blur > 0:
            steps.append(_GaussianBlur(aug.gaussian_blur, rng))
    else:
        steps.append(_Resize(cfg.image_size))

    steps.append(_NormalizeTensor(spec))
    return Compose(steps)
