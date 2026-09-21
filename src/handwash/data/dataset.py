"""基于 manifest 的 torch Dataset。

分层位置：L2。允许 import torch（数据层是深度学习边界的一部分），
但**禁止** import 任何模型实现。

对外契约
--------
``FrameManifestDataset.__getitem__`` 返回::

    {
        "image":       torch.Tensor  (3, H, W) float32，已归一化
        "label_index": int           —— 标签空间通道下标
        "clip_id":     str
        "frame_index": int
    }

其中 ``label_index`` 由 ``LabelSpace`` 统一给出：这是"标签错位"这类
神隐 bug 的唯一防线（见 core/labels.py 的说明）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from handwash.core.config import ResolvedConfig
from handwash.core.labels import LabelSpace, get_label_space
from handwash.core.schema import FrameRecord, Split
from handwash.errors import ConfigError, ManifestError
from handwash.io.manifest import filter_manifest, read_manifest
from handwash.logging import get_logger
from handwash.paths import DEFAULT_MANIFEST_RELPATH, resolve_relative

__all__ = ["FrameManifestDataset", "build_dataset", "collate_samples", "label_distribution"]

log = get_logger(__name__)


def _import_torch():
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:  # pragma: no cover - 环境缺 torch
        raise ConfigError(
            "本功能需要 PyTorch",
            hint="conda env update -f environment.yml 后重试。",
        ) from exc
    return torch, Dataset


torch, _TorchDataset = _import_torch()


class FrameManifestDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """帧级分类数据集。

    Parameters
    ----------
    records:
        该 split 的帧记录（来自 ``io.manifest``）。
    image_root:
        图像路径的前缀（manifest 里存的是相对路径，便于整包搬移数据集）。
    label_space:
        标签空间；决定 ``label_index`` 的取值。
    transform:
        ``data.transforms.build_transforms`` 的产物。
    """

    def __init__(
        self,
        records: Sequence[FrameRecord],
        *,
        image_root: str | Path,
        label_space: LabelSpace,
        transform=None,
    ) -> None:
        if not records:
            raise ManifestError("数据集为空：请检查该 split 的 manifest 是否生成成功")
        self.records = list(records)
        self.image_root = Path(image_root)
        self.label_space = label_space
        self.transform = transform
        # 预先算好下标，避免每个 batch 都做字典查找
        self._indices = [label_space.to_index(rec.label) for rec in self.records]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_classes(self) -> int:
        return len(self.label_space)

    def __getitem__(self, index: int) -> dict[str, object]:
        from PIL import Image

        rec = self.records[index]
        path = self._resolve_image_path(rec.image_path)
        try:
            with Image.open(path) as handle:
                image = np.asarray(handle.convert("RGB"))
        except FileNotFoundError as exc:
            raise ManifestError(
                f"帧图像不存在：{path}",
                hint="manifest 与图像目录不匹配；请用同一次 handwash prepare 的产物。",
            ) from exc

        if self.transform is not None:
            image = self.transform(image)
        tensor = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
        return {
            "image": tensor,
            "label_index": int(self._indices[index]),
            "clip_id": rec.clip_id,
            "frame_index": int(rec.frame_index),
        }

    def _resolve_image_path(self, raw: str) -> Path:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        return self.image_root / candidate

    def label_distribution(self) -> dict[str, int]:
        """类别分布：写进日志，用于发现"某一步只有 3 帧"这类问题。"""
        counter: Counter[str] = Counter(rec.label.value for rec in self.records)
        return dict(sorted(counter.items()))

    def clip_ids(self) -> list[str]:
        return sorted({rec.clip_id for rec in self.records})


def collate_samples(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    """把若干样本拼成 batch。

    为什么自己写 collate：需要同时保留字符串字段（clip_id）与张量字段，
    默认 collate 会把字符串也拼成 list-of-list，下游用起来很别扭。

    形状约定：**与 ``pipelines.common.clip_collate`` 完全对齐**，即
    ``image: (B, T, 3, H, W)``、``label_index: (B, T)``、``frame_index: (B, T)``。
    帧模式下每个样本自身构成一个长度为 1 的时间步（T=1）。
    这样训练/评估/推理三处只有一套形状逻辑，不会出现"Dataloader 出来的
    label_index 有时是二维有时是一维"这种最消耗调试时间的问题。
    """
    if not batch:
        raise ManifestError("collate_samples 收到空 batch")
    images = torch.stack([item["image"] for item in batch], dim=0).unsqueeze(1)  # (B, 1, 3, H, W)
    labels = torch.as_tensor([int(item["label_index"]) for item in batch], dtype=torch.long).unsqueeze(1)
    return {
        "image": images,
        "label_index": labels,  # (B, 1)
        "frame_index": torch.as_tensor(
            [int(item["frame_index"]) for item in batch], dtype=torch.long
        ).unsqueeze(1),  # (B, 1)
        "clip_id": [str(item["clip_id"]) for item in batch],
    }


def label_distribution(records: Sequence[FrameRecord]) -> dict[str, int]:
    """便捷函数：帧记录的类别分布。"""
    counter: Counter[str] = Counter(rec.label.value for rec in records)
    return dict(sorted(counter.items()))


def build_dataset(
    rc: ResolvedConfig,
    records: Sequence[FrameRecord],
    *,
    split: Split | str = "train",
    image_root: str | Path | None = None,
) -> FrameManifestDataset:
    """工厂：按配置构造某个 split 的数据集。

    训练用 ``mode="train"``（带增强），评估必须用 ``mode="eval"``。
    """
    from handwash.data.transforms import build_transforms

    mode = "train" if split == "train" else "eval"
    transform = build_transforms(
        rc.model,
        mode=mode,
        augment=rc.train.augment,
        seed=rc.runtime.seed + (0 if mode == "train" else 10_000),
    )
    space = get_label_space(rc.label_space)
    root = Path(image_root) if image_root is not None else _image_root_from(rc)
    dataset = FrameManifestDataset(records, image_root=root, label_space=space, transform=transform)
    log.info(
        "数据集已构建：split=%s，%d 帧 / %d 段视频，%d 类",
        split,
        len(dataset),
        len(dataset.clip_ids()),
        dataset.num_classes,
    )
    return dataset


def _image_root_from(rc: ResolvedConfig) -> Path:
    """从配置推断图像根目录。

    约定（务必与 ``pipelines/prepare.py`` 的写盘逻辑一致）：
    manifest 里的 ``image_path`` 是**相对数据集根目录**的路径
    （例如 ``frames/<clip_id>/00003.jpg``），因此这里解析 ``root``。
    使用相对路径的好处：整个数据目录可以搬走/挂载到别的盘，manifest 不用改。
    """
    spec = rc.dataset_spec()
    return resolve_relative(spec.get("root") or rc.dataset.root)


def _project_root() -> Path:
    from handwash.paths import PROJECT_ROOT

    return PROJECT_ROOT


def load_split_records(
    rc: ResolvedConfig,
    *,
    split: Split | str,
    manifest_path: str | Path | None = None,
) -> list[FrameRecord]:
    """读取并筛选某个 split 的帧记录（评估与训练共用同一入口）。"""
    spec = rc.dataset_spec()
    path = resolve_relative(manifest_path or spec.get("manifest") or DEFAULT_MANIFEST_RELPATH)
    records = read_manifest(path)
    subset = filter_manifest(records, split=str(split))
    if not subset:
        raise ManifestError(
            f"split={split} 在 manifest 中没有记录：{path}",
            hint="确认 prepare 阶段生成了该 split；val/test 为空时训练无法早停。",
        )
    return subset
