"""Pipeline 共享设施：设备、DataLoader、数据集解析、批次构造。

这个文件存在的唯一目的：让 train / evaluate / infer 三处**用同一套**数据装配逻辑。
只要三处各自写一遍，"训练时增强开了、评估时忘了关"这类事故必然发生。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from handwash.core.config import ResolvedConfig
from handwash.core.labels import get_label_space
from handwash.core.schema import FrameRecord, Split
from handwash.core.seeding import worker_init_fn
from handwash.errors import DataError, TrainingError
from handwash.logging import get_logger

__all__ = [
    "resolve_device",
    "describe_device",
    "build_manifest_loader",
    "clip_collate",
    "group_records_by_clip",
    "resolve_split_records",
    "maybe_enable_tf32",
]

log = get_logger(__name__)


# ============================================================================
# 设备
# ============================================================================
def resolve_device(spec: str = "auto") -> torch.device:
    """把配置里的 ``device`` 翻译成 ``torch.device``。

    ``auto``：有 CUDA 用 cuda，有 MPS 用 mps，否则 cpu。
    注意：**不要静默把用户指定的 cuda 降级为 cpu** —— 那会让实验悄悄变慢十倍。
    """
    key = str(spec).strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if key.startswith("cuda") and not torch.cuda.is_available():
        raise TrainingError(
            "配置要求使用 CUDA，但当前环境不可用",
            hint="检查显卡驱动与 pytorch-cuda 是否装对；或改 runtime.device=cpu。",
        )
    return torch.device(key)


def describe_device(device: torch.device) -> dict[str, Any]:
    """设备摘要，写进实验日志（报告里要交代算力条件）。"""
    info: dict[str, Any] = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(device))
        info["cuda_total_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / (1024**3), 2
        )
    return info


def maybe_enable_tf32(device: torch.device, *, enable: bool = True) -> None:
    """在 Ampere 及以上开启 TF32：矩阵乘更快，精度损失对本任务可接受。"""
    if device.type != "cuda":
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        log.info("TF32 已%s（仅影响 CUDA 矩阵乘精度）", "开启" if enable else "关闭")
    except AttributeError:  # pragma: no cover - 老版本 torch
        pass


# ============================================================================
# 数据
# ============================================================================
def resolve_split_records(
    rc: ResolvedConfig,
    *,
    split: Split | str,
    allow_synthetic: bool = True,
) -> list[FrameRecord]:
    """取某个 split 的帧记录。

    回退逻辑（**顺序固定，不要改**）：
        1. 若配置的 manifest 存在 -> 读它；
        2. 否则若 ``allow_synthetic`` -> 现场生成一份合成数据集（冒烟训练/演示用）；
        3. 否则报错并指向 docs/DATA.md。

    回退到合成数据时必须打日志警告，避免组员误以为在跑真实数据。
    """
    from handwash.data.dataset import load_split_records  # noqa: F401  （供外部按需导入）
    from handwash.io.manifest import read_manifest
    from handwash.paths import DEFAULT_MANIFEST_RELPATH, resolve_relative

    spec = rc.dataset_spec()
    path = resolve_relative(spec.get("manifest") or DEFAULT_MANIFEST_RELPATH)

    if path.exists():
        records = read_manifest(path)
        subset = [rec for rec in records if rec.split == str(split)]
        if subset:
            return subset
        log.warning("manifest 中没有 split=%s 的记录：%s", split, path)

    if not allow_synthetic:
        raise DataError(
            f"找不到 split={split} 的数据（{path}）",
            hint="先运行 handwash prepare，或见 docs/DATA.md 下载数据集。",
        )

    log.warning(
        "未找到真实数据（%s），本次使用**合成数据**：结果仅供验证代码链路，不能写进报告",
        path,
    )
    return _synthetic_records(rc, split=str(split))


def _synthetic_records(rc: ResolvedConfig, *, split: str) -> list[FrameRecord]:
    """生成（或复用）合成数据集的记录，并保证三种 split 是同一批文件。"""
    from handwash.data.synthetic import write_synthetic_dataset
    from handwash.paths import PROJECT_ROOT
    from handwash.io.manifest import read_manifest

    root = PROJECT_ROOT / "data" / "interim" / "synthetic"
    manifest = root / "manifest.csv"
    if not manifest.exists():
        write_synthetic_dataset(root, num_clips=24, frames_per_clip=18, image_size=96, seed=rc.runtime.seed)
    records = read_manifest(manifest)
    subset = [rec for rec in records if rec.split == split]
    if not subset:
        raise DataError(f"合成数据集缺少 split={split}")
    return subset


def build_manifest_loader(
    rc: ResolvedConfig,
    records: Sequence[FrameRecord],
    *,
    split: Split | str,
    shuffle: bool = False,
    for_clip_mode: bool = False,
) -> DataLoader:
    """构造 DataLoader。

    ``for_clip_mode=True`` 时使用按视频分组的 batch 采样器：
    一个 batch = 一段视频的连续帧，因此时序模型能看到真实的时间上下文。
    """
    from handwash.data.dataset import build_dataset, collate_samples

    dataset = build_dataset(rc, records, split=split)
    if for_clip_mode:
        sampler = ClipBatchSampler(
            records,
            clips_per_batch=max(1, rc.train.batch_size // max(1, rc.model.temporal.window)),
            shuffle=shuffle,
            seed=rc.runtime.seed,
        ).set_window(rc.model.temporal.window)
        if len(sampler) == 0:
            raise DataError(
                "按时序窗口切分后没有任何完整窗口",
                hint=(
                    f"model.temporal.window={rc.model.temporal.window} 大于最短视频的帧数；"
                    "请降低 window，或提高抽帧后每段视频的帧数。"
                ),
            )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=clip_collate,
            num_workers=rc.runtime.num_workers,
            pin_memory=rc.runtime.pin_memory and torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    return DataLoader(
        dataset,
        batch_size=rc.train.batch_size if shuffle else rc.train.eval_batch_size,
        shuffle=shuffle,
        num_workers=rc.runtime.num_workers,
        pin_memory=rc.runtime.pin_memory and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        collate_fn=collate_samples,  # 与 clip_collate 保持同一套形状契约
        drop_last=False,
    )


def group_records_by_clip(records: Sequence[FrameRecord]) -> dict[str, list[FrameRecord]]:
    """按 clip_id 分组，组内按 frame_index 升序（时间顺序是硬要求）。"""
    buckets: dict[str, list[FrameRecord]] = defaultdict(list)
    for rec in records:
        buckets[rec.clip_id].append(rec)
    for key in buckets:
        buckets[key].sort(key=lambda r: r.frame_index)
    return dict(buckets)


class ClipBatchSampler(torch.utils.data.Sampler[list[int]]):
    """把一个 batch 组织成"若干段完整视频片段"的采样器。

    为什么需要它：时序模型（GRU/TCN）需要连续帧。如果按帧随机采样，
    batch 里混着不同视频的随机帧，时序头学到的是噪声。

    实现：以窗口为单位切分每段视频，每个 batch 取 ``clips_per_batch`` 个窗口。
    最后一个不完整窗口会被丢弃（避免 padding 干扰因果卷积）。
    """

    def __init__(
        self,
        records: Sequence[FrameRecord],
        *,
        clips_per_batch: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.windows: list[list[int]] = []
        index_of: dict[tuple[str, int], int] = {
            (rec.clip_id, rec.frame_index): i for i, rec in enumerate(records)
        }
        by_clip = group_records_by_clip(records)
        window = 16  # 会被 set_window 覆盖
        self._by_clip = by_clip
        self._index_of = index_of
        self.clips_per_batch = max(1, int(clips_per_batch))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._window = window

    def set_window(self, window: int) -> "ClipBatchSampler":
        """由外部显式设置窗口长度（来自 model.temporal.window）。"""
        self._window = max(1, int(window))
        self._rebuild()
        return self

    def _rebuild(self) -> None:
        self.windows = []
        for _, recs in sorted(self._by_clip.items()):
            for start in range(0, len(recs) - self._window + 1, self._window):
                chunk = recs[start : start + self._window]
                if len(chunk) == self._window:
                    self.windows.append([self._index_of[(r.clip_id, r.frame_index)] for r in chunk])

    def __len__(self) -> int:
        if not self.windows:
            return 0
        return max(1, (len(self.windows) + self.clips_per_batch - 1) // self.clips_per_batch)

    def __iter__(self):
        order = list(range(len(self.windows)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)
        for i in range(0, len(order), self.clips_per_batch):
            group = order[i : i + self.clips_per_batch]
            batch: list[int] = []
            for idx in group:
                batch.extend(self.windows[idx])
            if batch:
                yield batch


def clip_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把"每组恰好一个窗口"的样本拼成 ``(B, T, 3, H, W)``。

    约定：``ClipBatchSampler`` 保证每个窗口长度一致，因此可以直接 stack。
    长度不一致会明确报错，而不是靠 padding 掩盖问题。
    """
    if not batch:
        raise DataError("clip_collate 收到空 batch")
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in batch:
        clips[str(item["clip_id"])].append(item)

    images: list[torch.Tensor] = []
    label_groups: list[torch.Tensor] = []
    index_groups: list[torch.Tensor] = []
    lengths: set[int] = set()
    clip_ids: list[str] = []
    for clip_id, items in clips.items():
        items.sort(key=lambda x: int(x["frame_index"]))
        lengths.add(len(items))
        clip_ids.append(clip_id)
        images.append(torch.stack([item["image"] for item in items], dim=0))
        label_groups.append(torch.as_tensor([int(item["label_index"]) for item in items], dtype=torch.long))
        index_groups.append(torch.as_tensor([int(item["frame_index"]) for item in items], dtype=torch.long))

    if len(lengths) != 1:
        raise DataError(
            f"同一 batch 内窗口长度不一致：{sorted(lengths)}",
            hint="请检查 ClipBatchSampler 是否正确丢弃了不完整窗口。",
        )

    return {
        "image": torch.stack(images, dim=0),  # (B, T, 3, H, W)
        "label_index": torch.stack(label_groups, dim=0),  # (B, T)
        "frame_index": torch.stack(index_groups, dim=0),  # (B, T)
        "clip_id": clip_ids,
    }


def label_names(rc: ResolvedConfig) -> list[str]:
    """当前标签空间的类别名（指标表与混淆矩阵的坐标轴）。"""
    return [step.value for step in get_label_space(rc.label_space).labels]
