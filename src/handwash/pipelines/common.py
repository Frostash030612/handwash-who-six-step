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
from handwash.core.temporal import window_starts
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
    if key.startswith("cuda:"):
        try:
            index = int(key.split(":", 1)[1])
        except ValueError as exc:
            raise TrainingError(f"CUDA 设备编号无效：{spec!r}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise TrainingError(
                f"CUDA 设备编号超出范围：{index}",
                hint=f"当前可用 CUDA 设备数：{torch.cuda.device_count()}。",
            )
    if key == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise TrainingError("配置要求使用 MPS，但当前环境不可用")
    try:
        return torch.device(key)
    except (RuntimeError, ValueError) as exc:
        raise TrainingError(f"运行设备配置无效：{spec!r}", hint=str(exc)) from exc


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
    allow_synthetic: bool = False,
    dataset_name: str | None = None,
) -> list[FrameRecord]:
    """取某个 split 的帧记录。

    合成数据只允许由显式 ``dataset.name=synthetic`` 配置触发；真实数据
    manifest 缺失、为空或缺少指定 split 时一律报错，不做静默替换。
    """
    from handwash.data.dataset import load_split_records  # noqa: F401  （供外部按需导入）
    from handwash.io.manifest import read_manifest
    from handwash.paths import DEFAULT_MANIFEST_RELPATH, resolve_relative

    name = dataset_name or str(rc.dataset.name)
    spec = rc.dataset_spec(name)
    path = resolve_relative(spec.get("manifest") or DEFAULT_MANIFEST_RELPATH)

    if path.exists():
        records = read_manifest(path)
        if name == "synthetic" and allow_synthetic and records:
            # 旧版合成 manifest 把 frames/ 也写进 image_path，训练时会变成
            # frames/frames/<clip>/<frame>.jpg。合成数据是可重建产物，自动修复。
            frames_dir = resolve_relative(spec.get("frames_dir") or path.parent / "frames")
            missing_image = False
            for rec in records:
                image_path = Path(rec.image_path)
                if not (image_path if image_path.is_absolute() else frames_dir / image_path).is_file():
                    missing_image = True
                    break
            if missing_image:
                log.warning("合成数据的图像路径与 manifest 不一致，正在重新生成：%s", path)
                return _synthetic_records(rc, split=str(split), manifest_path=path, regenerate=True)
        subset = [rec for rec in records if rec.split == str(split)]
        if subset:
            return subset
        if not allow_synthetic or name != "synthetic":
            raise DataError(
                f"manifest 中没有 dataset={name} split={split} 的记录：{path}",
                hint="确认 prepare 使用了相同数据集配置，且指定 split 已写入 manifest。",
            )

    if not allow_synthetic or name != "synthetic":
        raise DataError(
            f"找不到 dataset={name} split={split} 的数据（{path}）",
            hint="先运行 handwash prepare，或见 docs/DATA.md 下载数据集。",
        )

    log.warning(
        "未找到真实数据（%s），本次使用**合成数据**：结果仅供验证代码链路，不能写进报告",
        path,
    )
    return _synthetic_records(rc, split=str(split), manifest_path=path)


def _synthetic_records(
    rc: ResolvedConfig, *, split: str, manifest_path: Path, regenerate: bool = False
) -> list[FrameRecord]:
    """生成（或复用）合成数据集的记录，并保证三种 split 是同一批文件。"""
    from handwash.data.synthetic import write_synthetic_dataset
    from handwash.io.manifest import read_manifest

    root = manifest_path.parent
    manifest = manifest_path
    if regenerate or not manifest.exists():
        frames, _ = write_synthetic_dataset(
            root, num_clips=24, frames_per_clip=18, image_size=96, seed=rc.runtime.seed
        )
        subset = [record for record in frames if record.split == split]
        if subset:
            return subset
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
    image_root: str | Path | None = None,
    label_space_name: str | None = None,
) -> DataLoader:
    """构造 DataLoader。

    ``for_clip_mode=True`` 时使用按视频分组的 batch 采样器：
    一个 batch = 一段视频的连续帧，因此时序模型能看到真实的时间上下文。
    """
    from handwash.data.dataset import build_dataset, collate_samples

    dataset = build_dataset(
        rc,
        records,
        split=split,
        image_root=image_root,
        label_space_name=label_space_name,
        synchronize_clip_augment=for_clip_mode and str(split) == "train",
    )
    if for_clip_mode:
        sampler = ClipBatchSampler(
            records,
            clips_per_batch=max(1, rc.train.batch_size // max(1, rc.model.temporal.window)),
            shuffle=shuffle,
            seed=rc.runtime.seed,
        ).set_window(rc.model.temporal.window, rc.model.temporal.stride)
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

    实现：按配置的窗口和步长切分每段视频，最后保留较短窗口以覆盖片段末尾；一个 batch
    内同一视频最多出现一次，避免把同一视频的多个窗口拼成一段虚假的长序列。
    """

    def __init__(
        self,
        records: Sequence[FrameRecord],
        *,
        clips_per_batch: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.windows: list[tuple[str, list[int]]] = []
        self.batches: list[list[int]] = []
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
        self.epoch = 0
        self._window = window
        self._stride = window

    def set_window(self, window: int, stride: int | None = None) -> "ClipBatchSampler":
        """Set temporal window and stride from the resolved model configuration."""
        self._window = max(1, int(window))
        self._stride = self._window if stride is None else int(stride)
        if self._stride < 1 or self._stride > self._window:
            raise DataError("时序窗口步长必须在 1..window 范围内")
        self._rebuild()
        return self

    def _rebuild(self) -> None:
        self.windows = []
        by_length: dict[int, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
        for clip_id, recs in sorted(self._by_clip.items()):
            for start in window_starts(len(recs), self._window, self._stride):
                chunk = recs[start : start + self._window]
                if not chunk:
                    continue
                indices = [self._index_of[(r.clip_id, r.frame_index)] for r in chunk]
                self.windows.append((clip_id, indices))
                by_length[len(indices)][clip_id].append(indices)

        self.batches = []
        # Collation stacks tensors, so each batch combines equal-length windows.
        for length in sorted(by_length):
            queues = by_length[length]
            while any(queues.values()):
                active = [clip_id for clip_id in sorted(queues) if queues[clip_id]]
                selected = active[: self.clips_per_batch]
                batch: list[int] = []
                for clip_id in selected:
                    batch.extend(queues[clip_id].pop(0))
                if batch:
                    self.batches.append(batch)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
        self.epoch += 1
        for idx in order:
            yield self.batches[idx]


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
            hint="请检查 ClipBatchSampler 是否按窗口长度分组。",
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
