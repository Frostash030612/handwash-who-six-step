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
from handwash.core.temporal import window_starts
from handwash.errors import DataError, TrainingError
from handwash.logging import get_logger

__all__ = [
    "build_manifest_loader",
    "clip_collate",
    "describe_device",
    "group_records_by_clip",
    "maybe_enable_tf32",
    "resolve_device",
    "resolve_split_records",
]

log = get_logger(__name__)


# ============================================================================


# ============================================================================
def resolve_device(spec: str = "auto") -> torch.device:

    key = str(spec).strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if key.startswith("cuda") and not torch.cuda.is_available():
        raise TrainingError(
            "CUDA was requested but is unavailable in this environment",
            hint="Check the GPU driver and PyTorch CUDA installation, or set runtime.device=cpu.",
        )
    if key.startswith("cuda:"):
        try:
            index = int(key.split(":", 1)[1])
        except ValueError as exc:
            raise TrainingError(f"Invalid CUDA device index: {spec!r}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise TrainingError(
                f"CUDA device index is out of range: {index}",
                hint=f"Available CUDA device count: {torch.cuda.device_count()}.",
            )
    if key == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise TrainingError("MPS was requested but is unavailable in this environment")
    try:
        return torch.device(key)
    except (RuntimeError, ValueError) as exc:
        raise TrainingError(f"Invalid runtime device: {spec!r}", hint=str(exc)) from exc


def describe_device(device: torch.device) -> dict[str, Any]:

    info: dict[str, Any] = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(device))
        info["cuda_total_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / (1024**3), 2
        )
    return info


def maybe_enable_tf32(device: torch.device, *, enable: bool = True) -> None:

    if device.type != "cuda":
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        log.info(
            "TF32 %s (affects CUDA matrix multiplication precision only)", "enabled" if enable else "disabled"
        )
    except AttributeError:
        pass


# ============================================================================


# ============================================================================
def resolve_split_records(
    rc: ResolvedConfig,
    *,
    split: Split | str,
    allow_synthetic: bool = False,
    dataset_name: str | None = None,
) -> list[FrameRecord]:

    from handwash.data.dataset import load_split_records
    from handwash.io.manifest import read_manifest
    from handwash.paths import DEFAULT_MANIFEST_RELPATH, resolve_relative

    name = dataset_name or str(rc.dataset.name)
    spec = rc.dataset_spec(name)
    path = resolve_relative(spec.get("manifest") or DEFAULT_MANIFEST_RELPATH)

    if path.exists():
        records = read_manifest(path)
        if name == "synthetic" and allow_synthetic and records:
            frames_dir = resolve_relative(spec.get("frames_dir") or path.parent / "frames")
            missing_image = False
            for rec in records:
                image_path = Path(rec.image_path)
                if not (image_path if image_path.is_absolute() else frames_dir / image_path).is_file():
                    missing_image = True
                    break
            if missing_image:
                log.warning("Synthetic image paths do not match the manifest; regenerating: %s", path)
                return _synthetic_records(rc, split=str(split), manifest_path=path, regenerate=True)
        subset = [rec for rec in records if rec.split == str(split)]
        if subset:
            return subset
        if not allow_synthetic or name != "synthetic":
            raise DataError(
                f"No matching dataset in the manifest: dataset={name} split={split} records: {path}",
                hint="Use the same dataset configuration during preparation and include the selected split in the manifest.",
            )

    if not allow_synthetic or name != "synthetic":
        raise DataError(
            f"Cannot find dataset={name} split={split} data ({path})",
            hint="Provide a prepared dataset and a matching manifest first.",
        )

    log.warning(
        "Real data not found (%s); using SYNTHETIC data for pipeline checks only. Do not report these as model results.",
        path,
    )
    return _synthetic_records(rc, split=str(split), manifest_path=path)


def _synthetic_records(
    rc: ResolvedConfig, *, split: str, manifest_path: Path, regenerate: bool = False
) -> list[FrameRecord]:

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
        raise DataError(f"Synthetic dataset is missing split={split}")
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
                "No complete temporal windows are available",
                hint=(
                    f"model.temporal.window={rc.model.temporal.window} exceeds the frame count of the shortest clip; "
                    "reduce the window or provide more frames per clip."
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
        collate_fn=collate_samples,
        drop_last=False,
    )


def group_records_by_clip(records: Sequence[FrameRecord]) -> dict[str, list[FrameRecord]]:

    buckets: dict[str, list[FrameRecord]] = defaultdict(list)
    for rec in records:
        buckets[rec.clip_id].append(rec)
    for key in buckets:
        buckets[key].sort(key=lambda r: r.frame_index)
    return dict(buckets)


class ClipBatchSampler(torch.utils.data.Sampler[list[int]]):
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
        window = 16
        self._by_clip = by_clip
        self._index_of = index_of
        self.clips_per_batch = max(1, int(clips_per_batch))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._window = window
        self._stride = window

    def set_window(self, window: int, stride: int | None = None) -> ClipBatchSampler:
        """Set temporal window and stride from the resolved model configuration."""
        self._window = max(1, int(window))
        self._stride = self._window if stride is None else int(stride)
        if self._stride < 1 or self._stride > self._window:
            raise DataError("Temporal window stride must be in 1..window")
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

    if not batch:
        raise DataError("clip_collate received an empty batch")
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
            f"Inconsistent window lengths within a batch: {sorted(lengths)}",
            hint="Check that ClipBatchSampler groups samples by window length.",
        )

    return {
        "image": torch.stack(images, dim=0),  # (B, T, 3, H, W)
        "label_index": torch.stack(label_groups, dim=0),  # (B, T)
        "frame_index": torch.stack(index_groups, dim=0),  # (B, T)
        "clip_id": clip_ids,
    }


def label_names(rc: ResolvedConfig) -> list[str]:

    return [step.value for step in get_label_space(rc.label_space).labels]
