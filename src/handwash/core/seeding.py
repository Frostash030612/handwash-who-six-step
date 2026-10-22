"""Deterministic random-number initialization for retained training utilities."""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np

__all__ = ["DEFAULT_SEED", "seed_everything", "worker_init_fn"]

DEFAULT_SEED: Final[int] = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:

    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer; received {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):  # pragma: no cover
                pass
    except ImportError:
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:

    try:
        from torch.utils.data import get_worker_info

        info = get_worker_info()
    except ImportError:  # pragma: no cover - torch is optional for core utilities
        info = None
    base = int(os.environ.get("PYTHONHASHSEED", str(DEFAULT_SEED)))
    seed = int(info.seed if info is not None else base + worker_id)
    np_seed = seed % (2**32)
    np.random.seed(np_seed)
    random.seed(seed)

    # Project transforms keep a NumPy Generator rather than relying on global
    # NumPy state. Reseed those generators per worker as well; otherwise forked
    # workers inherit identical augmentation streams.
    if info is not None:
        shared_generators: dict[int, np.random.Generator] = {}

        def reseed(transform) -> None:
            generator = getattr(transform, "rng", None)
            if isinstance(generator, np.random.Generator):
                key = id(generator)
                if key not in shared_generators:
                    shared_generators[key] = np.random.default_rng(seed)
                transform.rng = shared_generators[key]
            for child in getattr(transform, "steps", ()):
                reseed(child)

        reseed(getattr(info.dataset, "transform", None))
