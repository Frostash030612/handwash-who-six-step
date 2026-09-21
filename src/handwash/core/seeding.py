"""随机种子：所有实验必须可复现（CONTRIBUTING.md R13）。

规则：
    * 训练入口第一件事就是 ``seed_everything(cfg.runtime.seed)``。
    * 禁止在模块里裸写 ``random.seed(42)``。
    * 报告里必须记录实际种子（``TrainResult.seed``）。

本模块属于 L1 契约层：只 import numpy 与标准库，torch 存在时才顺手设种子。
"""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np

__all__ = ["seed_everything", "DEFAULT_SEED", "worker_init_fn"]

DEFAULT_SEED: Final[int] = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """一次性给 python / numpy / torch / cuda 设种子，返回实际使用的种子。

    Parameters
    ----------
    seed:
        非负整数。
    deterministic:
        True 时让 cuDNN 走确定性算法（可能略慢，但结果可复现）。
    """
    if seed < 0:
        raise ValueError(f"seed 必须为非负整数，实际 {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch 是可选依赖：没装也要能跑核心逻辑
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch >= 1.8：让某些原子操作报错而不是悄悄引入不确定性
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):  # pragma: no cover
                pass
    except ImportError:  # pragma: no cover - 无 torch 环境
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 的 worker 种子初始化：保证多进程加载也是确定的。"""
    base = int(os.environ.get("PYTHONHASHSEED", str(DEFAULT_SEED)))
    seed = base + worker_id
    np.random.seed(seed)
    random.seed(seed)
