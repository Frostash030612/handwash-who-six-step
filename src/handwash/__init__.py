"""handwash —— 六步洗手动作识别与完整性评估。

本文件是**唯一**负责引导导入路径的地方，原因：
    src-layout 下，未安装包时直接执行 ``python scripts/train.py`` 会找不到
    ``handwash``。与其让每个脚本各写一份 sys.path 魔法，不如集中在这里。

分层规则（详见 docs/ARCHITECTURE.md）：
    L0  handwash（顶层：paths / logging，仅标准库）
    L1  handwash.core（契约层：schema / labels / config / metrics / protocol）
    L2  handwash.io, handwash.data
    L3  handwash.models, handwash.pipelines
    L4  handwash.cli, scripts/, apps/

    依赖只能从高层指向低层。core 永远不得 import torch/ultralytics/cv2。
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["__version__", "PROJECT_ROOT", "ensure_src_on_path"]

__version__ = "0.1.0"


def ensure_src_on_path() -> Path:
    """把 ``<project_root>/src`` 放到 ``sys.path`` 最前面并返回项目根目录。

    幂等：重复调用无副作用。仅在包尚未被 pip 安装时起作用。
    """
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.is_dir():
        src_str = str(src)
        if src_str in sys.path:
            sys.path.remove(src_str)
        sys.path.insert(0, src_str)
    return root


try:
    from handwash.paths import PROJECT_ROOT
except ImportError:  # pragma: no cover - 仅在包未安装且 paths 缺失时触发
    PROJECT_ROOT = ensure_src_on_path()
