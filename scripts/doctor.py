#!/usr/bin/env python
"""环境与配置自检（等价于 ``handwash doctor``，但不需要先安装本包）。

新组员 clone 下来的**第一条命令**。它回答：
    1. 依赖够不够（缺什么、怎么补）；
    2. 配置能不能严格加载（有没有拼错的键、越界的取值）；
    3. 数据在哪、manifest 生成了吗、划分是否泄漏；
    4. 有没有 GPU、主模型权重能不能加载。

只报告，不修改：doctor 永远不写文件、不下载权重、不改环境。

用法::

    python scripts/doctor.py
    python scripts/doctor.py config=configs/data/pskuss.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401  （副作用：注册导入路径）

from handwash.cli import split_argv  # noqa: E402
from handwash.cli_doctor import run_doctor  # noqa: E402
from handwash.core.config import load_config, parse_overrides  # noqa: E402
from handwash.errors import HandwashError  # noqa: E402
from handwash.logging import setup_logging  # noqa: E402


def main() -> int:
    plain, override_items = split_argv(sys.argv[1:])
    setup_logging(force=True)

    overrides = parse_overrides(override_items)
    base = ["configs/config.yaml"]
    overlay = overrides.pop("config", None)
    if overlay:
        base.append(str(overlay))

    try:
        rc = load_config(base, overrides=overrides)
    except HandwashError as exc:
        # 配置本身加载失败就是最重要的诊断结果，交给 doctor 打印而不是崩掉
        print(f"配置加载失败，下面把它作为一项诊断结果展示：\n{exc}\n")
        return run_doctor(None, exc)

    del plain
    return run_doctor(rc)


if __name__ == "__main__":
    raise SystemExit(main())
