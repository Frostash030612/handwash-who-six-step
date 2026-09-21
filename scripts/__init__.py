"""``scripts`` 包：面向"人"的薄封装脚本。

与 ``handwash`` CLI 的关系
--------------------------
    * 库代码（可复用、有测试、有契约）一律放 ``src/handwash/``；
    * ``scripts/`` 只写"一次性的、带个人实验味道的、需要多步组合的"流程，
      以及不便于塞进 CLI 的快捷动作（例如固定参数的 demo、批量出图）。
    * 脚本里禁止实现业务规则：需要规则时 import ``handwash.*``。

规则（CONTRIBUTING.md R19）：脚本第一行必须是
``from _bootstrap import PROJECT_ROOT  # noqa: F401``，以保证 src-layout 下可直接运行。
"""

from __future__ import annotations

__all__: list[str] = []
