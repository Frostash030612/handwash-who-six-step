"""IO 层：一切与外部世界交互的动作都集中在这里。

职责边界（CONTRIBUTING.md R16）
--------------------------------
* **允许**：读文件、写文件、解码视频、读写 CSV/JSON/JSONL/YAML、列目录。
* **禁止**：做业务判断（漏步/顺序/阈值），禁止 import torch。
  业务规则一律放 core，模型相关一律放 models。

正因为如此，测试里可以只替换这一层就完成端到端演练（见 tests/unit/io）。
"""

from __future__ import annotations

__all__ = ["manifest", "split", "utils", "video"]
