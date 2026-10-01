#!/usr/bin/env python
"""分层依赖体检（`make archcheck` / pre-commit 钩子）。

它守护的是 CONTRIBUTING.md 里最容易被违反的几条规则：
    R5  core 不得 import torch / ultralytics / cv2
    R5  io 不得 import torch；io 不得 import models / pipelines
    R16 业务代码不得出现硬编码目录字符串（应走 handwash.paths）
    R8  库代码不得用 print / sys.exit 输出

用法::

    python scripts/check_structure.py            # 检查全部
    python scripts/check_structure.py --verbose  # 打印通过项

退出码 0 表示通过；非 0 表示存在违规（CI 与 pre-commit 都依赖这一点）。
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401  （副作用：注册导入路径）

SRC = PROJECT_ROOT / "src" / "handwash"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}

#: 分层规则：层名 -> (允许 import 的项目内部顶层包, 禁止 import 的第三方包)
LAYER_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "core": {
        "forbid_internal": ("io", "data", "models", "pipelines", "cli"),
        "forbid_external": ("torch", "torchvision", "ultralytics", "cv2", "pandas", "matplotlib"),
    },
    "io": {
        "forbid_internal": ("data", "models", "pipelines", "cli"),
        "forbid_external": ("torch", "torchvision", "ultralytics"),
    },
    "data": {
        "forbid_internal": ("models", "pipelines", "cli"),
        "forbid_external": ("ultralytics",),
    },
    "models": {
        "forbid_internal": ("pipelines", "cli"),
        "forbid_external": (),
    },
    "pipelines": {
        "forbid_internal": ("cli",),
        "forbid_external": (),
    },
}

#: 允许违反上述规则的例外（必须写清理由，且只能逐文件、逐模块豁免）
ALLOWED_EXCEPTIONS: dict[str, set[str]] = {
    # core/config.py 是全项目唯一允许读 YAML 的契约层模块（配置加载本身）
    "core/config.py": {"yaml"},
    # core/metrics.py 等只用 numpy 计算，属允许范围
    "core/seeding.py": {"torch"},  # 可选依赖：为了设种子才 import，缺 torch 时降级
    "core/protocol.py": set(),
    # data/__init__.py 允许探测 torch 是否存在（缺 torch 时给出安装提示）
    "data/__init__.py": {"torch"},
}

#: 禁止直接出现的目录字面量（应走 handwash.paths 的常量）
FORBIDDEN_DIR_LITERALS = (
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "outputs",
    "models",
)

#: 允许出现上述字面量的文件。每一项都必须有理由，且必须是"配置默认值"或"检查器自身"。
LITERAL_EXCEPTIONS = {
    "cli_doctor.py",  # 自检输出里要打印人类可读的路径示例
    "paths.py",       # 常量的定义处
    "config.py",      # 配置默认值（datasets 档案的默认路径）
}

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)")
_PRINT_RE = re.compile(r"^\s*print\s*\(")
_EXIT_RE = re.compile(r"^\s*sys\.exit\s*\(")
_BARE_EXCEPT_RE = re.compile(r"except\s*:")


@dataclass
class Violation:
    path: Path
    lineno: int
    rule: str
    message: str

    def render(self) -> str:
        rel = self.path.relative_to(PROJECT_ROOT)
        return f"{rel}:{self.lineno}: [{self.rule}] {self.message}"


def _layer_of(path: Path) -> str | None:
    """判断文件属于哪一层（顶层文件返回 None，不做分层限制）。"""
    try:
        rel = path.relative_to(SRC)
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    return rel.parts[0]


def _module_name(name: str) -> str:
    """把 import 名归一化到顶层模块。"""
    return name.split(".")[0]


def _check_imports(path: Path, tree: ast.AST, layer: str) -> list[Violation]:
    rules = LAYER_RULES[layer]
    rel_key = str(path.relative_to(SRC)).replace("\\", "/")
    exceptions = ALLOWED_EXCEPTIONS.get(rel_key, set())
    out: list[Violation] = []

    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for raw in names:
            top = _module_name(raw)
            if top == "handwash":
                parts = raw.split(".")
                if len(parts) >= 2:
                    sub = parts[1]
                    if sub in rules["forbid_internal"]:
                        out.append(
                            Violation(
                                path,
                                getattr(node, "lineno", 0),
                                "layer",
                                f"{layer} 层禁止依赖 `handwash.{sub}`（见 docs/ARCHITECTURE.md 的分层图）",
                            )
                        )
            elif top in rules["forbid_external"] and top not in exceptions:
                out.append(
                    Violation(
                        path,
                        getattr(node, "lineno", 0),
                        "layer",
                        f"{layer} 层禁止 import `{top}`（契约层必须保持零重依赖，便于单元测试）",
                    )
                )
    return out


def _check_style(path: Path, lines: list[str]) -> list[Violation]:
    out: list[Violation] = []
    is_test = "tests" in path.parts
    # CLI 层是"给人看的输出界面"，用 print 是刻意的：结果不能混进日志流，
    # 否则 `handwash config > out.yaml` 之类的用法会被日志污染。
    is_output_layer = path.name in ("cli.py", "cli_doctor.py")
    for lineno, line in enumerate(lines, start=1):
        if _PRINT_RE.match(line) and not is_test and not is_output_layer:
            out.append(
                Violation(path, lineno, "logging", "禁止用 print 输出运行信息，请用 handwash.logging.get_logger")
            )
        if _EXIT_RE.match(line) and path.name not in ("cli.py", "_entry"):
            out.append(
                Violation(path, lineno, "errors", "库代码禁止 sys.exit，请抛 HandwashError 子类")
            )
        if _BARE_EXCEPT_RE.match(line):
            out.append(
                Violation(path, lineno, "errors", "禁止裸 except:，请捕获具体异常或至少 except Exception as exc")
            )
    return out


def _check_literals(path: Path, lines: list[str]) -> list[Violation]:
    """检查硬编码目录字面量。

    判定方式：该行是否出现**带引号的**目录名（例如 ``"data/processed"``、
    ``'outputs'``）。这样既能抓到 ``cfg.get("manifest", "data/processed/x.csv")``，
    又不会把**字符串模板**误判为违规：

        f"{DATA_PROCESSED_DIRNAME}/{name}"     -> 合规（引用的是常量）
        "data/processed"                        -> 违规（硬编码）

    注释与文档字符串里出现这些词是允许的（文档必须能举例），因此逐行跳过。
    """
    if path.name in LITERAL_EXCEPTIONS:
        return []
    out: list[Violation] = []
    in_docstring = False
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.count('"""') == 1:  # 进入/离开多行文档字符串
            in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith("#"):
            continue
        for literal in FORBIDDEN_DIR_LITERALS:
            if f'"{literal}"' in line or f"'{literal}'" in line:
                out.append(
                    Violation(
                        path,
                        lineno,
                        "paths",
                        f"硬编码目录字面量 `{literal}`；请改用 handwash.paths 的常量"
                        "（如 DATA_PROCESSED_DIRNAME / DEFAULT_MANIFEST_RELPATH）或配置项",
                    )
                )
    return out


def check_file(path: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Violation(path, exc.lineno or 0, "syntax", f"语法错误：{exc.msg}")]

    out: list[Violation] = []
    layer = _layer_of(path)
    if layer and layer in LAYER_RULES:
        out.extend(_check_imports(path, tree, layer))
    out.extend(_check_style(path, lines))
    out.extend(_check_literals(path, lines))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分层依赖与结构体检")
    parser.add_argument("--verbose", action="store_true", help="打印通过的文件数等细节")
    args = parser.parse_args(argv)

    files = sorted(
        p for p in SRC.rglob("*.py") if not any(part in SKIP_DIRS for part in p.parts)
    )
    violations: list[Violation] = []
    for path in files:
        violations.extend(check_file(path))

    if violations:
        print(f"结构体检未通过：发现 {len(violations)} 处违规。")
        for violation in violations:
            print("  " + violation.render())
        print(
            "\n修复指引见 docs/ARCHITECTURE.md（分层规则）与 CONTRIBUTING.md R5/R8/R16。\n"
            "确需豁免时，请在 scripts/check_structure.py 的 ALLOWED_EXCEPTIONS 中"
            "逐文件登记并写明理由。"
        )
        return 1

    print(f"结构体检通过：检查了 {len(files)} 个源文件，未发现越层依赖或硬编码路径。")
    if args.verbose:
        for layer in LAYER_RULES:
            count = len([p for p in files if _layer_of(p) == layer])
            print(f"  {layer:<10} {count} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
