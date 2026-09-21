#!/usr/bin/env python
"""从官方模板生成填好内容的项目提案 DOCX。

为什么需要它：官方模板 `docs/Project_Proposal_Template.docx` 是**一张单列表格**，
每行一个字段，字段名占第 1 段，后面跟着若干空段作为填写区
（例如 Project Descriptions 行有 71 个空段）。手工往这些格子里粘两页正文，
极易把表格结构弄坏：多出空行、行高错乱、导出 PDF 时正文被截断。

本脚本按**真实结构**写入：定位字段标签段，在其后重建内容段与嵌套表格，
保证每次生成的结果一致、可复现。

用法::

    # 最少用法：填上 Canvas 分组编号与四个人的姓名学号
    python scripts/make_proposal.py \
        --group-id "PRS-G07" \
        --members "张三:1000001" "李四:1000002" "王五:1000003" "赵六:1000004"

    # 只预览不写文件
    python scripts/make_proposal.py --members "A:1" "B:2" "C:3" "D:4" --dry-run

生成物：`deliverables/project_proposal.docx` → 用 Word 打开核对 → 导出 PDF → 交 Canvas。

设计约束（CONTRIBUTING.md R3）
    正文文案全部来自 `src/handwash/proposal_content.py`，**不在这里硬编码**，
    因此修改措辞是纯数据改动，不触碰本脚本的逻辑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401

from handwash.proposal_content import (  # noqa: E402
    BACKGROUND_OBJECTIVES,
    OBJECTIVES_TABLE,
    PROJECT_DESCRIPTIONS,
    PROPOSAL_DATE,
    PROPOSAL_TITLE,
    SPONSOR_CLIENT,
)

TEMPLATE = PROJECT_ROOT / "docs" / "Project_Proposal_Template.docx"
DEFAULT_OUT = PROJECT_ROOT / "deliverables" / "project_proposal.docx"

#: 官方模板的字段（按表格行顺序）。**Group ID 与 Group Members 共用同一行**，
#: 因此这里的"字段数"比表格行数多一个：校验时按行数 -1 的容差处理。
_FIELD_ORDER: tuple[str, ...] = (
    "Date of proposal",          # row 0
    "Project Title",             # row 1
    "Group ID",                  # row 2（与 Group Members 同格）
    "Group Members",             # row 2（同格，位于 Group ID 下方）
    "Sponsor/Client",            # row 3
    "Background/Aims/Objectives",  # row 4
    "Project Descriptions",      # row 5
)

#: 表格应有的最少行数（见上面注释：7 个字段占 6 行）
_MIN_TABLE_ROWS: Final[int] = 6


def _require_docx():
    try:
        import docx  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 环境缺依赖
        print(
            "需要 python-docx：conda install -c conda-forge python-docx"
            "（或 pip install python-docx）",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return docx


def _insert_paragraph_after(paragraph, text: str = "", *, bold: bool = False):
    """在指定段落后插入一个新段落，返回新段落。"""
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph

    new_element = paragraph._element.makeelement(qn("w:p"), {})
    paragraph._element.addnext(new_element)
    new_para = Paragraph(new_element, paragraph._parent)
    if text:
        run = new_para.add_run(text)
        run.bold = bold
    return new_para


def _insert_table_after(paragraph, header: list[str], rows: list[list[str]]):
    """在指定段落后插入一个带边框的表格。

    实现要点：这里直接在文档里插入自带 ``<w:tblPr>`` 与 ``<w:tblGrid>`` 的表格 XML。
    若手工构造 ``<w:tbl>`` 却漏掉 ``tblPr``，python-docx 会抛 InvalidXmlError；
    若改用 ``document.add_table()``，表格会先被追加到正文末尾、再搬到单元格内，
    中间态难以收拾。自己拼 XML 最干净。
    """
    from docx.oxml.ns import nsdecls
    from docx.oxml.parser import parse_xml
    from docx.table import Table

    column_width = max(1200, int(9000 / max(1, len(header))))
    grid = "".join(f'<w:gridCol w:w="{column_width}"/>' for _ in header)
    borders = (
        "<w:tblBorders>"
        + "".join(
            f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        + "</w:tblBorders>"
    )
    xml = (
        f"<w:tbl {nsdecls('w')}>"
        f"<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>{borders}</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"</w:tbl>"
    )
    table_element = parse_xml(xml)
    paragraph._element.addnext(table_element)
    table = Table(table_element, paragraph._parent)

    # 建行：每行先补足单元格，再填字
    for values in [list(header), *[list(row) for row in rows]]:
        cells = table.add_row().cells
        for index, value in enumerate(values[: len(cells)]):
            cells[index].text = str(value)
    # 表头加粗
    for cell in table.rows[0].cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
    return table


def _clear_rows(cell, label_prefix: str) -> None:
    """删掉某个字段标签之后的所有段落（即模板预留的空白填写区）。"""
    paragraphs = cell.paragraphs
    start = None
    for index, para in enumerate(paragraphs):
        if para.text.strip().startswith(label_prefix):
            start = index
            break
    if start is None:
        raise SystemExit(f"模板里找不到字段：{label_prefix!r}（模板结构可能变了）")
    for para in paragraphs[start + 1 :]:
        para._element.getparent().remove(para._element)


def _anchor(cell, label_prefix: str):
    """返回某个字段的标签段对象。"""
    for para in cell.paragraphs:
        if para.text.strip().startswith(label_prefix):
            return para
    raise SystemExit(f"模板里找不到字段：{label_prefix!r}（模板结构可能变了）")


def _write_simple(cell, label_prefix: str, value: str) -> None:
    """单行字段：清空填写区后写一行。"""
    _clear_rows(cell, label_prefix)
    _insert_paragraph_after(_anchor(cell, label_prefix), value)


def _write_multi(cell, label: str, lines: list[str]) -> None:
    """多行字段（例如组员列表）：逐行写入。"""
    _clear_rows(cell, label)
    cursor = _anchor(cell, label)
    for line in lines:
        cursor = _insert_paragraph_after(cursor, line)


def _write_blocks(cell, label: str, blocks: list[dict]) -> None:
    """把块列表写进字段的填写区：段落 / 小标题 / 列表 / 表格。"""
    _clear_rows(cell, label)
    cursor = _anchor(cell, label)
    for block in blocks:
        kind = block.get("type", "p")
        if kind == "p":
            cursor = _insert_paragraph_after(cursor, str(block["text"]))
        elif kind == "h":
            cursor = _insert_paragraph_after(cursor, str(block["text"]), bold=True)
        elif kind == "ul":
            for item in block["items"]:
                cursor = _insert_paragraph_after(cursor, f"• {item}")
        elif kind == "table":
            cursor = _insert_table_after(
                cursor,
                [str(h) for h in block["header"]],
                [[str(v) for v in row] for row in block["rows"]],
            )
        else:  # pragma: no cover - 防御性
            raise SystemExit(f"未知的内容块类型：{kind!r}")


def build(
    *,
    group_id: str,
    members: list[str],
    proposal_date: str,
    out_path: Path,
    template: Path | None = None,
) -> Path:
    """生成填好的提案 DOCX，返回输出路径。"""
    _require_docx()
    from docx import Document

    template_path = Path(template) if template is not None else TEMPLATE
    if not template_path.exists():
        raise SystemExit(f"找不到模板：{template_path}")
    doc = Document(str(template_path))
    if not doc.tables:
        raise SystemExit(f"模板里没有表格：{template_path}")
    rows = list(doc.tables[0].rows)
    if len(rows) < _MIN_TABLE_ROWS:
        raise SystemExit(
            f"模板表格只有 {len(rows)} 行，预期至少 {_MIN_TABLE_ROWS} 行；"
            "请确认用的是官方模板，或更新本脚本的字段映射。"
        )

    # --- 逐字段填写 -------------------------------------------------------
    _write_simple(rows[0].cells[0], "Date of proposal", proposal_date)
    _write_simple(rows[1].cells[0], "Project Title", PROPOSAL_TITLE)

    # 第 3 行同时含 Group ID 与 Group Members：先记录两个标签段，再重建
    cell_group = rows[2].cells[0]
    group_anchor = _anchor(cell_group, "Group ID")
    # 删掉 Group ID 标签之后的一切（含 members 标签与空段），然后按顺序重写两栏
    _clear_rows(cell_group, "Group ID")
    cursor = _insert_paragraph_after(group_anchor, group_id)
    cursor = _insert_paragraph_after(cursor, "")
    cursor = _insert_paragraph_after(cursor, "Group Members (name, Student ID): ", bold=True)
    for line in members:
        cursor = _insert_paragraph_after(cursor, line)

    _write_simple(rows[3].cells[0], "Sponsor/Client", SPONSOR_CLIENT)

    _write_blocks(
        rows[4].cells[0],
        "Background/Aims/Objectives",
        [
            {"type": "h", "text": "Background"},
            {"type": "p", "text": BACKGROUND_OBJECTIVES["background"]},
            {"type": "h", "text": "Aim"},
            {"type": "p", "text": BACKGROUND_OBJECTIVES["aim"]},
            {"type": "h", "text": "Objectives (measurable)"},
            {
                "type": "table",
                "header": ["#", "Objective", "Success criterion"],
                "rows": [list(row) for row in OBJECTIVES_TABLE],
            },
            {"type": "h", "text": "Scope boundaries"},
            {"type": "p", "text": BACKGROUND_OBJECTIVES["scope"]},
        ],
    )

    description_blocks: list[dict] = []
    for section in PROJECT_DESCRIPTIONS:
        description_blocks.append({"type": "h", "text": section["heading"]})
        for text in section.get("paragraphs", ()):
            description_blocks.append({"type": "p", "text": text})
        if section.get("bullets"):
            description_blocks.append({"type": "ul", "items": list(section["bullets"])})
        if section.get("table"):
            description_blocks.append(
                {
                    "type": "table",
                    "header": list(section["table"]["header"]),
                    "rows": [list(r) for r in section["table"]["rows"]],
                }
            )
    _write_blocks(rows[5].cells[0], "Project Descriptions", description_blocks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从官方模板生成填好的项目提案 DOCX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/make_proposal.py --group-id PRS-G07 \\\n"
            "      --members 张三:1000001 李四:1000002 王五:1000003 赵六:1000004\n"
            "\n成员格式为 姓名:学号；只写姓名也可以（学号留占位，稍后在 Word 里补）。"
        ),
    )
    parser.add_argument("--template", default=str(TEMPLATE), help="模板 DOCX 路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出 DOCX 路径")
    parser.add_argument("--group-id", default="[PASTE CANVAS GROUP ID]", help="Canvas 分组编号")
    parser.add_argument("--members", nargs="*", default=[], help='成员，"姓名:学号" 形式，可多个')
    parser.add_argument("--member-names", default="", help="只给姓名时的快捷方式：逗号分隔")
    parser.add_argument("--date", default=PROPOSAL_DATE, help="提案日期")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不写文件")
    args = parser.parse_args()

    members = list(args.members)
    if not members and args.member_names:
        members = [name.strip() for name in args.member_names.split(",") if name.strip()]
    if not members:
        members = ["[Name 1] — [Student ID]", "[Name 2] — [Student ID]",
                   "[Name 3] — [Student ID]", "[Name 4] — [Student ID]"]

    formatted: list[str] = []
    for index, raw in enumerate(members, start=1):
        if ":" in raw:
            name, _, sid = raw.partition(":")
            formatted.append(f"[{index}] {name.strip()} — {sid.strip()}")
        else:
            formatted.append(f"[{index}] {raw.strip()} — [Student ID]")

    if args.dry_run:
        print("模板  :", args.template)
        print("输出  :", args.out)
        print("分组  :", args.group_id)
        print("日期  :", args.date)
        print("成员  :")
        for line in formatted:
            print("   ", line)
        print(f"\n正文字数：Background/Aims/Objectives + Project Descriptions 共 "
              f"{len(PROJECT_DESCRIPTIONS)} 个小节、{len(OBJECTIVES_TABLE)} 条目标。")
        print("（--dry-run：未写文件）")
        return 0

    out = build(
        group_id=args.group_id,
        members=formatted,
        proposal_date=args.date,
        out_path=Path(args.out),
        template=Path(args.template),
    )
    print(f"已生成：{out}")
    print("下一步：用 Word 打开 → 核对表格内文字没有被截断 → 导出 PDF → 交到 Canvas。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
