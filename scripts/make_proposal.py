#!/usr/bin/env python
"""从官方模板生成填好内容的项目提案（DOCX 或 Markdown，中英文各一份）。

为什么需要它：官方模板 `docs/Project_Proposal_Template.docx` 是**一张单列表格**，
每行一个字段，字段名占第 1 段，后面跟着若干空段作为填写区
（例如 Project Descriptions 行有 71 个空段）。手工往这些格子里粘两页正文，
极易把表格结构弄坏：多出空行、行高错乱、导出 PDF 时正文被截断。

本脚本按**真实结构**写入：定位字段标签段，在其后重建内容段与嵌套表格，
保证每次生成的结果一致、可复现。

用法::

    # 英文正式提案（官方提交格式）：填上 Canvas 分组编号与四个人的姓名学号
    python scripts/make_proposal.py --lang en \
        --group-id "43" \
        --members "Shen Ziyi:A0350940J" "Wang Lepeng:A0357864L" \
                  "Zhu Jianyu:A0353769L" "Xu Wenzhe:A0328771W"

    # 中文正式提案（姓名沿用学籍登记拼写，避免臆造汉字姓名）
    python scripts/make_proposal.py --lang zh \
        --group-id "43" --members "Shen Ziyi:A0350940J" ...

    # Markdown 版本（便于评审与 diff），以及与 DOCX 完全同源的内容
    python scripts/make_proposal.py --lang zh --format md

生成物（默认）:
    deliverables/project_proposal_en.docx  ← 用 Word 打开 → 导出 PDF → 交 Canvas
    deliverables/project_proposal_zh.docx  ← 组内评审 / 存档

设计约束（CONTRIBUTING.md R3）
    正文文案全部来自 `src/handwash/proposal_content.py`（英文）与
    `src/handwash/proposal_content_zh.py`（中文），**不在这里硬编码**，
    因此修改措辞是纯数据改动，不触碰本脚本的逻辑。中英文两份内容结构对称，
    共用同一套写入逻辑，任一侧缺字段都会立刻暴露。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Final

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401
from handwash import (  # noqa: E402
    proposal_content,
    proposal_content_zh,
)

TEMPLATE = PROJECT_ROOT / "docs" / "Project_Proposal_Template.docx"
OUT_DIR = PROJECT_ROOT / "deliverables"

#: 语言 -> 内容模块。两个模块必须导出完全相同的字段名（见各自 __all__）。
CONTENT: Final[dict[str, Any]] = {"en": proposal_content, "zh": proposal_content_zh}

#: 官方模板的字段（按表格行顺序）。**Group ID 与 Group Members 共用同一行**，
#: 因此这里的"字段数"比表格行数多一个：校验时按行数 -1 的容差处理。
_FIELD_ORDER: Final[tuple[str, ...]] = (
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

#: 中文字体：写入 w:eastAsia，避免 Word 在没有显式东亚字体时回退成方框
_CJK_FONT: Final[str] = "宋体"

_LABELS: Final[dict[str, dict[str, str]]] = {
    "en": {
        "title": "Project Proposal",
        "date": "Date of proposal",
        "project_title": "Project Title",
        "group_id": "Group ID (As Enrolled in Canvas Class Groups)",
        "members": "Group Members (name, Student ID)",
        "sponsor": "Sponsor/Client",
        "background": "Background / Aims / Objectives",
        "descriptions": "Project Descriptions",
        "background_h": "Background",
        "aim_h": "Aim",
        "objectives_h": "Objectives (measurable)",
        "scope_h": "Scope boundaries",
        "obj_header": ("#", "Objective", "Success criterion"),
        "checklist_h": "Pre-submission checklist",
    },
    "zh": {
        "title": "项目提案",
        "date": "提案日期",
        "project_title": "项目标题",
        "group_id": "分组编号（Canvas Class Groups 中的编号）",
        "members": "组员（姓名，学号）",
        "sponsor": "委托方",
        "background": "背景 / 目标 / 具体目标",
        "descriptions": "项目描述",
        "background_h": "背景",
        "aim_h": "总目标",
        "objectives_h": "具体目标（可度量）",
        "scope_h": "范围边界",
        "obj_header": ("#", "目标", "成功标准"),
        "checklist_h": "提交前检查清单",
    },
}


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
    """在指定段落后插入一个新段落，返回新段落。

    文案里的 ``**强调**`` 会被渲染成真正的加粗 run（而不是留下两个星号），
    因为 DOCX 是给人看的正式文件，Markdown 记号必须在这一步消解掉。
    """
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph

    new_element = paragraph._element.makeelement(qn("w:p"), {})
    paragraph._element.addnext(new_element)
    new_para = Paragraph(new_element, paragraph._parent)
    if text:
        _write_runs(new_para, text, base_bold=bold)
    return new_para


#: ``**加粗**``：只匹配成对的星号，落单的星号原样保留（宁可少渲染，不可吞字）
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)


def _write_runs(paragraph, text: str, *, base_bold: bool = False) -> None:
    """把带 ``**加粗**`` 记号的文案写进段落，逐段切换 run 的 bold。"""
    pos = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > pos:
            run = paragraph.add_run(text[pos : match.start()])
            run.bold = base_bold
        run = paragraph.add_run(match.group(1))
        run.bold = True
        pos = match.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        run.bold = base_bold


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

    # 建行：每行先补足单元格，再填字（同样消解 **加粗** 记号）
    for values in [list(header), *[list(row) for row in rows]]:
        cells = table.add_row().cells
        for index, value in enumerate(values[: len(cells)]):
            _write_runs(cells[index].paragraphs[0], str(value))
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


def _set_east_asian_font(doc, font_name: str = _CJK_FONT) -> None:
    """给 Normal 样式补上 ``w:eastAsia`` 字体。

    为什么需要：python-docx 写的 run 只带 ascii/hAnsi 字体；中文在 Word 里会按
    "东亚字体"回退，缺省时可能显示为方框或与正文不协调。改样式而不是逐个 run，
    改动一处即全局生效。
    """
    from docx.oxml.ns import qn

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), font_name)


def _blocks_from_content(content: Any, lang: str) -> tuple[list[dict], list[dict]]:
    """把内容模块渲染成 (背景栏块, 描述栏块)。"""
    labels = _LABELS[lang]
    bg = content.BACKGROUND_OBJECTIVES

    background_blocks: list[dict] = [
        {"type": "h", "text": labels["background_h"]},
        {"type": "p", "text": bg["background"]},
        {"type": "h", "text": labels["aim_h"]},
        {"type": "p", "text": bg["aim"]},
        {"type": "h", "text": labels["objectives_h"]},
        {
            "type": "table",
            "header": list(labels["obj_header"]),
            "rows": [list(row) for row in content.OBJECTIVES_TABLE],
        },
        {"type": "h", "text": labels["scope_h"]},
        {"type": "p", "text": bg["scope"]},
    ]

    description_blocks: list[dict] = []
    for section in content.PROJECT_DESCRIPTIONS:
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
    return background_blocks, description_blocks


def build_docx(
    *,
    lang: str,
    group_id: str,
    members: list[str],
    proposal_date: str,
    out_path: Path,
    template: Path | None = None,
) -> Path:
    """生成填好的提案 DOCX，返回输出路径。"""
    _require_docx()
    from docx import Document

    content = CONTENT[lang]

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

    if lang == "zh":
        _set_east_asian_font(doc)

    # --- 逐字段填写 -------------------------------------------------------
    _write_simple(rows[0].cells[0], "Date of proposal", proposal_date)
    _write_simple(rows[1].cells[0], "Project Title", content.PROPOSAL_TITLE)

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

    _write_simple(rows[3].cells[0], "Sponsor/Client", content.SPONSOR_CLIENT)

    background_blocks, description_blocks = _blocks_from_content(content, lang)
    _write_blocks(rows[4].cells[0], "Background/Aims/Objectives", background_blocks)
    _write_blocks(rows[5].cells[0], "Project Descriptions", description_blocks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def render_markdown(
    *, lang: str, group_id: str, members: list[str], proposal_date: str
) -> str:
    """把同一份内容渲染成 Markdown（与 DOCX 严格同源）。"""
    content = CONTENT[lang]
    labels = _LABELS[lang]
    bg = content.BACKGROUND_OBJECTIVES
    out: list[str] = [f"# {labels['title']}", ""]
    out.append(f"**{labels['date']}:** {proposal_date}")
    out.append("")
    out.append(f"**{labels['project_title']}:** {content.PROPOSAL_TITLE}")
    out.append("")
    out.append(f"**{labels['group_id']}:** {group_id}")
    out.append("")
    out.append(f"**{labels['members']}:**")
    out.append("")
    for line in members:
        out.append(f"- {line}")
    out.append("")
    out.append(f"**{labels['sponsor']}:** {content.SPONSOR_CLIENT}")
    out.append("")
    out.append("---")
    out.append("")
    out.append(f"## {labels['background']}")
    out.append("")
    out.append(f"**{labels['background_h']}.** {bg['background']}")
    out.append("")
    out.append(f"**{labels['aim_h']}.** {bg['aim']}")
    out.append("")
    out.append(f"**{labels['objectives_h']}.**")
    out.append("")
    out.append("| " + " | ".join(labels["obj_header"]) + " |")
    out.append("| --- | --- | --- |")
    for row in content.OBJECTIVES_TABLE:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    out.append("")
    out.append(f"**{labels['scope_h']}.** {bg['scope']}")
    out.append("")
    out.append("---")
    out.append("")
    out.append(f"## {labels['descriptions']}")
    out.append("")
    for section in content.PROJECT_DESCRIPTIONS:
        out.append(f"### {section['heading']}")
        out.append("")
        for text in section.get("paragraphs", ()):
            out.append(str(text))
            out.append("")
        for item in section.get("bullets", ()):
            out.append(f"- {item}")
        if section.get("bullets"):
            out.append("")
        if section.get("table"):
            header = section["table"]["header"]
            out.append("| " + " | ".join(str(h) for h in header) + " |")
            out.append("| " + " | ".join("---" for _ in header) + " |")
            for row in section["table"]["rows"]:
                out.append("| " + " | ".join(str(v) for v in row) + " |")
            out.append("")
    out.append("---")
    out.append("")
    out.append(f"## {labels['checklist_h']}")
    out.append("")
    for item in content.PRE_SUBMISSION_CHECKLIST:
        out.append(f"- [ ] {item}")
    out.append("")
    return "\n".join(out)


def _format_members(members: list[str]) -> list[str]:
    formatted: list[str] = []
    for index, raw in enumerate(members, start=1):
        if ":" in raw:
            name, _, sid = raw.partition(":")
            formatted.append(f"[{index}] {name.strip()} — {sid.strip()}")
        else:
            formatted.append(f"[{index}] {raw.strip()} — [Student ID]")
    return formatted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从官方模板生成填好的项目提案（DOCX / Markdown，中英文）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/make_proposal.py --lang en --group-id 43 \\\n"
            "      --members 'Shen Ziyi:A0350940J' 'Wang Lepeng:A0357864L' \\\n"
            "                'Zhu Jianyu:A0353769L' 'Xu Wenzhe:A0328771W'\n"
            "\n成员格式为 姓名:学号；只写姓名也可以（学号留占位，稍后在 Word 里补）。"
        ),
    )
    parser.add_argument("--lang", choices=sorted(CONTENT), default="en", help="提案语言")
    parser.add_argument(
        "--format", choices=("docx", "md"), default="docx", help="输出格式（docx 为官方提交格式）"
    )
    parser.add_argument("--template", default=str(TEMPLATE), help="模板 DOCX 路径")
    parser.add_argument("--out", default="", help="输出路径（默认按语言与格式推导）")
    parser.add_argument("--group-id", default="[PASTE CANVAS GROUP ID]", help="Canvas 分组编号")
    parser.add_argument("--members", nargs="*", default=[], help='成员，"姓名:学号" 形式，可多个')
    parser.add_argument("--member-names", default="", help="只给姓名时的快捷方式：逗号分隔")
    parser.add_argument("--date", default="", help="提案日期（默认取内容模块的 PROPOSAL_DATE）")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不写文件")
    args = parser.parse_args()

    content = CONTENT[args.lang]
    proposal_date = args.date or content.PROPOSAL_DATE

    members = list(args.members)
    if not members and args.member_names:
        members = [name.strip() for name in args.member_names.split(",") if name.strip()]
    if not members:
        members = ["[Name 1]", "[Name 2]", "[Name 3]", "[Name 4]"]
    formatted = _format_members(members)

    default_name = f"project_proposal_{args.lang}.{args.format}"
    out = Path(args.out) if args.out else OUT_DIR / default_name

    if args.dry_run:
        print("语言  :", args.lang)
        print("格式  :", args.format)
        print("模板  :", args.template)
        print("输出  :", out)
        print("分组  :", args.group_id)
        print("日期  :", proposal_date)
        print("成员  :")
        for line in formatted:
            print("   ", line)
        print(
            f"\n正文字数：背景/目标 + 项目描述共 {len(content.PROJECT_DESCRIPTIONS)} 个小节、"
            f"{len(content.OBJECTIVES_TABLE)} 条目标。"
        )
        print("（--dry-run：未写文件）")
        return 0

    if args.format == "md":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            render_markdown(
                lang=args.lang,
                group_id=args.group_id,
                members=formatted,
                proposal_date=proposal_date,
            ),
            encoding="utf-8",
        )
    else:
        build_docx(
            lang=args.lang,
            group_id=args.group_id,
            members=formatted,
            proposal_date=proposal_date,
            out_path=out,
            template=Path(args.template),
        )

    print(f"已生成：{out}")
    if args.format == "docx":
        print("下一步：用 Word 打开 → 核对表格内文字没有被截断 → 导出 PDF → 交到 Canvas。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
