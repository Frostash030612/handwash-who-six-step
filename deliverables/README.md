# deliverables/ —— 提交件目录

本目录存放最终提交物：项目提案、最终报告、视频演示、PPT、打包 ZIP 等。
**整个目录被 `.gitignore` 排除**，因为提交件含姓名与学号；这是刻意的约定
（见 `.gitignore` 中"提交件（含个人姓名/学号，不进 Git）"一节）。

只有下面两类内容会进入仓库：

| 内容 | 是否进 Git | 原因 |
| --- | --- | --- |
| 本文件 `README.md` | ✅ | `.gitignore` 里显式白名单 |
| `repo/` 下的**脱敏副本** | ✅ | 学号已掩码为 `A012****X` 形式，可安全入库 |
| 其余文件（完整版提案、报告、视频、ZIP…） | ❌ | 含完整姓名/学号，只在本地与共享盘流转 |

## 目录结构

```
deliverables/
├── README.md                      # 本文件（进 Git）
├── project_proposal_en.docx       # 完整版英文提案 —— 交 Canvas 用（不进 Git）
├── project_proposal_en.md
├── project_proposal_zh.docx       # 完整版中文提案 —— 组内评审用（不进 Git）
├── project_proposal_zh.md
└── repo/                          # 脱敏副本（进 Git）
    ├── project_proposal_en.docx
    ├── project_proposal_en.md
    ├── project_proposal_zh.docx
    └── project_proposal_zh.md
```

> `deliverables/repo/` 里的副本**学号已脱敏，不能直接交 Canvas**。
> 交作业请用上一层的完整版；`repo/` 只是为了让仓库里能看到提案内容。

## 重新生成

提案由 `scripts/make_proposal.py` 确定性地生成，正文来自
`src/handwash/proposal_content.py`（英文）与 `src/handwash/proposal_content_zh.py`（中文），
因此改措辞不需要动脚本。

> 下面的命令里，姓名与学号一律写成占位符：**本文件会进 Git，不要把真实学号写进来**。
> 真实值见 Canvas 分组，或本目录完整版提案的成员栏。

```bash
# 完整版（本地 / Canvas 提交用）——学号替换成真实值后执行
python scripts/make_proposal.py --lang en --group-id 43 \
    --members "Full Name:A0000000X" "Full Name:A0000000Y" \
              "Full Name:A0000000Z" "Full Name:A0000000W"
python scripts/make_proposal.py --lang zh --group-id 43 --members ...

# 脱敏副本（进 Git 用；--mask-ids 会自动写到 deliverables/repo/）
python scripts/make_proposal.py --lang en --mask-ids --group-id 43 --members ...
python scripts/make_proposal.py --lang zh --mask-ids --group-id 43 --members ...

# 两种格式都想要就再加 --format md
```

## 提交前

1. 用 Word 打开 `project_proposal_en.docx`，核对表格内文字没有被截断。
2. 导出 PDF（本机没有 LibreOffice/LaTeX，PDF 导出需在 Word 中完成）。
3. 上传 Canvas → Assignments → Practice Module（每队一份）。
4. 提交回执截图存回本目录，并按需更新 `repo/` 下的脱敏副本。
