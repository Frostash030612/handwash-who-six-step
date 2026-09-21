# 变更日志（CHANGELOG）

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的结构。
**破坏性改动必须标 `BREAKING`，并写清"哪些历史实验需要重跑"。**

规则依据：[`CONTRIBUTING.md`](CONTRIBUTING.md) 第 2 节 L1 改动流程。

---

## [未发布]

### 新增
- `docs/PROJECT_PROPOSAL.md`（英文主）+ `docs/zh/PROJECT_PROPOSAL.md`（中文）：
  官方提案模板每一栏的可粘贴内容包（背景/目标/可度量目标表、四项能力说明、
  系统架构、数据、方法、评估、计划、风险）。
- `src/handwash/proposal_content.py`：提案文案的单一来源（纯数据，便于反复改措辞）。
- `scripts/make_proposal.py`：按官方模板的真实表格结构生成填好的提案 DOCX
  （`deliverables/project_proposal.docx`），支持 `--dry-run` 预览。
- `deliverables/README.md`：提交件清单 + 最终报告建议大纲（对齐官方四项报告要求）。
- `docs/PROJECT_PLAN_4_WEEK.md`（英文主）+ `docs/zh/PROJECT_PLAN_4_WEEK.md`（中文）：
  依据官方文件 `PRS-PatternRecognitionSystems-Practice-Module 7.0 - FT.pdf` 制定的
  四周四人分工计划 —— 计分项拆解、四个角色、逐周任务与验收标准、要求追溯矩阵、风险清单、
  完成定义。英文版与中文版同时维护。
- **双语文档结构**：英文成为主版本，中文保留为镜像，每个页面顶部都有语言切换行。
  - `README.md`（英文主） ↔ `README.zh-CN.md`
  - `CONTRIBUTING.md`（英文主，位于仓库根） ↔ `docs/zh/CONTRIBUTING.md`
  - `docs/{ARCHITECTURE,PROTOCOL,CONFIG,DATA,RULES_CARD}.md`（英文主）
    ↔ `docs/zh/{同名}.md`
  - 内部工作文档只有中文，不予翻译：`docs/EXPERIMENTS.md`、`docs/SELF_RECORDING.md`、
    `CHANGELOG.md`、`tests/README.md`
- `scripts/check_docs.py`：文档体检 —— 检查相对链接是否断链、双语配对是否都有语言切换行。
- `scripts/sync_doc_locales.py`：幂等地为双语文档插入/更新语言切换行（新增双语文档时改表重跑即可）。
- `.gitattributes`：统一仓库内换行为 LF，避免 Windows/macOS 之间的"整文件 diff"。
- 初始工程框架：分层架构（L0 基础设施 / L1 契约 / L2 数据与 IO / L3 模型 / L4 编排）。
- `core/labels.py`：WHO 六步的唯一权威标签空间 + 5 个公开数据集的别名映射。
- `core/protocol.py`：完整性判定（漏步 / 乱序 / 时长不足 / 重复步骤 + 综合得分）。
- `core/config.py`：YAML → 冻结 dataclass 的严格配置系统（未知键直接报错）。
- `core/metrics.py`：Accuracy / Macro-F1 / Weighted-F1 / 逐类 P-R / 混淆矩阵 / ECE / Bootstrap CI。
- `io/`：视频解码（OpenCV 优先，imageio 回退）、manifest 六条不变量校验、按**原始视频**划分的三道防泄漏。
- `data/`：帧级 Dataset、显式可测的增强流水线、合成数据生成器（冒烟与单测用）。
- `models/`：YOLO26n-cls 适配器、MobileNetV2/ResNet18/EfficientNet-B0 基线、GRU/TCN/MeanPool 时序头、注册表、checkpoint 存取。
- `pipelines/`：prepare / train / evaluate / infer / assess 五个流程 + 共享设施。
- `cli.py`：`handwash <prepare|train|evaluate|infer|assess|doctor|config>`。
- 结构守卫：`scripts/check_structure.py`（分层依赖 / 硬编码路径 / print）、`scripts/check_config.py`（配置契约 + 依赖声明一致性）。
- 测试：`tests/` 下 396 个契约层用例（无需数据、无需 GPU，秒级完成）。
- 文档：README、CONTRIBUTING（修改规则）、ARCHITECTURE（含 RFC 流程）、DATA、CONFIG、PROTOCOL、EXPERIMENTS、SELF_RECORDING。
- CI：格式 → 分层依赖 → 配置契约 → 单元测试 → 冒烟训练 → 依赖一致性。

### 变更
- **文档语言策略变更**：英文成为主版本（课程提交与评分使用），中文从"唯一版本"改为"镜像"。
  原 `CONTRIBUTING.md` 与 `docs/{ARCHITECTURE,PROTOCOL,CONFIG,DATA,RULES_CARD}.md`
  已移动到 `docs/zh/` 下，英文版本占用原路径。**历史链接需要更新**：
  `docs/CONFIG.md` 现在指向英文版，中文版在 `docs/zh/CONFIG.md`。
- `confusion_matrix(normalize=...)` 现在严格校验取值：只接受
  `None` / `True` / `"true"`（按行，召回视角）/ `"pred"`（按列，精确率视角）/ `"all"`。
  **以前任何非 `True` 的取值都会被静默当作按列归一化**，容易让混淆矩阵被误读。
- `parse_overrides` 现在拒绝"等号右边为空"的写法（如 `train.epochs=`），
  提示改写成 `key=null` 或补上取值。以前它会被 YAML 解析成 `None` 再往下走，
  报错位置离问题很远。
- `Registry.register` 现在先做类型检查、再 `strip` 后判空；
  `register("   ")` 由"注册成空字符串键"改为直接抛 `ConfigError`。
- `AppConfig()` 的默认 `datasets` 现在自带一套档案，因此默认配置**自洽可校验**。
  以前必须由 `configs/config.yaml` 提供 datasets 段才能通过校验。
- 所有模型（含单帧模型）统一接受 `(B, T, 3, H, W)` 输入并输出 `(B, T, C)`。

### 修复
- 视频写盘：`write_video` 在缺少 ffmpeg 后端时回退写出 GIF，保证"无编码器环境也能生成可解码的测试视频"。
- 训练：`TemporalClassifier` 的 `backbone_kwargs['pretrained']` 之前会被静默丢弃并回退为 `True`，导致离线环境尝试联网下载权重而失败。

---

## 迁移说明（如果历史实验受上面"变更"影响）

| 受影响的实验 | 需要做什么 |
| --- | --- |
| 任何用过非 `True` 的 `normalize` 取值生成混淆矩阵的记录 | 重新生成图（指标 JSON 不受影响） |
| 在 `AppConfig()` 上直接加过 `datasets` 段的临时脚本 | 可以删掉那段补丁 |
| 任何时序模型实验（`train.mode=clip/hybrid`） | **建议重跑**：`pretrained` 修复后骨干权重来源不同 |
| 依赖 `Registry.register("   ")` 成功行为的代码 | 改为传合法名字（本来就该如此） |

---

## 版本号约定

`MAJOR.MINOR.PATCH`：

- `MAJOR`：L1 契约破坏性变更（标签空间顺序、schema 字段语义、配置键删除/改名）；
- `MINOR`：新增模型 / 数据集 / 指标 / 流程（向后兼容）；
- `PATCH`：修复与文档。

当前版本：**0.1.0**（框架建立，尚未产出正式实验结果）。
