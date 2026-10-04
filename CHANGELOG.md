# 变更日志（CHANGELOG）

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的结构。
**破坏性改动必须标 `BREAKING`，并写清"哪些历史实验需要重跑"。**

规则依据：[`CONTRIBUTING.md`](CONTRIBUTING.md) 第 2 节 L1 改动流程。

---

## [未发布]

### 新增
- **2026-10-04 阶段成果**：将当前云端七分类帧模型 `exp.pt` 与配套
  `exp_temporal_head.pt`（GRU 时序头）作为本地演示的固定模型包。`--demo-exp`
  默认加载两者；可用 `--no-temporal-head` 回退到纯逐帧模型进行对照。
- **双标注数据质控**：PSKUS 数据准备新增双标注一致性过滤，并剔除动作切换边界帧；
  当前配置使用 `Annotator1/Annotator2`，边界半径为 7 个源标注帧。对应单元测试：
  `tests/unit/test_pskuss_quality_filter.py`。
- **本地离线演示链路**：摄像头页面可载入本地视频，展示逐帧预测、步骤时间线、覆盖情况、
  顺序异常及重复步骤。页面明确区分“离线验证视频”和实时摄像头结果。
- **演示验证集**：本地 `data/external/pskus_demo_validation/` 保留 8 条经公开标注核对的
  PSKUS 视频（正确流程 5 条、错误流程 3 条），仅用于开题演示和定性验证，不参与训练，
  也不提交 Git。
- `outputs/README.md`：说明运行报告与特征缓存属于可再生成产物；删除历史本地输出和
  Python 缓存后，当前模型、数据和交付件不受影响。
- `exp.pt` 七类 Ultralytics 可行性演示：`python scripts/run_camera.py --demo-exp` 复用本机
  摄像头网页和六步报告，严格核对权重类别顺序，演示产物单独保存到 `outputs/exp_demo/`。
- 当前产品主线改为云端训练 YOLO 逐帧分类模型，再在本机外接摄像头网页动态推理；
  新增 `configs/experiments/live_yolo_frame.yaml`、`handwash camera`、本机网页服务、
  逐帧会话状态与最终六步报告。GRU/TCN 与 PE 方案保留为可选研究。
- 浏览器逐帧发送采集时间和 JPEG；实时显示只平均已到达帧，长缺帧区间在最终报告中标为未知。
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
- `exp_temporal_head.pt` 由通用权重忽略规则改为与 `exp.pt` 一样允许提交，保证其他组员
  拉取仓库后可复现当前时序增强演示。
- 本地演示的完整性语义修正：仅覆盖六个步骤时显示“六步已覆盖，但顺序待改进”；只有覆盖
  且顺序正确时才显示“流程完整、顺序正确”。`other/unknown` 不再参与 WHO 六步顺序判断。
- 公开仓库说明改用项目相对路径，不再展示机器专属盘符或目录；根目录 `exp.pt`
  作为演示权重随仓库发布，NDJSON 数据导出清单仍保留在本地。历史提案中的真实姓名、
  学号改为占位信息，`deliverables/repo/` 的旧副本不再放行进 Git。
- **默认训练数据集改为 PSKUS**：`configs/config.yaml` 与 `AppConfig()` 现在默认选择
  `pskuss` 和其标签空间；旧 Kaggle 单帧基线的实验配置已显式指定 `kaggle`。
  如果自定义实验以前依赖默认 Kaggle 数据集，请显式添加 `dataset` 段或叠加
  `configs/data/kaggle.yaml`。详见 [RFC-0006](docs/ARCHITECTURE.md#rfc-0006-make-pskus-the-default-dataset-for-live-yolo-classification)。
- **BREAKING 配置版本 2**：时序头要求 `train.mode=clip` 且 `infer.mode=clip/hybrid`；训练不接受
  `hybrid`（它是推理融合模式）。`model.temporal.stride` 现在同时控制训练、验证和推理的窗口。
  验证 Macro-F1 按实际推理路径（窗口平均、TTA、概率平滑）选 checkpoint。自定义配置需将
  `schema_version` 更新为 2。详见 [RFC-0005](docs/ARCHITECTURE.md#rfc-0005-align-temporal-windows-and-checkpoint-selection-with-inference)。
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
- 修复 GitHub CI：依赖一致性任务安装配置解析依赖；测试夹具同步当前配置版本与标签空间；
  摄像头模型测试移到需 torch 的集成任务；Ruff 规则允许正常中文标点，并放行根目录演示权重 `exp.pt`。
- 合成数据生成的 manifest 现在把 `image_path` 写成相对 `frames_dir` 的路径；
  旧版含重复 `frames/` 前缀的本地产物在训练时会自动重建，避免冒烟训练首批读取失败。
- PSKUS 数据准备叠加配置已改为当前 YOLO 逐帧主线，与训练配置统一使用 5 fps、224 像素、
  JPEG 质量 85 和相同标签空间；移除旧 GRU、clip、fp16/CUDA 参数，避免准备与训练口径不一致。
- README 区分 macOS 本地推理安装与 Linux CUDA 云端训练安装，避免 macOS 直接使用含 CUDA 的环境文件。
- 时序训练：同一视频同一轮的所有帧现在共享随机增强参数，避免独立裁剪、翻转和调色造成模型输入中的假闪烁。
- 修复标签空间名称归一化错误：新增的 `metc_public` 标签空间以前会因下划线被删除而无法加载，现能被数据准备、配置校验与跨域评估正确解析。
- 修复 `class_weights: balanced` 对训练集中缺失类别赋予过大权重的问题；未出现类别现在权重为 0，并明确记录缺失类别。
- 视频写盘：`write_video` 在缺少 ffmpeg 后端时回退写出 GIF，保证"无编码器环境也能生成可解码的测试视频"。
- 训练：`TemporalClassifier` 的 `backbone_kwargs['pretrained']` 之前会被静默丢弃并回退为 `True`，导致离线环境尝试联网下载权重而失败。
- METC：按公开的 0..6 movement code 增加唯一标签映射和七类 `metc_public` 标签空间，保留旧 `metc` 通道顺序；新增同名 JSON 与视频适配及 external 评估读取。
- 训练与评估：真实数据缺失时不再回退到合成样本；checkpoint 结构和关键配置严格校验；时序窗口覆盖尾帧，fp16 梯度累积尾批仍经 GradScaler 更新。
- 数据准备：抽帧标签按原始时间戳对齐，manifest 保留原视频帧号；修复视频时间单位、均匀限帧、预置 split 保护、帧文件自然排序和源视频分组。
- 路径与权重：`data/` 下的配置路径遵循 `HANDWASH_DATA_ROOT`；doctor 不下载 YOLO 权重；根目录 `models/` 忽略规则不再遮蔽 `src/handwash/models/`。

---

## 迁移说明（如果历史实验受上面"变更"影响）

| 受影响的实验 | 需要做什么 |
| --- | --- |
| E2、E3-C/D、E4 等时序模型实验 | **重新训练并评估**：训练窗口从不重叠改为配置步长滑窗，且验证 Macro-F1 使用实际推理输出 |
| 逐帧实验 E1、E3-A/B | 模型训练不受窗口改动影响；若需 schema 2 的 `config_hash`，使用原 checkpoint 重新评估 |
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
