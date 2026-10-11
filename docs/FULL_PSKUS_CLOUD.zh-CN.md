# PSKUS 全分片云端训练

本次只扩展 [PSKUS（Zenodo 4537209）](https://zenodo.org/records/4537209)，不把 Jurmala、METC 或 Kaggle 混进训练集。PSKUS 原始记录有 11 个 `DataSet*.zip` 视频分片；两名独立标注者的 CSV 必须存在且时间轴对齐，才能用于训练。全部分片下载后先按 Zenodo MD5 校验，再解压。

## 共用的数据规则

- 第 1、4 分片保留旧模型原来的标注员配对和视频划分：DataSet1 用 Annotator2/3，DataSet4 用 Annotator1/2。这样旧权重见过的训练视频不会被重新划到测试集。
- 其余分片优先选择 Annotator1/2；如果缺少，按标注员编号选择另外两名。**不能根据哪一对更一致来挑**，否则会人为抬高标签质量。
- 只用两人标签一致、且前后各 7 个原始帧没有标签切换的稳定画面训练。`movement_code` 0、7 和非洗手画面合并为 `other`；它不等于单纯没洗手。
- 新增分片按原视频哈希排序做 70/15/15 划分。同一视频的帧不能跨 train、val、test。
- YOLO 训练保留所有六步动作的合格帧；为了避免 `other` 淹没动作类，仅训练集的 `other` 最多保留到与六步动作总帧数 1:1，采用确定性哈希抽样。验证、测试集不做这项下采样。
- VideoMAE 使用同一批原始视频划分和双人一致规则，输入连续 16 帧窗口；YOLO 输入单帧。两者的输入不同，不能把成绩差异单独归因于网络结构。

## DataSet1/3/4 先行试训（2026-10-11）

为避免等全部分片上传，本轮先用 DataSet1、3、4 验证两条训练链路；这不是 11 个分片的全量结果。DataSet3 已上传、校验并解压。双人一致且通过边界过滤后得到 263 段视频，按原视频划分为 train 184、val 39、test 40。

- YOLO26m-cls 从项目 `exp.pt` 微调，最多 30 轮、patience=5。测试集帧级 Top-1 为 0.907、Top-5 为 0.999。测试集 `other` 有 6,017 / 7,430 张图，因此不能只用总体准确率；该数值也不代表整段视频或洗手流程合规率。
- VideoMAE FP16 首次试训在第 3 轮解冻主干后出现 NaN，已停止，失败产物保留在 `outputs/pskuss_ds134_videomae/`。修复版采用 BF16、主干学习率 `1e-6`、分类头学习率 `1e-5`，输出到 `outputs/pskuss_ds134_videomae_stable/`；第 2 轮最佳权重在 1,235 个稳定测试窗口上得到 Accuracy 0.9749、Macro-F1 0.8762。少数动作类窗口数很小，这不是整段视频合规率。
- YOLO 整段视频评估中，验证集选中“宽度 3 + 连续确认 3 帧”的滞回后处理。40 段测试视频共 1,895 个一致标注窗口中心，Macro-F1 从未平滑的 0.6914 提升到 0.7181；切换附近 Macro-F1 从 0.6499 提升到 0.6945。参数只由验证集选择，且不强制 WHO 顺序；这是离线指标，未接入摄像头实时路径。详见 `docs/EXPERIMENTS.md`。
- DataSet2 已完成本地到云端的文件传输，但这版 1/3/4 试训没有纳入 DataSet2。其余分片齐备并校验前，不得把本轮结果称作 PSKUS 全量训练。

## DataSet2 加入 YOLO（2026-10-11，进行中）

`DataSet2.zip` 已在 AutoDL 通过 MD5 校验并解压。新的七类分类数据使用 DataSet1/2/3/4，仍按双人标签一致、动作切换前后各 7 个原始帧稳定、按视频隔离划分。使用 `--split-reference data/processed/pskuss_ds134_yolo/video_splits.csv` 锁定旧 1/3/4 视频的划分，新增加的 DataSet2 视频另行哈希划分，避免拿已见过的旧视频当测试样本。

云端 `data/processed/pskuss_ds1234_yolo/` 已构建完成：615 段双标注视频（train 430、val 92、test 93），共 93,118 张合格图像；新增 DataSet2 352 段，旧 263 段视频的划分和标注员配对核对后均未变化。`pskus_ds1234_yolo` 队列已用 1/3/4 YOLO 最佳权重开始微调（最多 30 轮、patience=5）。队列状态：`outputs/pskuss_ds1234_yolo_queue/status.json`；训练日志：同目录 `train.log`；测试结果：`outputs/pskuss_ds1234_yolo/test_metrics.json`。这仍不是 PSKUS 全 11 分片训练。

VideoMAE 也已排在该 YOLO 任务之后：`pskus_ds1234_videomae` 队列复用同一个 `video_splits.csv`，从 1/3/4 修复版 VideoMAE 最佳权重继续微调。使用 16 帧窗口；双标注一致与边界稳定规则检查出 22,378 个窗口。训练集 `other` 窗口从 13,349 个按固定哈希抽到 2,194 个（与六步动作总数 1:1），验证和测试保持原始分布，总计进入本轮的窗口数为 11,223。队列状态在 `outputs/pskuss_ds1234_videomae_queue/status.json`，模型和最终测试在 `outputs/pskuss_ds1234_videomae/`；只有队列状态为 `complete` 才能引用最终结果。

## AutoDL 后台任务

四个 `screen` 会话 `pskus_dl1`～`pskus_dl4` 负责互不重叠的大文件。`pskus_pipeline` 等待它们完成，然后自动执行：全部文件再校验和补传 → 解压 → 构建七类 YOLO 图像 → 继续训练 YOLO → 完整视频测试 → 继续训练 VideoMAE → 完整视频测试。每一步失败即停止，不能拿不完整数据继续训练。

```bash
# 登录 AutoDL 后查看进度
screen -ls
cd /root/autodl-tmp/handwash
python scripts/download_data.py --status
cat outputs/pskuss_full_pipeline/status.json

# 各阶段详细日志
ls outputs/pskuss_full_pipeline/*.log
```

云端数据盘只有 50GB；VideoMAE 解码帧缓存特意放在系统盘 `/root/pskuss_full_videomae_cache`，避免和约 17GB 的原始 ZIP、解压视频以及 YOLO 图像争用数据盘。若实际抽帧大小超出预估，构建阶段会停止并报告剩余空间，不会自动删除原始数据。

预期产物：

- `data/raw/pskuss/SOURCES.json`：14 个公开文件的 MD5 和校验状态。
- `data/processed/pskuss_full_yolo/summary.json`：视频划分及各类图像数量。
- `outputs/pskuss_full_yolo/weights/best.pt`、`test_metrics.json`：YOLO 权重及稳定帧测试。
- `outputs/pskuss_full_videomae/best_model/`、`test_metrics.json`：VideoMAE 权重及稳定窗口测试。
- `outputs/pskuss_full_yolo_video_test/summary.json` 和 `outputs/pskuss_full_videomae_video_test/summary.json`：动作切换也保留的整段视频测试。

注：下载和训练是长任务。查看 `status.json` 的 `state` 为 `complete` 且最终阶段 `all` 完成，才能说两种模型均已完成；`running` 不是结果。
