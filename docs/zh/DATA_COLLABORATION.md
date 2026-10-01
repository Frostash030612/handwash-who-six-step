> [English](/docs/DATA_COLLABORATION.md) | **中文**

# 不上传 GitHub 也能多人协作取用数据

> 下文 2 fps 的体积估算属于旧的压缩共享方案。当前动态 YOLO 分类主线在
> `configs/data/pskuss.yaml` 与 `configs/experiments/live_yolo_frame.yaml` 中统一使用
> 5 fps、224 像素、JPEG 质量 85。

数据集有 2–17 GB，绝不能进 Git，而 Git LFS 也顶不住（§5 有算式）。
本文说明四个人怎么在**没有任何人托管数据**的前提下，协作使用同一份数据。

---

## 1. 关键认识：你们已经有数据托管方了

本项目用到的每个数据集都发在 **Zenodo 上，有匿名直链**，而且**每个文件都带 md5 校验值**：

| 数据集 | 记录号 | 体积 | 分片 | 校验值 |
| --- | --- | --- | --- | --- |
| PSKUS | [4537209](https://zenodo.org/records/4537209) | 17.14 GB | 11 个 `DataSet*.zip` + 3 个 csv | ✅ 每文件 md5 |
| METC | [5808789](https://zenodo.org/records/5808789) | 1.98 GB | 3 个 `Interface_number_*.zip` + 2 个 csv | ✅ 每文件 md5 |
| Jurmala | [5808764](https://zenodo.org/records/5808764) | 15.79 GB | 4 个 `Location_*.zip` + 2 个 csv | ✅ 每文件 md5 |

不用登录、不用申请、不会过期、没有配额。

所以协作问题**不是**"17 GB 放哪儿"，而是：

> 怎么保证四个人的数据**字节完全一致、划分完全相同**？

这个问题的答案很便宜：**把来源和校验值记进 Git，各自从来源下载。**
Git 当**索引**，不当**仓库**。

### PSKUS 本来就切好了分片 —— 直接用

```text
DataSet4.zip    107.2 MB    ← 单个分片就可以作为起点
DataSet3.zip    450.2 MB
DataSet1.zip    555.6 MB
DataSet7.zip    641.2 MB
DataSet9.zip    858.1 MB
DataSet11.zip  1076.2 MB
DataSet10.zip  1241.4 MB
DataSet2.zip   2161.8 MB
DataSet5.zip   2454.9 MB
DataSet6.zip   3529.4 MB
DataSet8.zip   4475.3 MB
```

**没人需要下全 17 GB。** 把分片分给四个人 —— 每人下 3–4 GB，各自处理自己的分片，
处理产物进共享介质（§3）。

---

## 2. 三层状态，各放各的地方

最容易犯的错是把"数据"当成一个东西。它其实是三种东西，体积和共享需求完全不同：

| 层 | 体积 | 放哪 | 为什么 |
| --- | --- | --- | --- |
| **1. 原始源** | 2–17 GB | **公开来源**（Zenodo），把 URL + md5 记进 `data/raw/SOURCES.json`（提交进 Git） | 公开、不可变、可校验。随时能重下，所以谁都不需要在 Git 里留副本 |
| **2. 抽帧结果 + manifest** | 视抽帧率而定；当前 5 fps 估算约 6.4 GB | **共享介质**（§3）：OneDrive / 实验室 NAS / 点对点 | Git 放不下，但每台机器重新生成一次太贵 |
| **3. 划分、配置、指标** | 几 MB | **Git** | 这才是让结果可比的东西，必须版本化、可评审 |

**第 3 层才是学术可复现的关键。** 两次实验可比，靠的是相同的 `config_hash`、固定种子、
**以及同一个视频级划分** —— 而不是"你我都碰巧有同一批 mp4"。
把划分放进 Git，就算视频一个字节都没流动过，科学结论也是可核验的。

---

## 3. 第 2 层：共享介质

> **本项目采用的做法**：由一个人下载全量原始数据、**抽一次帧**、打包，把包传到网盘；
> 其余三人下载同一个包。实测体积与完整命令见 **§3.0**，先看它；后面的方案是备选。

### 3.0 推荐做法：一个人抽帧，一个包，一个链接

**为什么先抽帧再上传。** 原始数据是 17.1 GiB、11 个 zip 分片（最大 4.4 GB），上传又慢又脆。
而大家真正需要的是抽帧结果，它小得多：

| 抽帧配置 | PSKUS 全集体积 | 说明 |
| --- | --- | --- |
| 5 fps, 256 px, q92 | 约 10.4 GB | 比预想的大 —— 见下方提示 |
| **5 fps, 224 px, q85** | **约 6.4 GB** | **当前实时 YOLO 配置** |
| 2 fps, 224 px, q85 | 约 2.8 GB | 旧的压缩共享方案 |
| 1 fps, 224 px, q85 | 约 1.4 GB | 太粗，做不了"某步时长不足"的细粒度判定 |

> **一个反直觉的实测发现：** PSKUS 原视频分辨率只有 **320×240**，
> 所以抽帧**并不会自动变小** —— 决定体积的是**帧率**，不是分辨率。
> 上表数字来自"对已下载数据实测每帧 jpg 字节数 × 官方 summary.csv 的真实总时长"
> （3,185 段 / 139,881 秒 / 原生 30 fps），并且已计入 `max_frames_per_clip` 截断。
> 随时可以复核：
>
> ```bash
> python scripts/pack_processed_data.py --dataset pskuss --report
> ```

**这里有两种角色，跑的命令不同。** 只看属于你的那一行。

| 你的角色 | 你跑什么 | 为什么 |
| --- | --- | --- |
| **A. 抽帧的人**（一个人，R1） | `bootstrap_dataset.py` | 需要完整的原始数据才能产出权威划分 |
| **B. 其余三人** | `pack_processed_data.py --verify` 然后 `--unpack` | 只需要抽好的帧；不用下载、不用抽帧 |

**抽帧的人有三种启动方式，挑顺手的用。**

| 方式 | 适合谁 | 做什么 |
| --- | --- | --- |
| **双击** `scripts/launchers/一键出结果.bat`（Windows）或 `一键出结果.command`（macOS） | 不想碰命令行的人 | 跑完整的权威流程，没有任何参数可以填错 |
| `python scripts/bootstrap_dataset.py --dataset pskuss` | 熟悉终端的人 | 同样的事，另外支持全部变体参数 |
| 分别跑四个子脚本 | 调试某一阶段 | `download_data.py` → `prepare_data.py` → `pack_processed_data.py` |

双击入口**刻意除了 `--dataset` 不接受任何参数**。会去双击文件的人，通常不会读参数说明，
而这里唯一会致命的错误 —— "用只下了一部分的数据去抽帧" —— 产出的划分**看起来完全正常**，
却让全组实验数字都不可比。所以双击入口永远跑完整流程。

```bash
# ---- 角色 A 方式一：双击（Windows）----
scripts\launchers\一键出结果.bat

# ---- 角色 A 方式二：一条命令 ----
python scripts/bootstrap_dataset.py --dataset pskuss
```

**抽帧之前，数据不完整会直接中止。** 这是唯一一处故意中止的检查，
因为它是唯一一个"错了也不会有任何报错"的错误：

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! 数据不完整 —— 不能产出权威划分
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  视频分片：3/11 个已就绪
  缺失    ：DataSet3.zip, DataSet7.zip, DataSet9.zip, ...
  怎么办，二选一：
    1) 下全数据后重跑（推荐）：python scripts/bootstrap_dataset.py --dataset pskuss
    2) 只想先验证链路（结果不可用于正式实验）：加 --allow-partial
```

这个脚本**可以放心反复重跑**，每个阶段都会自己判断已完成的部分：

| 已经有的东西 | 重跑时会怎样 |
| --- | --- |
| 已校验通过的分片 | **校验 md5 后跳过**。校验失败的分片会被删掉以便重新下载，而不是被静默使用 |
| 已抽好的帧 + `manifest.csv` | **整个抽帧阶段直接跳过**（它是最慢的一步）。要故意重抽请加 `--force-prepare` |
| 已存在的压缩包 | 重新打包，旧包在被替换前一直保留 |

所以**中断了、或者只下了一部分分片，再跑同一条命令就能续上** —— 不需要先清理任何东西。

```bash
python scripts/bootstrap_dataset.py --dataset pskuss --dry-run         # 只打印计划，不改动任何东西
python scripts/bootstrap_dataset.py --dataset pskuss --skip-download   # 数据已在本地
python scripts/bootstrap_dataset.py --dataset pskuss --skip-pack       # 只要帧，自己训练
python scripts/bootstrap_dataset.py --dataset pskuss --force-prepare   # 故意重新抽帧
python scripts/bootstrap_dataset.py --dataset pskuss \
    --shards DataSet4.zip,DataSet3.zip                                 # 约 1.2 GB 验证链路
```

> **下全分片才产出"权威划分"。** 只下部分分片时脚本照样会跑完，但会打印醒目警告：
> 它基于子集产生的划分与别人的不一致，实验数字就不可比了。
> 用 `--shards` 验证链路，然后去掉这个参数重跑，才产出真正的划分。

**第 3 步就是整个方案的价值所在：一次上传约 2.8 GB，而不是 17.1 GB**，
其余三人各下一个文件，不用去协调 11 个分片。

```bash
# 其余三人
python scripts/pack_processed_data.py --dataset pskuss --verify pskuss_frames_<日期>.zip
python scripts/pack_processed_data.py --dataset pskuss --unpack pskuss_frames_<日期>.zip
python scripts/train_model.py config=configs/experiments/exp02_yolo26n_gru.yaml \
    config=configs/data/pskuss.yaml
```

包参数支持**裸文件名**（会自动在 `<data_root>/packages/` 下查找）、相对路径、完整路径，
或通配符如 `pskuss_frames_*.zip`。找不到时，报错会列出所有查找位置和该目录下实际存在的包，
省得靠猜。

> **注意配置叠加顺序：实验配置在前，数据配置在后。**
> `config=` 是依次深合并、后者覆盖前者，而 `configs/experiments/*.yaml` 里也写了
> `dataset.name`。把数据配置放前面会被覆盖，训练就会静默地跑在默认数据集上。
> `bootstrap_dataset.py` 会把正确顺序的命令直接打印给你。

**为什么包里要连划分一起带。** 包里含 `frames/`、`manifest.csv`、`clip_splits.json`、
`split_report.json` 和 `SOURCES.json`。manifest 保存相对 `frames_dir` 的图像路径；
`clip_splits.json` 冻结原始视频的归属，之后若重抽帧仍会沿用同一份划分。
解包后四个人使用同一测试集，结果才能比较。

> **解包之后不要再跑 `prepare_data.py`。** 重新抽帧会重新采样，可能得到不同的划分，
> 那你的结果就会在不知不觉中变得不可比。解包，然后训练。

### 3.1 方案 A —— OneDrive / SharePoint 共享文件夹（学校提供的话最省事）

如果 NUS 给了 OneDrive/SharePoint 组空间，这是首选。校园机器上已经装好，不用引入新工具，
而且"按需同步"意味着每个人只把自己要用的拉下来。

```text
<OneDrive 共享文件夹>/
  datasets/
    pskus/                 从 Zenodo 下的原始 zip，不改动
    metc/
    kaggle/
  processed/
    pskus/frames/          抽出来的帧（这才是真正省时间的部分）
    pskus/manifest.csv
    pskus/clip_splits.json
    pskus/split_report.json
  checkpoints/             每次实验的 best.pt，让别人也能评估任何模型
  outputs/                 每次运行的 resolved_config.yaml + metrics.json
```

规矩：**原始 zip 可以放这儿，但 `processed/` 才是真正省时间的**——
从 11 个分片抽帧要跑几个小时，不该有人重复第二遍。

### 方案 B —— 局域网 SSH/SCP 点对点（不借第三方、不上传）

每个人在本机跑一个 Python 文件服务，或者直接用 `scp`。不用账号、没有配额，
数据不出校园网。同一个实验室或同一个 Wi-Fi 下最实用。

```bash
# 已经处理好数据的人对外提供：
python scripts/share_data.py --serve data/processed/pskuss --port 8000

# 其他人拉取（带校验）：
python scripts/share_data.py --pull http://<对方IP>:8000 --into data/processed/pskuss
```

### 方案 C —— DVC + 非 GitHub 的 remote

DVC 用内容哈希给数据版本化，Git 里只留很小的指针文件，真正的字节放到"remote"——
可以是共享盘、S3 兼容桶，甚至一个普通目录。

```bash
pip install dvc
dvc init
dvc remote add -d storage "<shared-storage-location>"  # 替换成你自己的存储位置
dvc add data/processed/pskuss
git add data/processed/pskuss.dvc && git commit -m "data: 固化 PSKUS 处理集版本"
dvc push
```

只有当你**真的需要多份数据版本**（比如用不同 `fps` 重建过帧，还得复现旧结果）时才选 DVC。
如果只有一份冻结的处理集，方案 A 用更少的机械换来同样的结果。

> **绝对不要**把 DVC remote 指向 GitHub —— 那等于把 LFS 的配额问题原样搬回来。

### 方案 D —— 各自下同一份源（零基础设施）

最朴素的兜底：每人对着 Zenodo 下载**分给自己的分片**，划分放 Git。
墙钟时间更慢，但完全不需要共享存储，而且校验值能证明大家数据一致。

如果全组异地、又没有共享盘，就选这个。

---

## 4. 要校验，不要相信

整套方案建立在一个习惯上：**md5 对不上，就不算下载完。**

```
期望（Zenodo API）  md5:be7d8776e6222705175298b7dbf720c6   DataSet1.zip
实际（下载后）      md5:be7d8776e6222705175298b7dbf720c6   ✅ 一致
```

`scripts/download_data.py` 会自动做这件事：跳过已校验通过的文件、断点续传。
它还会写出 `data/raw/SOURCES.json`：

```json
{
  "pskuss": {
    "record": "https://zenodo.org/records/4537209",
    "title": "Hand Washing Video Dataset Annotated According to WHO Guidelines",
    "license": "CC-BY-SA-4.0",
    "retrieved": "2026-09-22",
    "files": [
      { "key": "DataSet1.zip", "size": 582624798, "md5": "be7d8776e6222705175298b7dbf720c6", "verified": true }
    ]
  }
}
```

**把 `SOURCES.json` 提交进 Git。** 它只有几 KB，却是让你能在报告里写下
"用了哪份数据、哪个版本、何时获取、如何校验"的东西 ——
这正是官方交付物清单里 "Datasets" 那一项要考的意思。

---

## 5. 为什么不用 Git LFS（算给你看）

GitHub Free 的 Git LFS 额度是 **10 GiB 存储 + 10 GiB/月流量**
（[GitHub 官方文档](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)）。

```
光 PSKUS 一个                    17.14 GB  >  10 GiB 存储额度      ← 已经超了
四个组员各克隆一次               68.6  GiB  >  10 GiB/月 流量       ← 超 7 倍
```

而且流量额度用尽且账户没绑支付方式时，**LFS 支持会被停用到当月月底** ——
队友直接拉不到文件。另外还有单文件 100 MiB 的硬限制，而 `DataSet8.zip` 是 4.4 GB。

注意这个不对称：**上传不消耗流量，但每一次下载都消耗，而且记在仓库所有者头上。**
所以把数据放进 LFS，等于用配额为每个队友、每次 CI 运行付费。

LFS **适合**小的、可从源码派生的二进制：几 MB 的模型权重、报告 PDF。
**不适合**一个本来就有永久公开主页的 17 GB 数据集。

---

## 6. 每个成员具体做什么

```powershell
# 0) 默认数据目录是 data/；只有搬到别处时才设置 HANDWASH_DATA_ROOT

# 1) 看有哪些文件、哪些已经校验通过
python scripts/download_data.py --dataset pskuss --list

# 2) 只下分给你的分片（断点续传 + md5 校验）
python scripts/download_data.py --dataset pskuss --files DataSet1.zip,DataSet4.zip

# 3) 顺手把小的也下了（很便宜地补齐全貌）
python scripts/download_data.py --dataset metc          # 1.98 GB
python scripts/download_data.py --dataset kaggle        # 约 300 MB

# 4) 解压、按原始视频划分、生成 manifest
python scripts/prepare_data.py --config configs/data/pskuss.yaml

# 5) 把 MANIFEST + 划分提交进 Git（很小，而且这才是关键协作契约）
git add data/processed/pskuss/manifest.csv data/processed/pskuss/clip_splits.json data/processed/pskuss/split_report.json
git commit -m "data: PSKUS 子集 manifest 与视频级划分"
```

第 1–4 步是每台机器各自做；第 5 步是协作契约。

---

## 7. 选型对照

| 你的情况 | 用哪个 |
| --- | --- |
| 学校提供 OneDrive/SharePoint | **方案 A** |
| 同一个实验室 / 同一个局域网，不想借外部服务 | **方案 B** |
| 需要复现历史数据版本 | **方案 C**（DVC + 非 GitHub remote） |
| 全组异地、没有共享存储 | **方案 D**，把分片分给四个人 |
| 只是要传一个 5 MB 的 checkpoint | Git LFS 没问题（或者直接提交，低于 100 MiB 限制） |

无论选哪个，有两条规矩不能破：

1. **`SOURCES.json` 和划分必须在 Git 里。** 没有它们，什么都不可复现。
2. **原始数据和帧图像绝不进 Git。** 走 LFS 也不行。
