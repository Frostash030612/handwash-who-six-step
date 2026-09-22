> [English](/docs/DATA_COLLABORATION.md) | **中文**

# 不上传 GitHub 也能多人协作取用数据

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
| **2. 抽帧结果 + manifest** | 1–3 GB | **共享介质**（§3）：OneDrive / 实验室 NAS / 点对点 | Git 放不下，但每台机器重新生成一次太贵 |
| **3. 划分、配置、指标** | 几 MB | **Git** | 这才是让结果可比的东西，必须版本化、可评审 |

**第 3 层才是学术可复现的关键。** 两次实验可比，靠的是相同的 `config_hash`、固定种子、
**以及同一个视频级划分** —— 而不是"你我都碰巧有同一批 mp4"。
把划分放进 Git，就算视频一个字节都没流动过，科学结论也是可核验的。

---

## 3. 第 2 层：共享介质，四选一

### 方案 A —— OneDrive / SharePoint 共享文件夹（学校提供的话最省事）

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
dvc remote add -d storage "E:/handwash-dvc"     # 或指向某个 S3/R2/OneDrive 同步目录
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
# 0) 一次性：告诉代码你的数据放在哪（仓库里一个字节都不用有）
$env:HANDWASH_DATA_ROOT = "D:\handwash-data"

# 1) 看有哪些文件、哪些已经校验通过
python scripts/download_data.py --dataset pskus --list

# 2) 只下分给你的分片（断点续传 + md5 校验）
python scripts/download_data.py --dataset pskus --files DataSet1.zip,DataSet4.zip

# 3) 顺手把小的也下了（很便宜地补齐全貌）
python scripts/download_data.py --dataset metc          # 1.98 GB
python scripts/download_data.py --dataset kaggle        # 约 300 MB

# 4) 解压、按原始视频划分、生成 manifest
python scripts/prepare_data.py --config configs/data/pskuss.yaml

# 5) 把 MANIFEST + 划分提交进 Git（很小，而且这才是关键协作契约）
git add data/processed/pskuss/manifest.csv data/processed/pskuss/split_report.json
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
