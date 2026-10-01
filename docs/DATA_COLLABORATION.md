> **English** | [中文](/docs/zh/DATA_COLLABORATION.md)

# Data Collaboration Without Uploading Anything

> The 2 fps sharing estimate below describes an older compact data package. The current
> live YOLO frame classifier uses 5 fps, 224 px, JPEG quality 85 in
> `configs/data/pskuss.yaml` and `configs/experiments/live_yolo_frame.yaml`.

The datasets are 2–17 GB. They must never enter Git, and Git LFS is not a workable substitute
(see §5 for the arithmetic). This document describes how four people collaborate on the same data
without anyone hosting it.

---

## 1. The key realisation: you already have a data host

Every dataset used by this project is published on **Zenodo with anonymous direct download
links**, and every file ships with an **md5 checksum**:

| Dataset | Record | Size | Shards | Checksums |
| --- | --- | --- | --- | --- |
| PSKUS | [4537209](https://zenodo.org/records/4537209) | 17.14 GB | 11 × `DataSet*.zip` + 3 csv | ✅ md5 per file |
| METC | [5808789](https://zenodo.org/records/5808789) | 1.98 GB | 3 × `Interface_number_*.zip` + 2 csv | ✅ md5 per file |
| Jurmala | [5808764](https://zenodo.org/records/5808764) | 15.79 GB | 4 × `Location_*.zip` + 2 csv | ✅ md5 per file |

Nobody logs in, nobody asks permission, nothing expires, and there is no quota.

So the collaboration problem is **not** "where do we host 17 GB". It is:

> How do we guarantee that all four members end up with **byte-identical data, split identically**?

That question has a cheap answer: **record the source and the checksum in Git, and let each
person download from the source.** Git becomes the *index*, not the *store*.

### PSKUS is already sharded — use that

```text
DataSet4.zip    107.2 MB    ← a single shard is a legitimate starting point
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

No member needs all 17 GB. **Split the shards across the team** — each person downloads 3–4 GB,
processes their shards, and the processed artefacts go into the shared medium (§3).

---

## 2. The three layers of state, and where each lives

The mistake to avoid is treating "data" as one thing. It is three things with different sizes and
different sharing needs:

| Layer | Size | Where it lives | Why |
| --- | --- | --- | --- |
| **1. Raw source** | 2–17 GB | **The public source** (Zenodo), pinned by URL + md5 in `data/raw/SOURCES.json` (committed) | Public, immutable, verifiable. Re-downloading is always possible, so nobody needs a copy in Git |
| **2. Processed frames + manifest** | Depends on sampling rate; about 6.4 GB at the current 5 fps setting | **A shared medium** (§3): OneDrive / SharePoint / lab NAS / peer-to-peer | Too big for Git, too expensive to regenerate on every machine |
| **3. Splits, configs, metrics** | a few MB | **Git** | This is what makes results comparable. It must be versioned and reviewable |

**Layer 3 is the one that actually matters for academic reproducibility.** Two runs are comparable
when they share a `config_hash`, a fixed seed, **and the same video-level split** — not because you
both happen to have the same mp4 files. Put the split in Git and the science is verifiable even if
the videos never move.

---

## 3. Layer 2: the shared medium

> **The setup this project uses:** one person downloads the full raw dataset, extracts frames
> **once**, packs the result, and uploads that package to cloud storage. Everyone else downloads
> the one package. Measured sizes and the exact commands are in **§3.0** — read that first; the
> options after it are alternatives.

### 3.0 Recommended: one extractor, one archive, one link

**Why extract before uploading.** The raw dataset is 17.1 GiB across 11 zip shards (the largest is
4.4 GB), which is slow and fragile to upload. The extracted frames are what everyone actually needs,
and they are much smaller:

| Extraction setting | Full PSKUS frame set | Note |
| --- | --- | --- |
| 5 fps, 256 px, q92 | ~10.4 GB | larger than you would expect — see the note below |
| **5 fps, 224 px, q85** | **~6.4 GB** | **current live YOLO profile** |
| 2 fps, 224 px, q85 | ~2.8 GB | older compact sharing profile |
| 1 fps, 224 px, q85 | ~1.4 GB | too coarse for per-step duration checks |

> **Surprise worth knowing:** PSKUS source video is only **320×240**, so extracting frames does
> *not* automatically shrink the data — frame **rate** controls the size, not resolution.
> These figures come from measuring real jpg sizes on downloaded data and multiplying by the
> official `summary.csv` totals (3,185 clips / 139,881 s / native 30 fps), including the
> `max_frames_per_clip` capping. Re-check any time with:
>
> ```bash
> python scripts/pack_processed_data.py --dataset pskuss --report
> ```

**There are two roles, and they run different commands.** Read the row that applies to you.

| Your role | What you run | Why |
| --- | --- | --- |
| **A. The extractor** (one person, R1) | `bootstrap_dataset.py` | Needs the full raw dataset to produce the authoritative split |
| **B. Everyone else** | `pack_processed_data.py --verify` then `--unpack` | Only needs the finished frames; no download, no extraction |

**The extractor has three ways to start, pick whichever suits you.**

| How | Who it is for | What it does |
| --- | --- | --- |
| **Double-click** `scripts/launchers/一键出结果.bat` (Windows) or `一键出结果.command` (macOS) | anyone who does not want to touch a terminal | Runs the full, authoritative pipeline with no options to get wrong |
| `python scripts/bootstrap_dataset.py --dataset pskuss` | comfortable with a terminal | Same thing, plus every variant flag |
| The four sub-scripts individually | debugging a single stage | `download_data.py` → `prepare_data.py` → `pack_processed_data.py` |

The double-click entry **deliberately accepts no options** beyond `--dataset`. Someone
double-clicking a file is unlikely to read a flag list, and the one mistake that matters here —
extracting from a partial download — produces a split that *looks* fine but silently makes every
team member's numbers incomparable. So the launcher always runs the complete path.

```bash
# ---- Role A, option 1: double-click (Windows) ----
scripts\launchers\一键出结果.bat

# ---- Role A, option 2: one command ----
python scripts/bootstrap_dataset.py --dataset pskuss
```

**Before frame extraction the script now stops if the data is incomplete.** This is the only
check that aborts on purpose, because it is the only mistake that produces no error at all:

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

It is designed to be **re-run without fear**. Each stage detects what is already done:

| Already present | What happens on a re-run |
| --- | --- |
| Verified shard files | **md5-checked and skipped.** A shard that fails its checksum is deleted so it gets re-downloaded rather than silently used |
| Extracted frames + `manifest.csv` | **Frame extraction is skipped entirely** (it is the slowest stage). Pass `--force-prepare` to deliberately re-extract |
| An existing archive | Repacked; the old one stays in place until replaced |

So an interrupted run, or a run that only covered some shards, resumes by running the *same*
command again — nothing needs to be cleaned up first.

```bash
python scripts/bootstrap_dataset.py --dataset pskuss --dry-run         # print the plan, change nothing
python scripts/bootstrap_dataset.py --dataset pskuss --skip-download   # data already on disk
python scripts/bootstrap_dataset.py --dataset pskuss --skip-pack       # frames for your own training
python scripts/bootstrap_dataset.py --dataset pskuss --force-prepare   # re-extract on purpose
python scripts/bootstrap_dataset.py --dataset pskuss \
    --shards DataSet4.zip,DataSet3.zip                                 # ~1.2 GB pipeline test
```

> **Downloading every shard is what makes the split authoritative.** If you fetch only some
> shards the script still finishes, but prints a loud warning: the split it produced is based on a
> subset, so it will not match anyone else's and the numbers stop being comparable. Use
> `--shards` to validate the pipeline, then re-run without it to produce the real split.

```bash
# ---- Role B: teammates ----
python scripts/pack_processed_data.py --dataset pskuss --verify pskuss_frames_<date>.zip
python scripts/pack_processed_data.py --dataset pskuss --unpack pskuss_frames_<date>.zip
python scripts/train_model.py config=configs/experiments/exp02_yolo26n_gru.yaml \
    config=configs/data/pskuss.yaml
```

The archive argument accepts a **bare filename** (looked up in `<data_root>/packages/`), a relative
path, a full path, or a glob such as `pskuss_frames_*.zip`. If it cannot be found, the error lists
every location searched and every package that does exist.

> **Mind the config overlay order:** the **experiment** config first, the **data** config second.
> Overlays are merged in order and later ones win, and every `configs/experiments/*.yaml` also
> sets `dataset.name`. Put the data config first and it gets overwritten — training then silently
> runs on the default dataset. `bootstrap_dataset.py` prints the correct command for you.

**Why the archive carries the split too.** The package contains `frames/`, `manifest.csv`,
`clip_splits.json`, `split_report.json` **and** `SOURCES.json`. The manifest stores paths relative
to `frames_dir`; `clip_splits.json` freezes the source-video assignment for any later frame rebuild.
Unpacking therefore preserves one canonical test set for all four members.

> **Do not run `prepare_data.py` again after unpacking.** Re-extracting would resample frames and
> can produce a different split, which silently makes your results incomparable. Unpack, train.

### 3.1 Option A — OneDrive / SharePoint shared folder (simplest if your institution provides it)

Recommended if NUS gives you a OneDrive/SharePoint group. It is already installed on campus
machines, requires no new tooling, and Sync-on-demand means each member only pulls what they use.

```text
<OneDrive shared folder>/
  datasets/
    pskus/                 raw zips, unchanged from Zenodo
    metc/
    kaggle/
  processed/
    pskus/frames/          extracted frames (the expensive-to-regenerate part)
    pskus/manifest.csv
    pskus/clip_splits.json
    pskus/split_report.json
  checkpoints/             best.pt per experiment, so anyone can evaluate any model
  outputs/                 resolved_config.yaml + metrics.json per run
```

Rule: **raw zips may live here, but `processed/` is the part that saves real time** — extracting
frames from 11 shards takes hours, and nobody should repeat it.

### Option B — Peer-to-peer over SSH/SCP (no third party, no upload)

Every member runs a Python file server or uses `scp` on the LAN. No accounts, no quotas, nothing
leaves the campus network. Practical when you are in the same lab or on the same Wi-Fi.

```bash
# The member who processed the data serves it:
python scripts/share_data.py --serve data/processed/pskuss --port 8000

# The others pull it, with checksum verification:
python scripts/share_data.py --pull http://<peer-ip>:8000 --into data/processed/pskuss
```

### Option C — DVC with a non-GitHub remote

DVC versions data by content hash and keeps only small pointer files in Git; the actual bytes go to
a "remote" that can be a shared drive, an S3-compatible bucket, or even a plain directory.

```bash
pip install dvc
dvc init
dvc remote add -d storage "<shared-storage-location>"  # replace with your storage location
dvc add data/processed/pskuss
git add data/processed/pskuss.dvc && git commit -m "data: version PSKUS processed set"
dvc push
```

Choose DVC only if you genuinely need **multiple data versions** (e.g. you rebuild frames with
different `fps` settings and must reproduce an older result). For a single frozen processed set,
Option A is less machinery for the same outcome.

> **Never** point the DVC remote at GitHub — that reintroduces exactly the LFS quota problem.

### Option D — Everyone downloads the same source (zero infrastructure)

The honest fallback: each member runs the download script against Zenodo for **their assigned
shards**, and the split stays in Git. Slower in wall-clock terms, but it needs no shared storage at
all, and the checksum verification proves everyone has identical data.

This is the right choice if the team is fully remote and no shared drive is available.

---

## 4. Verify, don't trust

The whole scheme rests on one habit: **a download is not finished until its md5 matches.**

```
expected (from Zenodo API)  md5:be7d8776e6222705175298b7dbf720c6   DataSet1.zip
actual   (after download)   md5:be7d8776e6222705175298b7dbf720c6   ✅ identical
```

`scripts/download_data.py` does this automatically, skips files already verified, and resumes
partial downloads. It also writes `data/raw/SOURCES.json`:

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

**Commit `SOURCES.json`.** It is a few KB, and it is the artefact that lets you write in the report:
*which* data, *which* version, *when* retrieved, *verified how*. That is exactly what an examiner
means by "datasets" in the deliverables list.

---

## 5. Why not Git LFS (the arithmetic)

Git LFS on GitHub Free includes **10 GiB of storage and 10 GiB/month of bandwidth**
([GitHub docs](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)).

```
PSKUS alone                     17.14 GB  >  10 GiB storage quota        ← already over
Four members each cloning       68.6  GiB  >  10 GiB/month bandwidth     ← 7× over
```

And when the bandwidth quota is exhausted without a payment method on file, **LFS support is
disabled for the rest of the month** — collaborators cannot fetch files at all. There is also a
hard 100 MiB per-file push limit, and `DataSet8.zip` is 4.4 GB.

Note the asymmetry: uploading does **not** consume bandwidth, but **every download does, and it is
charged to the repository owner**. So hosting the data in LFS means paying — in quota — for every
teammate and every CI run that touches it.

LFS *is* the right tool for small derivable binaries: a few MB of model weights, a report PDF.
It is the wrong tool for a 17 GB public dataset that already has a permanent home.

---

## 6. What each member does (concrete)

```powershell
# 0) Data is stored under data/ by default; set HANDWASH_DATA_ROOT only if you moved it

# 1) See what is available and what you already have verified
python scripts/download_data.py --dataset pskuss --list

# 2) Download only your assigned shards (resumable, md5-verified)
python scripts/download_data.py --dataset pskuss --files DataSet1.zip,DataSet4.zip

# 3) Also grab the small ones (they cheaply complete the picture)
python scripts/download_data.py --dataset metc          # 1.98 GB
python scripts/download_data.py --dataset kaggle        # ~300 MB

# 4) Extract, split by source video, build the manifest
python scripts/prepare_data.py --config configs/data/pskuss.yaml

# 5) Publish the MANIFEST + SPLIT to Git (small, and this is the part that matters)
git add data/processed/pskuss/manifest.csv data/processed/pskuss/clip_splits.json data/processed/pskuss/split_report.json
git commit -m "data: PSKUS subset manifest and video-level split"
```

Steps 1–4 are per-machine. Step 5 is the collaboration contract.

---

## 7. Decision guide

| Your situation | Use |
| --- | --- |
| Institution gives you OneDrive/SharePoint | **Option A** |
| Same lab / same LAN, want nothing external | **Option B** |
| Need to reproduce older data versions | **Option C** (DVC, non-GitHub remote) |
| Fully remote, no shared storage | **Option D**, with shards divided across the team |
| Need to share a 5 MB checkpoint | Git LFS is fine (or just commit it, under the 100 MiB limit) |

Whatever you choose, two rules are non-negotiable:

1. **`SOURCES.json` + the split are in Git.** Without them, nothing is reproducible.
2. **Raw data and frame images are never in Git.** Not even via LFS.
