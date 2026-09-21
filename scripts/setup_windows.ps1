# Windows 一键初始化：conda 环境 + 可编辑安装 + pre-commit + 自检
# ---------------------------------------------------------------------------
# 用法（在仓库根目录，PowerShell 中执行）：
#     pwsh -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
#     或双击运行（若已允许脚本执行）
#
# 设计原则：不静默失败。任何一步失败都会打印原因并退出，不会继续往下装。
# 只做"环境搭建"，不改任何源文件。

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Write-Warn2($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow }
function Write-Fail($text) { Write-Host "  [FAIL] $text" -ForegroundColor Red }

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Host "仓库根目录：$repoRoot"

# --- 1) 前提检查 ------------------------------------------------------------
Write-Step "1/6 检查前提条件"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Fail "找不到 python。请先安装 Miniconda 并把它加入 PATH。"
    exit 1
}
Write-Ok ("python: " + (& python --version 2>&1))

$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    Write-Warn2 "找不到 conda。将跳过环境创建，直接在当前 Python 里做可编辑安装。"
    Write-Warn2 "强烈建议安装 Miniconda：https://docs.conda.io/en/latest/miniconda.html"
}

$envFile = Join-Path $repoRoot "environment.yml"
if (-not (Test-Path $envFile)) {
    Write-Fail "找不到 environment.yml，确认脚本位于 scripts/ 目录下。"
    exit 1
}

# --- 2) 创建/更新 conda 环境 -----------------------------------------------
Write-Step "2/6 创建或更新 conda 环境（handwash）"
if ($conda) {
    & conda env list | Out-String | Write-Host
    try {
        & conda env update -f $envFile --prune
        Write-Ok "conda 环境已就绪"
    } catch {
        Write-Fail "conda env update 失败：$_"
        Write-Warn2 "常见原因：网络问题、conda 源不可用、CUDA 版本不匹配。"
        Write-Warn2 "纯 CPU 机器请把 environment.yml 里的 pytorch-cuda / cudatoolkit 两行删掉后重试。"
        exit 1
    }
} else {
    Write-Warn2 "跳过 conda 环境创建。"
}

# --- 3) 可编辑安装本仓库 ----------------------------------------------------
Write-Step "3/6 以可编辑模式安装本仓库"
try {
    & python -m pip install --upgrade pip --quiet
    & python -m pip install -e ".[all]"
    Write-Ok "handwash 已安装（可编辑模式）"
} catch {
    Write-Fail "pip install -e 失败：$_"
    Write-Warn2 "若只是 ultralytics 装不上，可先跑：pip install -e `".[torch,video,viz,dev]`""
    exit 1
}

# --- 4) pre-commit 钩子 ----------------------------------------------------
Write-Step "4/6 安装提交前钩子（pre-commit）"
try {
    & python -m pre_commit install
    Write-Ok "pre-commit 钩子已安装"
} catch {
    Write-Warn2 "pre-commit 安装失败（通常不影响开发）：$_"
}

# --- 5) 目录占位 ------------------------------------------------------------
Write-Step "5/6 创建数据与产物目录"
foreach ($dir in @("data\raw", "data\interim", "data\processed", "data\external",
                   "models", "outputs")) {
    $full = Join-Path $repoRoot $dir
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Ok "已创建 $dir"
    }
}

# --- 6) 自检 ----------------------------------------------------------------
Write-Step "6/6 环境自检（handwash doctor）"
& python -m handwash.cli doctor
$doctorExit = $LASTEXITCODE

Write-Host ""
Write-Host "----------------------------------------"
if ($doctorExit -eq 0) {
    Write-Host "初始化完成。建议下一步：" -ForegroundColor Green
} else {
    Write-Host "初始化基本完成，但 doctor 报出必须解决的问题（见上方 FAIL 项）。" -ForegroundColor Yellow
    Write-Host "请按提示补齐后重新运行 doctor。建议下一步：" -ForegroundColor Yellow
}
Write-Host "  python -m handwash.cli train config=configs/experiments/smoke.yaml   # 30 秒冒烟"
Write-Host "  然后阅读 CONTRIBUTING.md（修改规则，必读）"
Write-Host "----------------------------------------"

exit $doctorExit
