#!/bin/bash
# ============================================================================
#  双击这个文件即可：一键把 PSKUS 数据从零跑到"可以发给队友的分享包"
# ----------------------------------------------------------------------------
#  macOS 用法：在 Finder 里双击本文件。
#  首次使用如果提示"无法打开，因为它来自身份不明的开发者"：
#      右键 → 打开 → 打开；或执行一次
#          chmod +x "scripts/launchers/一键出结果.command"
#
#  也支持带参数：在终端里运行
#      ./scripts/launchers/一键出结果.command --dataset metc
#
#  说明见 docs/DATA_COLLABORATION.md 第 3.0 节。
# ============================================================================

# 无论从哪里双击，都切到脚本所在目录（否则找不到仓库）
cd "$(dirname "$0")" || exit 1

echo
echo "======================================================================"
echo "  六步洗手项目 - 一键准备数据"
echo "======================================================================"
echo
echo "  做四件事（做过的会自动跳过，随时可以关掉窗口，下次双击继续）："
echo "    1. 检查磁盘空间与已下载的分片"
echo "    2. 下载 PSKUS 原始数据（约 17 GB，断点续传 + md5 校验）"
echo "    3. 抽帧 + 按原始视频划分 + 生成 manifest"
echo "    4. 打包成一个压缩包，供上传网盘分享给队友"
echo
echo "  预计耗时：下载看网速，抽帧约 20-40 分钟。"
echo
echo "  重要：只有下全 11 个分片，产出的划分才能给全组使用。"
echo "        只下了一部分时脚本会主动中止，并告诉你怎么补齐。"
echo
read -r -p "按回车开始（Ctrl+C 取消）…… " _

# 找 python：优先 python3（macOS 通常只有 python3）
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo
    echo "  [错误] 找不到 python3 命令。"
    echo
    echo "  请先安装 Miniconda："
    echo "      https://docs.conda.io/en/latest/miniconda.html"
    echo
    echo "  Apple Silicon / Intel 都可以用官方安装包，装完重新双击本文件。"
    echo
    read -r -p "按回车关闭此窗口…… " _
    exit 1
fi

"$PY" -X utf8 "one_click_dataset.py" "$@"
EXITCODE=$?

echo
echo "======================================================================"
if [ "$EXITCODE" -eq 0 ]; then
    echo "  完成。上方「交接清单」里列出了要上传的三个文件和队友要跑的命令。"
else
    echo "  没有全部完成（退出码 $EXITCODE），上方有具体原因。"
    echo
    echo "  若提示「数据不完整」：再双击一次本文件即可，会自动补齐缺失的分片。"
fi
echo "======================================================================"
echo
read -r -p "按回车关闭此窗口…… " _
exit "$EXITCODE"
