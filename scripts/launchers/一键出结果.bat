@REM ====================================================================
@REM  六步洗手项目 - 一键准备数据（双击本文件即可）
@REM ====================================================================
@REM  做四件事，做过的会自动跳过；随时可以关掉窗口，下次双击继续：
@REM    1. 检查磁盘空间与已下载的分片
@REM    2. 下载 PSKUS 原始数据（约 17 GB，断点续传 + md5 校验）
@REM    3. 抽帧 + 按原始视频划分 + 生成 manifest
@REM    4. 打包成压缩包，供上传网盘分享给队友
@REM
@REM  重要：只有下全 11 个分片，产出的划分才能给全组使用。
@REM        只下了一部分时脚本会主动中止，并告诉你怎么补齐。
@REM
@REM  本文件只负责"双击友好"的部分，真正的逻辑在 one_click_dataset.py
@REM  和 scripts/bootstrap_dataset.py。见 docs/DATA_COLLABORATION.md 第 3.0 节。
@REM ====================================================================

@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo ======================================================================
echo   六步洗手项目 - 一键准备数据
echo ======================================================================
echo.
echo   做四件事（做过的会自动跳过，随时可以关掉窗口，下次双击继续）：
echo     1. 检查磁盘空间与已下载的分片
echo     2. 下载 PSKUS 原始数据（约 17 GB，断点续传 + md5 校验）
echo     3. 抽帧 + 按原始视频划分 + 生成 manifest
echo     4. 打包成一个压缩包，供上传网盘分享给队友
echo.
echo   预计耗时：下载看网速，抽帧约 20-40 分钟。
echo.
echo   重要：只有下全 11 个分片，产出的划分才能给全组使用。
echo         只下了一部分时脚本会主动中止，并告诉你怎么补齐。
echo.
pause

where python >nul 2>nul
if errorlevel 1 goto nopython

python -X utf8 "%~dp0one_click_dataset.py" %*
set EXITCODE=%ERRORLEVEL%

echo.
echo ======================================================================
if "%EXITCODE%"=="0" (
  echo   完成。上方「交接清单」里列出了要上传的三个文件和队友要跑的命令。
) else (
  echo   没有全部完成（退出码 %EXITCODE%），上方有具体原因。
  echo.
  echo   若提示「数据不完整」：再双击一次本文件即可，会自动补齐缺失的分片。
  echo   若提示找不到 python：见下方说明。
)
echo ======================================================================
echo.
pause
exit /b %EXITCODE%

:nopython
echo.
echo   [错误] 找不到 python 命令。
echo.
echo   请先安装 Miniconda，安装时勾选 "Add to PATH"：
echo       https://docs.conda.io/en/latest/miniconda.html
echo.
echo   装好后重新双击本文件即可。
echo.
pause
exit /b 1
