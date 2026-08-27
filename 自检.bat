@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title QQ 2D桌球 斯诺克瞄准器 - 自检
echo ==============================================
echo    QQ 2D桌球 斯诺克瞄准器  自检
echo ==============================================
echo.

rem 和 start.bat 一样解析真实 Python：优先 .venv，其次 py -3，最后 PATH。
rem 直接 where python 会命中微软商店的假 python（无参数时会弹商店），不能用。
set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE (
    echo [错误] 未找到 Python，请先安装 python.org 官方版 Python 3.10+
    pause
    endlocal
    exit /b 1
)
echo [PYTHON] "%PYTHON_EXE%"
"%PYTHON_EXE%" --version

echo.
echo [1/2] 单元测试...
"%PYTHON_EXE%" -m pytest tests/ -q
if errorlevel 1 (
    echo [错误] 单元测试失败。
    pause
    endlocal
    exit /b 1
)
echo.
echo [2/2] 合成台面自检...
"%PYTHON_EXE%" main.py --demo
if errorlevel 1 (
    echo [错误] 合成台面自检失败。
    pause
    endlocal
    exit /b 1
)
echo.
echo ============ 自检完成 ============
pause
endlocal
exit /b 0