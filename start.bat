@echo off
rem ============================================================
rem  QQ 2D Billiard Aim launcher v2.7 (ASCII + CRLF only)
rem  Double-click the vbs launcher in this folder to run this
rem  file. Markers: startup_trace.log (this file), startup.log
rem  and runtime.log (written by main.py).
rem ============================================================
pushd "%~dp0" >nul 2>&1 || goto :fatal
setlocal EnableExtensions
set "TRACE=startup_trace.log"
echo [%date% %time%] LAUNCHER v2.7 >> "%TRACE%"
echo [%time%] cwd=%CD% >> "%TRACE%"
echo [%time%] arg0=%~f0 >> "%TRACE%"
echo === QQ 2D Billiard Aim launcher v2.7 ===

rem ---- find a working python (each candidate tried at most once) ----
set "PYTHON_EXE="
call :try "%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE call :try "C:\Python314\python.exe"
if not defined PYTHON_EXE call :try "C:\Python313\python.exe"
if not defined PYTHON_EXE call :try "C:\Python312\python.exe"
if not defined PYTHON_EXE call :try "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PYTHON_EXE call :try "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYTHON_EXE call :try "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do call :try "%%P"
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do call :try "%%P"
if not defined PYTHON_EXE goto :nopython
goto :pyok

:try
if not exist "%~1" goto :eof
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON_EXE=%~1"
echo [%time%] python="%~1" >> "%TRACE%"
goto :eof

:nopython
echo [%time%] NO WORKING PYTHON FOUND >> "%TRACE%"
echo [ERROR] Cannot find a working Python 3 on this PC.
echo        Tell me and I will read startup_trace.log.
pause
endlocal
exit /b 1

:pyok
echo [PYTHON] "%PYTHON_EXE%"
"%PYTHON_EXE%" --version
echo [%time%] python="%PYTHON_EXE%" >> "%TRACE%"

rem ---- dependency check / auto install (first start may take a while) ----
echo === checking dependencies ===
"%PYTHON_EXE%" -c "import numpy, cv2, mss, PIL, tkinter" >nul 2>&1
set "DEPS_RC=%errorlevel%"
echo [%time%] deps_rc=%DEPS_RC% >> "%TRACE%"
if "%DEPS_RC%"=="0" goto :deps_recheck
echo [INFO] Installing dependencies (first start, please wait 1-3 minutes)...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
set "DEPS_RC=%errorlevel%"
echo [%time%] pip_rc=%DEPS_RC% >> "%TRACE%"
if "%DEPS_RC%"=="0" goto :deps_recheck
echo [INFO] Retrying install for current user only...
"%PYTHON_EXE%" -m pip install --user -r "%~dp0requirements.txt"
set "DEPS_RC=%errorlevel%"
echo [%time%] pip2_rc=%DEPS_RC% >> "%TRACE%"
:deps_recheck
"%PYTHON_EXE%" -c "import numpy, cv2, mss, PIL, tkinter" >nul 2>&1
set "DEPS_RC=%errorlevel%"
echo [%time%] deps2_rc=%DEPS_RC% >> "%TRACE%"
if not "%DEPS_RC%"=="0" (
    echo [%time%] DEPS STILL MISSING >> "%TRACE%"
    echo [ERROR] Dependencies still incomplete. Tell me and I will read startup_trace.log.
    pause
    endlocal
    exit /b 1
)
echo        Dependencies OK

rem ---- launch main.py in the background (pythonw; main.py logs to runtime.log) ----
echo === starting the aim overlay ===
set "QQ_AIM_HEADLESS=1"
echo [%time%] launching "%PYTHON_EXE%" -u "%~dp0main.py" >> "%TRACE%"
for %%I in ("%PYTHON_EXE%") do set "PYTHONW_EXE=%%~dpIpythonw.exe"
if exist "%PYTHONW_EXE%" (
    echo [%time%] pythonw="%PYTHONW_EXE%" >> "%TRACE%"
    start "aim-overlay" /b "%PYTHONW_EXE%" -u "%~dp0main.py"
) else (
    rem Some Python distributions omit pythonw; retain a diagnosable fallback.
    echo [%time%] pythonw_missing_fallback_console >> "%TRACE%"
    start "aim-overlay" /min "%PYTHON_EXE%" -u "%~dp0main.py"
)
echo [%time%] start returned >> "%TRACE%"
echo Started OK. Quit with Esc. Log: runtime.log
endlocal
exit /b 0

:fatal
echo [FATAL] Cannot enter folder: %~dp0
pause
exit /b 1
