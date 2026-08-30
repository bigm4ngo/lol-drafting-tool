@echo off
setlocal
cd /d "%~dp0"
title League Draft Lab collector status

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
echo === Collected data ===
.venv\Scripts\python.exe collector_daemon.py --status
echo.
echo === Auto-start task / daemon ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collector_autostart.ps1" -Action status
echo.
pause
