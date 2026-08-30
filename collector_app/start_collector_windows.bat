@echo off
setlocal
cd /d "%~dp0"
title League Draft Lab collector

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Starting the collector with a visible console. Press Ctrl+C (or close this
echo window) to stop it. Logs also go to scraper.log in this folder.
echo.
.venv\Scripts\python.exe collector_daemon.py
if errorlevel 1 pause
