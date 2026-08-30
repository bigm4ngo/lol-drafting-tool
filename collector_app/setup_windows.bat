@echo off
setlocal
cd /d "%~dp0"
title League Draft Lab collector setup

echo === League Draft Lab - Windows data collector setup ===
echo.
echo This sets up the headless Riot data collector on Windows, for running the
echo whole project (collector + draft app) on one PC.
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11 or 3.12 from python.org,
  echo then run this script again.
  pause
  exit /b 1
)

echo [1/6] Creating virtual environment...
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv || py -3 -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

echo [2/6] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your internet connection.
  pause
  exit /b 1
)

echo [3/6] Refreshing champion + static data for the current patch...
python data_dragon_maps.py
python -c "from static_data import StaticDataCatalog; StaticDataCatalog.load(refresh=True)"
python -c "from config_manager import load_profile; load_profile(); print('config_profile.json ready')"

echo [4/6] Riot API key...
if not exist config.env (
  copy /y config.env.example config.env >nul
  echo Created config.env. Notepad is opening it now: replace the placeholder
  echo with your Riot development key from https://developer.riotgames.com/
  start notepad config.env
) else (
  echo config.env already present.
)

echo.
echo [5/6] Single-device data flow...
echo If the draft app folder (draft_app) sits next to this folder on this same
echo PC, the collector can write its bundles straight into the app's inbox.
choice /C YN /M "Link collector output to draft_app\sync_inbox on this PC"
if errorlevel 2 goto skip_link
python single_device_sync.py --link
goto after_link
:skip_link
python single_device_sync.py --show
:after_link

echo.
echo [6/6] Auto-start at Windows sign-in...
echo The task runs the collector hidden in the background every time you sign
echo in, restarts it if it fails, and has no time limit.
choice /C YN /M "Install the auto-start task (Task Scheduler)"
if errorlevel 2 goto skip_autostart
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collector_autostart.ps1" -Action install
goto after_autostart
:skip_autostart
echo Skipped. Install it later with install_autostart_windows.bat.
:after_autostart

echo.
echo Setup complete.
echo   Run now, visible console : start_collector_windows.bat
echo   Status / DB counts       : status_collector_windows.bat
echo   Stop the collector       : stop_collector_windows.bat
echo   Log file                 : scraper.log in this folder
echo.
pause
