@echo off
setlocal
title League Draft Lab collector - remove auto-start
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collector_autostart.ps1" -Action remove
pause
