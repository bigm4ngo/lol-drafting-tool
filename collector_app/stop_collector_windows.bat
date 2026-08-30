@echo off
setlocal
title League Draft Lab collector - stop
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collector_autostart.ps1" -Action stop
pause
