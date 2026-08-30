@echo off
setlocal
title League Draft Lab collector - install auto-start
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collector_autostart.ps1" -Action install
pause
