@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)

.venv\Scripts\python.exe scraper.py --validate-key
set EXIT_CODE=%ERRORLEVEL%
echo.
if "%EXIT_CODE%"=="0" (
  echo The saved Riot API key is valid.
) else (
  echo The saved Riot API key was not accepted. Open Settings,
  echo paste a newly generated development key, then save the settings.
)
pause
exit /b %EXIT_CODE%
