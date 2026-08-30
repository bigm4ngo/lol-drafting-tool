@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat

echo Verifying required packages inside the project virtual environment...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)
python -c "import customtkinter, lcu_driver, PIL, numpy; import ml_runtime, role_inference, patch_utils, ingest; print('V3.1 packaging imports verified:', customtkinter.__file__)"
if errorlevel 1 (
  echo Required GUI or model packages are missing from .venv.
  pause
  exit /b 1
)

python -m pip install "pyinstaller>=6.10,<7"
if errorlevel 1 (
  echo PyInstaller installation failed.
  pause
  exit /b 1
)

echo Reconciling any legacy API key from the previous dist folder...
python -c "from config_manager import ENV_PATH, reconcile_legacy_api_key; source=reconcile_legacy_api_key(); print('Shared config:', ENV_PATH); print('Migrated from:', source or 'no migration needed')"
if errorlevel 1 (
  echo Could not reconcile the existing API key. The old dist folder was not removed.
  pause
  exit /b 1
)

rmdir /s /q build 2>nul
rmdir /s /q dist\LeagueDraftLab 2>nul
python -m PyInstaller --noconfirm --clean LeagueDraftLab.spec
if errorlevel 1 (
  echo EXE build failed. Review the messages above.
  pause
  exit /b 1
)

rem The packaged EXE and source launcher intentionally share ONE mutable root.
rem From dist\LeagueDraftLab, ..\.. resolves to this main project directory.
> dist\LeagueDraftLab\shared_project_root.txt echo ..\..
copy /y config.env.example dist\LeagueDraftLab\config.env.example >nul
if exist README.md copy /y README.md dist\LeagueDraftLab\README.md >nul

rem Never copy config.env, config_profile.json, or data into dist. Duplicating those
rem files was the cause of source mode and EXE mode using different API keys and
rem databases. The EXE follows shared_project_root.txt instead.

echo Running packaged dependency and shared-path self-test...
del /q dist\LeagueDraftLab\build_self_test.txt 2>nul
start "" /wait "dist\LeagueDraftLab\LeagueDraftLab.exe" --self-test
if errorlevel 1 (
  echo Packaged self-test failed.
  if exist dist\LeagueDraftLab\build_self_test.txt type dist\LeagueDraftLab\build_self_test.txt
  pause
  exit /b 1
)
if not exist dist\LeagueDraftLab\build_self_test.txt (
  echo Packaged self-test did not produce its success marker.
  pause
  exit /b 1
)
type dist\LeagueDraftLab\build_self_test.txt

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$exe=(Resolve-Path 'dist\LeagueDraftLab\LeagueDraftLab.exe').Path;" ^
  "$desktop=[Environment]::GetFolderPath('Desktop');" ^
  "$shell=New-Object -ComObject WScript.Shell;" ^
  "$shortcut=$shell.CreateShortcut((Join-Path $desktop 'League Draft Lab.lnk'));" ^
  "$shortcut.TargetPath=$exe; $shortcut.WorkingDirectory=(Split-Path $exe); $shortcut.IconLocation=\"$exe,0\"; $shortcut.Save()" >nul 2>nul

echo.
echo League Draft Lab v3.1 EXE build and packaged self-test complete:
echo %CD%\dist\LeagueDraftLab\LeagueDraftLab.exe
echo.
echo The EXE uses lol_draft_icon_option_2.ico as its icon.
echo Source mode and EXE mode now share these files in the MAIN project folder:
echo   config.env
echo   config_profile.json
echo   data\
echo   sync_inbox\  and  sync_inbox_processed\
echo.
echo Do not create or edit a second config.env inside dist\LeagueDraftLab.
pause
