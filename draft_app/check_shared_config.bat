@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "from runtime_paths import PROJECT_ROOT, EXECUTABLE_DIR; from config_manager import ENV_PATH, PROFILE_PATH, api_key_fingerprint, read_api_key; from ingest import resolve_inbox; print('Shared runtime root :', PROJECT_ROOT); print('Python/EXE directory:', EXECUTABLE_DIR); print('API key file        :', ENV_PATH); print('Profile file        :', PROFILE_PATH); print('Saved key           :', api_key_fingerprint(read_api_key())); print('Sync inbox          :', resolve_inbox())"
pause
