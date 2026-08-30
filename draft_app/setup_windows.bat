@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11 or 3.12 from python.org.
  pause
  exit /b 1
)
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python data_dragon_maps.py
python -c "from static_data import StaticDataCatalog; StaticDataCatalog.load(refresh=True)"
if not exist config.env echo RIOT_API_KEY=^>config.env
echo.
echo Setup complete. Run launch_app.bat.
echo For RTX GPU model training, also run setup_gpu_ml.bat.
pause
