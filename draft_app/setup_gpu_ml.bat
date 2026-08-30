@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)

set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo.
echo Installing the current official PyTorch CUDA 12.8 wheel...
echo This is a large download. PyTorch stays in the project .venv and is not bundled into the EXE.
"%PY%" -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :nightly

"%PY%" gpu_probe.py --human
if not errorlevel 1 goto :ready

echo.
echo The stable wheel installed but did not pass the CUDA test.
echo Trying the official CUDA 12.8 nightly, which may add support for newer GPU architectures...

:nightly
"%PY%" -m pip install --pre --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu128
if errorlevel 1 goto :failed
"%PY%" gpu_probe.py --human
if errorlevel 1 goto :cuda_failed

:ready
echo.
echo GPU machine-learning support is ready.
echo Open Model ^& Features and click Rebuild analytics + model.
pause
exit /b 0

:cuda_failed
echo.
echo PyTorch is installed but CUDA did not pass the smoke test.
echo Update the NVIDIA Game Ready or Studio driver, restart Windows, then run check_gpu_ml.bat.
pause
exit /b 2

:failed
echo.
echo GPU machine-learning setup failed. Review the error above.
pause
exit /b 1
