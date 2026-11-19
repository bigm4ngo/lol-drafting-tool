@echo off
REM Windows: create .venv with CUDA torch (cu128 supports RTX 50xx) and the engine requirements.
cd /d "%~dp0"
py -3.12 -m venv .venv || python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
pause
