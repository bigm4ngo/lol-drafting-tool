#!/usr/bin/env bash
# Linux/macOS: create .venv with CUDA torch (falls back to CPU torch if no NVIDIA GPU).
set -euo pipefail
cd "$(dirname "$0")"
python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
.venv/bin/pip install -U pip
if command -v nvidia-smi >/dev/null; then
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
else
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
