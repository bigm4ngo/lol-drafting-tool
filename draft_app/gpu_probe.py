"""Probe PyTorch/CUDA from the project virtual environment.

The packaged EXE intentionally excludes PyTorch, so the GUI launches this helper
with the shared project's .venv Python and reads its JSON result.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any


def probe() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_installed": False,
        "cuda_available": False,
        "device_count": 0,
        "device": "none",
        "torch_version": "none",
        "cuda_runtime": "none",
        "capability": "n/a",
        "smoke_test": "not run",
    }
    try:
        import torch
    except Exception as exc:
        payload["error"] = f"PyTorch import failed: {exc}"
        return payload

    payload["torch_installed"] = True
    payload["torch_version"] = str(torch.__version__)
    payload["cuda_runtime"] = str(torch.version.cuda or "none")
    try:
        available = bool(torch.cuda.is_available())
        payload["cuda_available"] = available
        payload["device_count"] = int(torch.cuda.device_count())
        if available:
            payload["device"] = str(torch.cuda.get_device_name(0))
            payload["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            # Small operation catches architecture/driver incompatibility that a
            # simple is_available() check can miss.
            x = torch.randn((512, 512), device="cuda")
            payload["smoke_test"] = float((x @ x).mean().item())
        else:
            payload["device"] = "CPU only"
            payload["error"] = (
                "PyTorch is installed, but CUDA is unavailable. Update the NVIDIA "
                "driver, then rerun setup_gpu_ml.bat."
            )
    except Exception as exc:
        payload["cuda_available"] = False
        payload["error"] = f"CUDA smoke test failed: {exc}"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()
    payload = probe()
    if args.human and not args.json:
        print(f"Python: {payload['python']}")
        print(f"Python version: {payload['python_version']}")
        print(f"PyTorch: {payload['torch_version']}")
        print(f"Built CUDA runtime: {payload['cuda_runtime']}")
        print(f"CUDA available: {payload['cuda_available']}")
        print(f"Device count: {payload['device_count']}")
        print(f"Device: {payload['device']}")
        print(f"Compute capability: {payload['capability']}")
        print(f"CUDA matrix smoke test: {payload['smoke_test']}")
        if payload.get("error"):
            print(f"Error: {payload['error']}")
    else:
        print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload.get("cuda_available") else 2)


if __name__ == "__main__":
    main()
