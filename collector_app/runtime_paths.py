"""Stable shared paths for the headless Ubuntu collector.

The collector runs from source under systemd (or manually), so the application
root is the ``collector_app`` directory itself. The Windows EXE/shared-root
logic is not applicable here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_MARKERS = (
    "config_profile.json",
    "collector_daemon.py",
    "outbox",
    "data",
)


def executable_directory() -> Path:
    """Return the directory containing the running module."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def application_root() -> Path:
    """Return the collector root directory."""
    return executable_directory()


EXECUTABLE_DIR = executable_directory()
PROJECT_ROOT = application_root()
DATA_DIR = PROJECT_ROOT / "data"
