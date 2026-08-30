"""Stable shared paths for source runs and PyInstaller builds.

The source launcher and the packaged EXE must use the same mutable files.  A
PyInstaller onedir build normally runs from ``dist/LeagueDraftLab``; in that
layout this module follows ``shared_project_root.txt`` (written by
``build_exe.bat``) back to the main project directory.  Consequently both
launch methods read one config.env, one profile, one SQLite database, and one
set of caches.

If an EXE folder is copied away from the project, the marker no longer resolves
to a valid project and the application safely falls back to storing its files
beside that EXE.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED_ROOT_MARKER = "shared_project_root.txt"
_PROJECT_MARKERS = (
    "config_profile.json",
    "launch_app.bat",
    "build_exe.bat",
    "data",
)


def executable_directory() -> Path:
    """Return the directory containing the running EXE or Python module."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _looks_like_project_root(path: Path) -> bool:
    """Return True when *path* resembles a League Draft Lab project root."""
    return path.is_dir() and any((path / marker).exists() for marker in _PROJECT_MARKERS)


def _root_from_marker(exe_dir: Path) -> Path | None:
    marker = exe_dir / _SHARED_ROOT_MARKER
    if not marker.is_file():
        return None
    try:
        relative_or_absolute = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not relative_or_absolute:
        return None
    candidate = Path(relative_or_absolute)
    if not candidate.is_absolute():
        candidate = exe_dir / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    return candidate if _looks_like_project_root(candidate) else None


def _auto_detect_source_root(exe_dir: Path) -> Path | None:
    """Recognise the normal ``<project>/dist/LeagueDraftLab`` layout."""
    if exe_dir.parent.name.casefold() != "dist":
        return None
    candidate = exe_dir.parent.parent
    return candidate.resolve() if _looks_like_project_root(candidate) else None


def application_root() -> Path:
    """Return the single writable root shared by source mode and EXE mode."""
    exe_dir = executable_directory()
    if not getattr(sys, "frozen", False):
        return exe_dir
    return _root_from_marker(exe_dir) or _auto_detect_source_root(exe_dir) or exe_dir


EXECUTABLE_DIR = executable_directory()
PROJECT_ROOT = application_root()
DATA_DIR = PROJECT_ROOT / "data"
