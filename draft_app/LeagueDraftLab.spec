# Robust PyInstaller onedir build for League Draft Lab v3.1.
# CustomTkinter ships non-Python theme/font assets, so its package directory is
# explicitly bundled in addition to hidden module discovery.

from pathlib import Path

import customtkinter
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

project_root = Path(SPECPATH).resolve()
ctk_root = Path(customtkinter.__file__).resolve().parent
icon_path = project_root / "lol_draft_icon_option_2.ico"

ctk_datas, ctk_binaries, ctk_hidden = collect_all("customtkinter")
lcu_datas, lcu_binaries, lcu_hidden = collect_all("lcu_driver")

datas = list(ctk_datas) + list(lcu_datas)
datas.append((str(ctk_root), "customtkinter"))
if icon_path.is_file():
    datas.append((str(icon_path), "."))
for distribution in ("customtkinter", "lcu-driver"):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

hiddenimports = sorted(set(
    [
        "customtkinter",
        "lcu_driver",
        "lcu_driver.connector",
        "lcu_driver.connection",
        "lcu_driver.websocket",
        "ingest",
    ]
    + ctk_hidden
    + lcu_hidden
    + collect_submodules("customtkinter")
    + collect_submodules("lcu_driver")
))

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=ctk_binaries + lcu_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "torch", "torchvision", "torchaudio"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LeagueDraftLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LeagueDraftLab",
)
