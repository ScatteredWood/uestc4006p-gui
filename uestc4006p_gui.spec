# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


PROJECT_ROOT = Path(globals().get("SPECPATH", Path.cwd())).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
ENTRY_SCRIPT = SRC_ROOT / "uestc4006p_gui" / "app.py"
RUNTIME_HOOK = PROJECT_ROOT / "scripts" / "pyi_rth_runtime.py"

datas = [
    (str(PROJECT_ROOT / "configs" / "default_models.yaml"), "configs"),
]
datas += collect_data_files("PySide6", include_py_files=False)

binaries = []
binaries += collect_dynamic_libs("PySide6")
binaries += collect_dynamic_libs("cv2")

hiddenimports = [
    "cv2",
    "cv2.data",
    "numpy",
    "yaml",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "shiboken6",
]
for optional_mod in ("PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets"):
    if importlib.util.find_spec(optional_mod) is not None:
        hiddenimports.append(optional_mod)

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SRC_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(RUNTIME_HOOK)],
    excludes=["PyQt5", "PyQt6"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="uestc4006p_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
