from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []


def _prepend_env_path(key: str, path: Path) -> None:
    if not path.exists():
        return
    current = os.environ.get(key, "")
    value = str(path)
    if current:
        os.environ[key] = value + os.pathsep + current
    else:
        os.environ[key] = value


def _register_dll_directory(path: Path) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if not path.exists():
        return
    try:
        # Keep returned handles alive for the whole process lifetime.
        _DLL_HANDLES.append(os.add_dll_directory(str(path)))
    except Exception:
        pass


def configure_frozen_runtime() -> None:
    if not getattr(sys, "frozen", False):
        return

    bundle_root = Path(getattr(sys, "_MEIPASS", "")).resolve()
    if not bundle_root.exists():
        return

    # Let PyInstaller's built-in pyi_rth_pyside6 configure Qt-related env vars.
    # This hook only ensures native DLL discovery is robust in frozen mode.
    dll_dirs = (
        bundle_root,
        bundle_root / "PySide6",
        bundle_root / "shiboken6",
        bundle_root / "torch" / "lib",
        bundle_root / "cv2",
    )

    for dll_dir in dll_dirs:
        _prepend_env_path("PATH", dll_dir)
        _register_dll_directory(dll_dir)


configure_frozen_runtime()
