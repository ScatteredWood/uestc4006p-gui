from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_env_path(key: str, path: Path) -> None:
    if not path.exists():
        return
    current = os.environ.get(key, "")
    value = str(path)
    if current:
        os.environ[key] = value + os.pathsep + current
    else:
        os.environ[key] = value


def configure_frozen_runtime() -> None:
    if not getattr(sys, "frozen", False):
        return

    bundle_root = Path(getattr(sys, "_MEIPASS", "")).resolve()
    if not bundle_root.exists():
        return

    pyside_root = bundle_root / "PySide6"
    plugin_root = pyside_root / "plugins"
    platform_root = plugin_root / "platforms"
    qml_root = pyside_root / "qml"

    if platform_root.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platform_root))
    if plugin_root.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_root))
    if qml_root.exists():
        os.environ.setdefault("QML2_IMPORT_PATH", str(qml_root))

    _prepend_env_path("PATH", bundle_root)
    _prepend_env_path("PATH", pyside_root)


configure_frozen_runtime()
