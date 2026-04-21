from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

try:
    from .core.paths import BUNDLE_ROOT, IS_FROZEN, RUNTIME_ROOT
    from .ui.main_window import MainWindow
except ImportError:  # pragma: no cover
    # 兼容 PyInstaller 以脚本入口执行时的绝对导入。
    from uestc4006p_gui.core.paths import BUNDLE_ROOT, IS_FROZEN, RUNTIME_ROOT
    from uestc4006p_gui.ui.main_window import MainWindow


def _build_env_hint(language: str = "zh") -> str:
    is_en = str(language).lower().startswith("en")
    if IS_FROZEN:
        if is_en:
            return (
                "[ENV] Runtime mode: PyInstaller onefile. "
                f"executable={sys.executable}, runtime_root={RUNTIME_ROOT}, bundle_root={BUNDLE_ROOT}"
            )
        return (
            "[ENV] 当前运行模式: PyInstaller onefile。"
            f" executable={sys.executable}, runtime_root={RUNTIME_ROOT}, bundle_root={BUNDLE_ROOT}"
        )

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    exe = sys.executable
    if conda_env == "fyp_gui":
        if is_en:
            return f"[ENV] Current interpreter: {exe} | CONDA_DEFAULT_ENV=fyp_gui"
        return f"[ENV] 当前解释器: {exe} | CONDA_DEFAULT_ENV=fyp_gui"
    if is_en:
        return (
            "[ENV][WARN] Current interpreter may not be fyp_gui. "
            f"executable={exe}, CONDA_DEFAULT_ENV={conda_env or 'N/A'}. "
            "Please switch to fyp_gui in VS Code (bottom-right) before running."
        )
    return (
        "[ENV][WARN] 当前解释器可能不是 fyp_gui。"
        f" executable={exe}, CONDA_DEFAULT_ENV={conda_env or 'N/A'}。"
        "请在 VS Code 右下角切换到 fyp_gui 后再运行。"
    )


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.append_startup_hint(_build_env_hint(win.language))
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
