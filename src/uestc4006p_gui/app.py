from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def _build_env_hint() -> str:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    exe = sys.executable
    if conda_env == "fyp_gui":
        return f"[ENV] 当前解释器: {exe} | CONDA_DEFAULT_ENV=fyp_gui"
    return (
        "[ENV][WARN] 当前解释器可能不是 fyp_gui。"
        f" executable={exe}, CONDA_DEFAULT_ENV={conda_env or 'N/A'}。"
        "请在 VS Code 右下角切换到 fyp_gui 后再运行。"
    )


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.append_startup_hint(_build_env_hint())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
