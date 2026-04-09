from __future__ import annotations

import os
from pathlib import Path


# 项目根目录: .../uestc4006p-gui
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_MODELS_YAML = CONFIG_DIR / "default_models.yaml"

# 允许通过环境变量覆盖参考仓库路径。
ULTRALYTICS_REPO = Path(
    os.environ.get("UESTC4006P_ULTRALYTICS_REPO", r"E:\repositories\ultralytics")
)
ULTRALYTICS_SCRIPTS_DIR = ULTRALYTICS_REPO / "uestc4006p" / "scripts"


def ensure_runtime_dirs() -> None:
    """确保运行时目录存在。"""
    for folder in (CACHE_DIR, OUTPUTS_DIR, LOGS_DIR, CONFIG_DIR):
        folder.mkdir(parents=True, exist_ok=True)