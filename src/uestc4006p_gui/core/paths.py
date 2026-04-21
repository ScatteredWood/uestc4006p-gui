from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Iterator


# 同时兼容源码运行与 PyInstaller onefile 运行。
IS_FROZEN = bool(getattr(sys, "frozen", False))
if IS_FROZEN:
    RUNTIME_ROOT = Path(sys.executable).resolve().parent
    BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", RUNTIME_ROOT)).resolve()
else:
    RUNTIME_ROOT = Path(__file__).resolve().parents[3]
    BUNDLE_ROOT = RUNTIME_ROOT

APP_ORG = "UESTC4006P"
APP_NAME = "GUI"


def _default_user_data_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", "")).expanduser()
        if not str(base):
            base = Path.home() / "AppData" / "Local"
        return (base / APP_ORG / APP_NAME).resolve()
    return (Path.home() / ".local" / "share" / APP_ORG / APP_NAME).resolve()


# 开发态仍写仓库目录，冻结态写用户目录，避免 Program Files 写权限问题。
DATA_ROOT = _default_user_data_root() if IS_FROZEN else RUNTIME_ROOT

PROJECT_ROOT = RUNTIME_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_DIR = DATA_ROOT / "configs"
CACHE_DIR = DATA_ROOT / "cache"
OUTPUTS_DIR = DATA_ROOT / "outputs"
LOGS_DIR = DATA_ROOT / "logs"


def _iter_unique_paths(paths: Iterable[Path | None]) -> Iterator[Path]:
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        p = Path(path).resolve()
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        yield p


def iter_resource_candidates(relative_path: str | Path, prefer_runtime: bool = True) -> tuple[Path, ...]:
    rel = Path(relative_path)
    if rel.is_absolute():
        return (rel.resolve(),)

    roots = [DATA_ROOT, BUNDLE_ROOT] if prefer_runtime else [BUNDLE_ROOT, DATA_ROOT]
    return tuple((root / rel).resolve() for root in _iter_unique_paths(roots))


def resolve_resource_path(
    relative_path: str | Path,
    *,
    prefer_runtime: bool = True,
    must_exist: bool = False,
) -> Path:
    candidates = iter_resource_candidates(relative_path, prefer_runtime=prefer_runtime)
    for path in candidates:
        if path.exists():
            return path
    if must_exist:
        raise FileNotFoundError(f"资源文件不存在: {relative_path}, candidates={candidates}")
    return candidates[0]


DEFAULT_MODELS_REL = Path("configs") / "default_models.yaml"
DEFAULT_MODELS_YAML_CANDIDATES = iter_resource_candidates(DEFAULT_MODELS_REL, prefer_runtime=True)
DEFAULT_MODELS_YAML = resolve_resource_path(DEFAULT_MODELS_REL, prefer_runtime=True, must_exist=False)


def resolve_configured_path(raw_value: str, *, base_dir: Path | None = None) -> Path | None:
    value = (raw_value or "").strip()
    if not value:
        return None

    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute():
        return expanded.resolve()

    candidates = [
        (base_dir / expanded) if base_dir else None,
        DATA_ROOT / expanded,
        RUNTIME_ROOT / expanded,
        BUNDLE_ROOT / expanded,
    ]
    dedup_candidates = tuple(_iter_unique_paths(candidates))
    for path in dedup_candidates:
        if path.exists():
            return path
    return dedup_candidates[0] if dedup_candidates else expanded


def _copy_default_config_if_needed() -> None:
    runtime_cfg = (DATA_ROOT / DEFAULT_MODELS_REL).resolve()
    bundled_cfg = (BUNDLE_ROOT / DEFAULT_MODELS_REL).resolve()
    if runtime_cfg.exists():
        return
    if not bundled_cfg.exists():
        return
    runtime_cfg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled_cfg, runtime_cfg)


def ensure_runtime_dirs() -> None:
    """确保运行时目录存在。"""
    for folder in (CACHE_DIR, OUTPUTS_DIR, LOGS_DIR, CONFIG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    if IS_FROZEN:
        _copy_default_config_if_needed()
