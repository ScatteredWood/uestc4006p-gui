from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Iterator


# 同时兼容源码运行与 PyInstaller onefile 运行。
IS_FROZEN = bool(getattr(sys, "frozen", False))
if IS_FROZEN:
    RUNTIME_ROOT = Path(sys.executable).resolve().parent
    BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", RUNTIME_ROOT))
else:
    RUNTIME_ROOT = Path(__file__).resolve().parents[3]
    BUNDLE_ROOT = RUNTIME_ROOT

PROJECT_ROOT = RUNTIME_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_DIR = RUNTIME_ROOT / "configs"
CACHE_DIR = RUNTIME_ROOT / "cache"
OUTPUTS_DIR = RUNTIME_ROOT / "outputs"
LOGS_DIR = RUNTIME_ROOT / "logs"


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

    roots = (
        [RUNTIME_ROOT, BUNDLE_ROOT]
        if prefer_runtime
        else [BUNDLE_ROOT, RUNTIME_ROOT]
    )
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
        RUNTIME_ROOT / expanded,
        PROJECT_ROOT / expanded,
        BUNDLE_ROOT / expanded,
    ]
    dedup_candidates = tuple(_iter_unique_paths(candidates))
    for path in dedup_candidates:
        if path.exists():
            return path
    return dedup_candidates[0] if dedup_candidates else expanded


def _build_ultralytics_repo_candidates() -> tuple[Path, ...]:
    env_repo = os.environ.get("UESTC4006P_ULTRALYTICS_REPO", "").strip()
    candidate_paths: list[Path | None] = []
    if env_repo:
        candidate_paths.append(Path(os.path.expandvars(os.path.expanduser(env_repo))))
    candidate_paths.extend(
        [
            RUNTIME_ROOT / "ultralytics",
            RUNTIME_ROOT.parent / "ultralytics",
        ]
    )
    return tuple(_iter_unique_paths(candidate_paths))


ULTRALYTICS_REPO_CANDIDATES = _build_ultralytics_repo_candidates()


def _pick_first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0] if paths else (RUNTIME_ROOT / "ultralytics").resolve()


ULTRALYTICS_REPO = _pick_first_existing(ULTRALYTICS_REPO_CANDIDATES)
ULTRALYTICS_SCRIPTS_DIR = ULTRALYTICS_REPO / "uestc4006p" / "scripts"


def format_ultralytics_repo_candidates() -> str:
    return " | ".join(str(path) for path in ULTRALYTICS_REPO_CANDIDATES)


def ensure_runtime_dirs() -> None:
    """确保运行时目录存在。"""
    for folder in (CACHE_DIR, OUTPUTS_DIR, LOGS_DIR, CONFIG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
