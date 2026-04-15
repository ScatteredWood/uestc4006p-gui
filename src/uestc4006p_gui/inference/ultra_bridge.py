from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.paths import (
    ULTRALYTICS_REPO,
    ULTRALYTICS_REPO_CANDIDATES,
    ULTRALYTICS_SCRIPTS_DIR,
    format_ultralytics_repo_candidates,
)


@dataclass(frozen=True)
class BridgeObjects:
    YOLO: Any
    cascade_one_image_v3c: Any
    draw_det_boxes: Any
    overlay_mask_red: Any
    postprocess_mask: Any


_BRIDGE_CACHE: BridgeObjects | None = None


def _inject_sys_paths() -> None:
    """
    仅在 bridge 中注入 sys.path，其他模块禁止修改 sys.path。
    """
    for path in (ULTRALYTICS_REPO, ULTRALYTICS_SCRIPTS_DIR):
        if not path.exists():
            continue
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)


def _validate_bridge_paths() -> None:
    if ULTRALYTICS_REPO.exists() and ULTRALYTICS_SCRIPTS_DIR.exists():
        return

    checked = format_ultralytics_repo_candidates()
    if not checked:
        checked = str(ULTRALYTICS_REPO)

    raise RuntimeError(
        "未找到可用的 ultralytics 推理脚本目录。\n"
        f"当前仓库路径: {ULTRALYTICS_REPO}\n"
        f"期望脚本路径: {ULTRALYTICS_SCRIPTS_DIR}\n"
        f"已检查候选路径: {checked}\n"
        "请将 ultralytics 仓库放到以上任一路径，或设置环境变量 "
        "UESTC4006P_ULTRALYTICS_REPO。"
    )


def get_bridge_objects() -> BridgeObjects:
    global _BRIDGE_CACHE
    if _BRIDGE_CACHE is not None:
        return _BRIDGE_CACHE

    _validate_bridge_paths()
    _inject_sys_paths()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "无法导入 ultralytics.YOLO。"
            "请确认当前 Python 环境已安装 ultralytics 及其依赖。"
        ) from exc

    try:
        from cascade_infer_detseg import (
            cascade_one_image_v3c,
            draw_det_boxes,
            overlay_mask_red,
            postprocess_mask,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "无法导入 cascade_infer_detseg 所需对象。"
            f"请确认脚本目录存在并可导入: {ULTRALYTICS_SCRIPTS_DIR}\n"
            f"候选仓库路径: {' | '.join(str(p) for p in ULTRALYTICS_REPO_CANDIDATES)}"
        ) from exc

    _BRIDGE_CACHE = BridgeObjects(
        YOLO=YOLO,
        cascade_one_image_v3c=cascade_one_image_v3c,
        draw_det_boxes=draw_det_boxes,
        overlay_mask_red=overlay_mask_red,
        postprocess_mask=postprocess_mask,
    )
    return _BRIDGE_CACHE
