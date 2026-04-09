from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.paths import ULTRALYTICS_REPO, ULTRALYTICS_SCRIPTS_DIR


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


def _validate_path_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise RuntimeError(
            f"{desc} 不存在: {path}\n"
            "请检查本机目录是否正确，或设置环境变量 UESTC4006P_ULTRALYTICS_REPO。"
        )


def get_bridge_objects() -> BridgeObjects:
    global _BRIDGE_CACHE
    if _BRIDGE_CACHE is not None:
        return _BRIDGE_CACHE

    _validate_path_exists(ULTRALYTICS_REPO, "ultralytics 仓库目录")
    _validate_path_exists(ULTRALYTICS_SCRIPTS_DIR, "ultralytics scripts 目录")
    _inject_sys_paths()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "无法导入 ultralytics.YOLO。请确认 VS Code 使用的是 fyp_gui 环境。"
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
            "请确认 scripts 路径存在且依赖可在当前解释器导入。"
        ) from exc

    _BRIDGE_CACHE = BridgeObjects(
        YOLO=YOLO,
        cascade_one_image_v3c=cascade_one_image_v3c,
        draw_det_boxes=draw_det_boxes,
        overlay_mask_red=overlay_mask_red,
        postprocess_mask=postprocess_mask,
    )
    return _BRIDGE_CACHE