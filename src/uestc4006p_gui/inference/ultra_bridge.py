from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BridgeObjects:
    YOLO: Any
    cascade_one_image_v3c: Any
    draw_det_boxes: Any
    overlay_mask_red: Any
    postprocess_mask: Any


_BRIDGE_CACHE: BridgeObjects | None = None


def get_bridge_objects() -> BridgeObjects:
    global _BRIDGE_CACHE
    if _BRIDGE_CACHE is not None:
        return _BRIDGE_CACHE

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "无法导入 ultralytics.YOLO。"
            "请确认运行环境已安装 ultralytics、torch 及其依赖。"
        ) from exc

    try:
        from .cascade_ops import (
            cascade_one_image_v3c,
            draw_det_boxes,
            overlay_mask_red,
            postprocess_mask,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "无法导入内置级联推理模块（cascade_ops）。"
            "请确认程序安装完整。"
        ) from exc

    _BRIDGE_CACHE = BridgeObjects(
        YOLO=YOLO,
        cascade_one_image_v3c=cascade_one_image_v3c,
        draw_det_boxes=draw_det_boxes,
        overlay_mask_red=overlay_mask_red,
        postprocess_mask=postprocess_mask,
    )
    return _BRIDGE_CACHE
