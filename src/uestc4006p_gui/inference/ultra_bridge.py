from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

from ..core.paths import ULTRALYTICS_REPO, ULTRALYTICS_SCRIPTS_DIR


@dataclass(frozen=True)
class BridgeObjects:
    YOLO: Any
    cascade_one_image_v3c: Any
    draw_det_boxes: Any
    overlay_mask_red: Any


_BRIDGE_CACHE: BridgeObjects | None = None


def _inject_sys_paths() -> None:
    """
    只在本文件内注入 sys.path。
    其他模块禁止直接修改 sys.path。
    """
    candidates = [ULTRALYTICS_REPO, ULTRALYTICS_SCRIPTS_DIR]
    for path in candidates:
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


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
    except Exception as exc:  # pragma: no cover - 错误分支用于运行时提示
        raise RuntimeError(
            "无法导入 ultralytics.YOLO。请确认 VS Code 使用的是 fyp_gui 环境，"
            "并且该环境可访问本地 ultralytics 代码。"
        ) from exc

    try:
        from cascade_infer_detseg import (
            cascade_one_image_v3c,
            draw_det_boxes,
            overlay_mask_red,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "无法导入 cascade_infer_detseg 所需对象。"
            "请确认 E:\\repositories\\ultralytics\\uestc4006p\\scripts 存在，"
            "且脚本依赖项可在当前解释器中导入。"
        ) from exc

    _BRIDGE_CACHE = BridgeObjects(
        YOLO=YOLO,
        cascade_one_image_v3c=cascade_one_image_v3c,
        draw_det_boxes=draw_det_boxes,
        overlay_mask_red=overlay_mask_red,
    )
    return _BRIDGE_CACHE
