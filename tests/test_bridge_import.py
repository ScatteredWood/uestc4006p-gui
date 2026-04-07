from __future__ import annotations

import pytest


def test_bridge_import_objects():
    """
    基础导入测试：
    1) bridge 能否完成路径注入
    2) 关键对象是否可导入
    """
    try:
        from uestc4006p_gui.inference.ultra_bridge import get_bridge_objects

        bridge = get_bridge_objects()
    except Exception as exc:  # pragma: no cover - 用于给出清晰报错
        pytest.fail(
            "bridge 导入失败。请确认 VS Code 解释器为 fyp_gui，"
            "并确认 E:\\repositories\\ultralytics 路径可访问。"
            f"\n原始错误: {exc}"
        )

    assert hasattr(bridge, "YOLO"), "bridge 缺少 YOLO"
    assert callable(bridge.cascade_one_image_v3c), "bridge 缺少 cascade_one_image_v3c"
    assert callable(bridge.draw_det_boxes), "bridge 缺少 draw_det_boxes"
    assert callable(bridge.overlay_mask_red), "bridge 缺少 overlay_mask_red"
