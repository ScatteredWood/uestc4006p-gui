from __future__ import annotations

import pytest


def test_bridge_import_objects() -> None:
    """
    基础导入测试：
    1) bridge 能否在不依赖外部仓库路径的情况下导入
    2) 关键对象是否可导入
    """
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")

    try:
        from uestc4006p_gui.inference.ultra_bridge import get_bridge_objects

        bridge = get_bridge_objects()
    except Exception as exc:  # pragma: no cover
        pytest.fail(
            "bridge 导入失败。请确认已安装 ultralytics/torch，"
            "且打包产物包含内置推理模块。"
            f"\n原始错误: {exc}"
        )

    assert hasattr(bridge, "YOLO"), "bridge 缺少 YOLO"
    assert callable(bridge.cascade_one_image_v3c), "bridge 缺少 cascade_one_image_v3c"
    assert callable(bridge.draw_det_boxes), "bridge 缺少 draw_det_boxes"
    assert callable(bridge.overlay_mask_red), "bridge 缺少 overlay_mask_red"
    assert callable(bridge.postprocess_mask), "bridge 缺少 postprocess_mask"
