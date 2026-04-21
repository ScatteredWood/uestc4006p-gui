from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def _clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    ix1 = max(0, min(int(x1), width - 1))
    iy1 = max(0, min(int(y1), height - 1))
    ix2 = max(0, min(int(x2), width))
    iy2 = max(0, min(int(y2), height))
    if ix2 <= ix1:
        ix2 = min(width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(height, iy1 + 1)
    return ix1, iy1, ix2, iy2


def _expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    pad_ratio: float = 0.15,
    pad_min: int = 16,
) -> tuple[int, int, int, int]:
    bw, bh = float(x2 - x1), float(y2 - y1)
    pad = int(max(float(pad_min), float(pad_ratio) * max(bw, bh)))
    return _clamp_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, width, height)


def _predict_one(model, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int, device: str):
    kwargs = {"conf": float(conf), "iou": float(iou), "verbose": False, "device": device}
    if int(imgsz) > 0:
        kwargs["imgsz"] = int(imgsz)
    return model.predict(image_bgr, **kwargs)[0]


def overlay_mask_red(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    mask = mask_u8 > 0
    overlay[mask] = (img_bgr[mask] * (1 - float(alpha)) + red[mask] * float(alpha)).astype(np.uint8)
    return overlay


def yolo_seg_union_mask(seg_result, roi_h: int, roi_w: int, seg_conf: float = 0.25, thr: float = 0.5) -> np.ndarray:
    if seg_result is None or getattr(seg_result, "masks", None) is None:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    masks = seg_result.masks.data.detach().cpu().numpy()
    if masks.size == 0:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    if getattr(seg_result, "boxes", None) is not None and len(seg_result.boxes) == masks.shape[0]:
        scores = seg_result.boxes.conf.detach().cpu().numpy()
        keep = scores >= float(seg_conf)
        masks = masks[keep] if keep.any() else masks[:0]

    if masks.shape[0] == 0:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    union = np.max(masks, axis=0)
    union = (union > float(thr)).astype(np.uint8) * 255
    return cv2.resize(union, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)


def draw_det_boxes(img_bgr: np.ndarray, det_res, names: dict, conf_thr: float = 0.2) -> np.ndarray:
    out = img_bgr.copy()
    boxes = getattr(det_res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)

    for box, score, cls_id in zip(xyxy, conf, cls):
        if float(score) < float(conf_thr):
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{names.get(cls_id, str(cls_id))} {float(score):.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return out


def postprocess_mask(mask_u8: np.ndarray, open_ksize: int = 0, close_ksize: int = 0, min_area: int = 0) -> np.ndarray:
    out = mask_u8.copy()

    if int(open_ksize) >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_ksize), int(open_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)

    if int(close_ksize) >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)

    if int(min_area) > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(out, dtype=np.uint8)
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= int(min_area):
                keep[labels == idx] = 255
        out = keep

    return out


def _is_allowed_class(class_id: int, allowed_set: set[int] | None) -> bool:
    if allowed_set is None:
        return True
    return int(class_id) in allowed_set


def cascade_one_image_v3c(
    image_bgr: np.ndarray,
    det_model,
    seg_model,
    *,
    det_conf: float = 0.15,
    det_iou: float = 0.5,
    det_imgsz: int = 0,
    seg_conf: float = 0.1,
    seg_thr: float = 0.3,
    seg_iou: float = 0.5,
    seg_imgsz: int = 1280,
    pad_ratio: float = 0.15,
    pad_min: int = 16,
    max_rois: int = 80,
    allowed_det_classes: Iterable[int] | None = None,
    max_area_ratio: float = 0.60,
    device: str = "cpu",
    **_kwargs,
):
    """轻量级 Det->ROI->Seg 级联实现，接口与历史脚本保持兼容。"""
    det_res = _predict_one(det_model, image_bgr, det_conf, det_iou, int(det_imgsz), device=device)
    height, width = image_bgr.shape[:2]
    full_mask = np.zeros((height, width), dtype=np.uint8)

    boxes = getattr(det_res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return full_mask, det_res

    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
    conf_arr = boxes.conf.detach().cpu().numpy()

    roi_count = 0
    image_area = max(1, width * height)
    allowed_set = {int(item) for item in allowed_det_classes} if allowed_det_classes is not None else None
    for box, cls_id, score in zip(xyxy, cls_arr, conf_arr):
        if float(score) < float(det_conf):
            continue
        if not _is_allowed_class(int(cls_id), allowed_set):
            continue
        if roi_count >= int(max_rois):
            break

        x1, y1, x2, y2 = _expand_box(
            box[0],
            box[1],
            box[2],
            box[3],
            width,
            height,
            pad_ratio=float(pad_ratio),
            pad_min=int(pad_min),
        )
        area_ratio = float((x2 - x1) * (y2 - y1)) / float(image_area)
        if float(max_area_ratio) > 0 and area_ratio > float(max_area_ratio):
            continue

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        seg_res = _predict_one(seg_model, roi, seg_conf, seg_iou, int(seg_imgsz), device=device)
        roi_mask = yolo_seg_union_mask(seg_res, roi.shape[0], roi.shape[1], seg_conf=seg_conf, thr=seg_thr)

        target = full_mask[y1:y2, x1:x2]
        np.maximum(target, roi_mask, out=target)
        roi_count += 1

    return full_mask, det_res
