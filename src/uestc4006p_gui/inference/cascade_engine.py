from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from ..core.schemas import FrameResult, RunRequest, RunSummary
from .ultra_bridge import get_bridge_objects

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]
FrameFn = Callable[[FrameResult], None]
StopFn = Callable[[], bool]


class CascadeEngine:
    """
    推理封装层：
    1. 负责模型缓存
    2. 负责参数映射到现有脚本函数
    3. 不处理 GUI 控件
    """

    def __init__(self) -> None:
        self.bridge = get_bridge_objects()
        self._model_cache: dict[str, object] = {}

    def _get_model(self, weight_path: Path, role: str) -> object:
        p = str(weight_path.resolve())
        if p in self._model_cache:
            return self._model_cache[p]
        if not weight_path.exists():
            raise FileNotFoundError(f"{role} 模型不存在: {weight_path}")
        model = self.bridge.YOLO(p)
        self._model_cache[p] = model
        return model

    def _run_detection_only(self, image_bgr: np.ndarray, det_model: object, det_conf: float):
        det_res = det_model.predict(image_bgr, conf=float(det_conf), iou=0.5, verbose=False)[0]
        det_names = det_model.names if hasattr(det_model, "names") else {}
        overlay = self.bridge.draw_det_boxes(image_bgr, det_res, det_names, conf_thr=float(det_conf))
        return overlay, None

    def _run_cascade(self, image_bgr: np.ndarray, det_model: object, seg_model: object, request: RunRequest):
        p = request.params
        mask_u8, det_res = self.bridge.cascade_one_image_v3c(
            image_bgr,
            det_model,
            seg_model,
            det_conf=float(p.det_conf),
            det_iou=0.50,
            det_imgsz=0,
            seg_conf=float(p.seg_conf),
            seg_thr=float(p.seg_thr),
            seg_iou=0.50,
            seg_imgsz=1280,
            pad_ratio=0.15,
            pad_min=16,
            max_rois=80,
            allowed_det_classes=None,
            max_area_ratio=0.60,
            tile_min_side=1400,
            tile=1280,
            overlap=256,
            use_tile_for_big_roi=True,
            big_damage_class_ids=(2, 3),
            big_seg_conf=0.08,
            big_seg_thr=0.25,
            big_seg_iou=0.50,
            big_seg_imgsz=1280,
            big_force_tile=True,
            big_tile=1280,
            big_overlap=384,
            big_use_clahe=False,
            clahe_clip=2.0,
            clahe_grid=8,
            debug_dir=None,
            debug_prefix="",
            post_open_ksize=int(p.post_open),
            post_close_ksize=int(p.post_close),
            post_min_area=int(p.post_min_area),
            roi_v3=False,
        )
        det_names = det_model.names if hasattr(det_model, "names") else {}
        det_overlay = self.bridge.draw_det_boxes(image_bgr, det_res, det_names, conf_thr=float(p.det_conf))
        overlay = self.bridge.overlay_mask_red(det_overlay, mask_u8, alpha=0.45)
        return overlay, mask_u8

    def _infer_one_frame(
        self,
        frame_bgr: np.ndarray,
        request: RunRequest,
        frame_index: int,
        source_name: str,
        log: LogFn | None = None,
    ) -> FrameResult:
        if not request.enable_detection:
            if log:
                log("检测开关关闭：当前帧直接展示原图，不执行推理。")
            return FrameResult(
                mode=request.mode,
                frame_index=frame_index,
                source_name=source_name,
                original_bgr=frame_bgr,
                overlay_bgr=frame_bgr.copy(),
                mask_u8=None,
            )

        det_model = self._get_model(request.det_model_path, "检测")

        if request.enable_segmentation:
            seg_model = self._get_model(request.seg_model_path, "分割")
            overlay_bgr, mask_u8 = self._run_cascade(frame_bgr, det_model, seg_model, request)
        else:
            overlay_bgr, mask_u8 = self._run_detection_only(frame_bgr, det_model, request.params.det_conf)

        return FrameResult(
            mode=request.mode,
            frame_index=frame_index,
            source_name=source_name,
            original_bgr=frame_bgr,
            overlay_bgr=overlay_bgr,
            mask_u8=mask_u8,
        )

    def run_image(self, request: RunRequest, log: LogFn | None = None) -> RunSummary:
        t0 = perf_counter()
        if not request.input_path.exists():
            raise FileNotFoundError(f"输入图片不存在: {request.input_path}")

        image_bgr = cv2.imread(str(request.input_path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片: {request.input_path}")

        frame_result = self._infer_one_frame(
            frame_bgr=image_bgr,
            request=request,
            frame_index=1,
            source_name=request.input_path.name,
            log=log,
        )
        elapsed = perf_counter() - t0
        return RunSummary(
            mode="image",
            source_path=str(request.input_path),
            total_frames=1,
            processed_frames=1,
            elapsed_seconds=elapsed,
            stopped=False,
            message="图片推理完成",
            last_frame_result=frame_result,
        )

    def run_video(
        self,
        request: RunRequest,
        on_progress: ProgressFn | None = None,
        on_frame: FrameFn | None = None,
        should_stop: StopFn | None = None,
        log: LogFn | None = None,
    ) -> RunSummary:
        t0 = perf_counter()
        if not request.input_path.exists():
            raise FileNotFoundError(f"输入视频不存在: {request.input_path}")

        cap = cv2.VideoCapture(str(request.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {request.input_path}")

        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(request.params.max_frames)
        frame_step = max(1, int(request.params.frame_step))
        total_frames = (
            min(total_frames_raw, max_frames)
            if max_frames > 0 and total_frames_raw > 0
            else (max_frames if max_frames > 0 else total_frames_raw)
        )

        read_count = 0
        processed_count = 0
        last_result: FrameResult | None = None
        stopped = False

        try:
            while True:
                if should_stop and should_stop():
                    stopped = True
                    if log:
                        log("收到停止信号，正在安全结束视频推理。")
                    break

                if max_frames > 0 and read_count >= max_frames:
                    break

                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                read_count += 1
                if on_progress:
                    on_progress(read_count, total_frames)

                # 按 frame_step 抽帧处理，减少演示等待时间。
                if (read_count - 1) % frame_step != 0:
                    continue

                processed_count += 1
                last_result = self._infer_one_frame(
                    frame_bgr=frame_bgr,
                    request=request,
                    frame_index=read_count,
                    source_name=request.input_path.name,
                    log=log,
                )
                if on_frame:
                    on_frame(last_result)
        finally:
            cap.release()

        elapsed = perf_counter() - t0
        msg = "视频推理已停止" if stopped else "视频推理完成"
        return RunSummary(
            mode="video",
            source_path=str(request.input_path),
            total_frames=max(read_count, total_frames),
            processed_frames=processed_count,
            elapsed_seconds=elapsed,
            stopped=stopped,
            message=msg,
            last_frame_result=last_result,
        )
