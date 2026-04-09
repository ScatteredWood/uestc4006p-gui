from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from ..core.paths import CACHE_DIR
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

    @staticmethod
    def _safe_token(text: str) -> str:
        s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))
        s = s.strip("._")
        return s or "video"

    def _create_video_cache_dir(self, source_path: Path) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = self._safe_token(source_path.stem)
        run_dir = CACHE_DIR / f"video_cache_{stem}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _extract_det_stats(det_res) -> tuple[int, float, float]:
        if det_res is None or getattr(det_res, "boxes", None) is None:
            return 0, 0.0, 0.0
        if len(det_res.boxes) == 0:
            return 0, 0.0, 0.0
        conf = det_res.boxes.conf.detach().cpu().numpy()
        if conf.size == 0:
            return 0, 0.0, 0.0
        return int(conf.size), float(conf.max()), float(conf.mean())

    def _get_model(self, weight_path: Path, role: str) -> object:
        p = str(weight_path.resolve())
        if p in self._model_cache:
            return self._model_cache[p]
        if not weight_path.exists():
            raise FileNotFoundError(f"{role}模型不存在: {weight_path}")
        model = self.bridge.YOLO(p)
        self._model_cache[p] = model
        return model

    def _run_detection_only(self, image_bgr: np.ndarray, det_model: object, det_conf: float):
        t0 = perf_counter()
        det_res = det_model.predict(image_bgr, conf=float(det_conf), iou=0.5, verbose=False)[0]
        model_infer_ms = (perf_counter() - t0) * 1000.0

        t1 = perf_counter()
        det_names = det_model.names if hasattr(det_model, "names") else {}
        overlay = self.bridge.draw_det_boxes(image_bgr, det_res, det_names, conf_thr=float(det_conf))
        viz_ms = (perf_counter() - t1) * 1000.0
        return overlay, None, det_res, model_infer_ms, 0.0, viz_ms

    def _run_cascade(self, image_bgr: np.ndarray, det_model: object, seg_model: object, request: RunRequest):
        p = request.params
        t0 = perf_counter()
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
            # 这里关闭脚本内后处理，改为 GUI engine 计时后再处理。
            post_open_ksize=0,
            post_close_ksize=0,
            post_min_area=0,
            roi_v3=False,
        )
        model_infer_ms = (perf_counter() - t0) * 1000.0

        t1 = perf_counter()
        if mask_u8 is not None and (int(p.post_open) or int(p.post_close) or int(p.post_min_area)):
            mask_u8 = self.bridge.postprocess_mask(
                mask_u8,
                open_ksize=int(p.post_open),
                close_ksize=int(p.post_close),
                min_area=int(p.post_min_area),
            )
        postprocess_ms = (perf_counter() - t1) * 1000.0

        t2 = perf_counter()
        det_names = det_model.names if hasattr(det_model, "names") else {}
        det_overlay = self.bridge.draw_det_boxes(image_bgr, det_res, det_names, conf_thr=float(p.det_conf))
        overlay = self.bridge.overlay_mask_red(det_overlay, mask_u8, alpha=0.45)
        viz_ms = (perf_counter() - t2) * 1000.0
        return overlay, mask_u8, det_res, model_infer_ms, postprocess_ms, viz_ms

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
                log("检测开关关闭：当前帧直接显示原图，不执行推理。")
            return FrameResult(
                mode=request.mode,
                frame_index=frame_index,
                source_name=source_name,
                original_bgr=frame_bgr,
                overlay_bgr=frame_bgr.copy(),
                mask_u8=None,
                segmentation_enabled=False,
            )

        det_model = self._get_model(request.det_model_path, "检测")

        if request.enable_segmentation:
            seg_model = self._get_model(request.seg_model_path, "分割")
            (
                overlay_bgr,
                mask_u8,
                det_res,
                model_infer_ms,
                postprocess_ms,
                viz_ms,
            ) = self._run_cascade(frame_bgr, det_model, seg_model, request)
        else:
            (
                overlay_bgr,
                mask_u8,
                det_res,
                model_infer_ms,
                postprocess_ms,
                viz_ms,
            ) = self._run_detection_only(frame_bgr, det_model, request.params.det_conf)

        det_count, det_conf_max, det_conf_mean = self._extract_det_stats(det_res)
        total_ms = model_infer_ms + postprocess_ms + viz_ms
        est_fps = 1000.0 / total_ms if total_ms > 1e-6 else 0.0

        return FrameResult(
            mode=request.mode,
            frame_index=frame_index,
            source_name=source_name,
            original_bgr=frame_bgr,
            overlay_bgr=overlay_bgr,
            mask_u8=mask_u8,
            det_count=det_count,
            det_conf_max=det_conf_max,
            det_conf_mean=det_conf_mean,
            segmentation_enabled=request.enable_segmentation,
            model_infer_ms=model_infer_ms,
            postprocess_ms=postprocess_ms,
            viz_ms=viz_ms,
            est_fps=est_fps,
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
        if log:
            total_ms = frame_result.model_infer_ms + frame_result.postprocess_ms + frame_result.viz_ms
            log(
                "[PERF][image] "
                f"infer={frame_result.model_infer_ms:.1f}ms, "
                f"post={frame_result.postprocess_ms:.1f}ms, "
                f"viz={frame_result.viz_ms:.1f}ms, "
                f"total={total_ms:.1f}ms, fps={frame_result.est_fps:.2f}"
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
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        out_fps = src_fps if src_fps > 0 else 25.0

        max_frames = int(request.params.max_frames)
        frame_step = max(1, int(request.params.frame_step))
        preview_interval = max(1, int(request.params.preview_interval))

        total_frames = (
            min(total_frames_raw, max_frames)
            if max_frames > 0 and total_frames_raw > 0
            else (max_frames if max_frames > 0 else total_frames_raw)
        )

        read_count = 0
        processed_count = 0
        stopped = False
        last_result: FrameResult | None = None
        last_emitted_processed_idx = 0

        perf_model_sum = 0.0
        perf_post_sum = 0.0
        perf_viz_sum = 0.0
        perf_log_interval = 10

        overlay_writer = None
        mask_writer = None
        cache_overlay_path = ""
        cache_mask_path = ""
        cache_video_playable = False

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

                # 视频模式默认都生成 cache 成品视频，便于 GUI 直接播放。
                if overlay_writer is None:
                    cache_video_dir = self._create_video_cache_dir(request.input_path)
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    overlay_path = cache_video_dir / "overlay_cache.mp4"
                    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, out_fps, (w, h))
                    if overlay_writer.isOpened():
                        cache_overlay_path = str(overlay_path)
                    else:
                        overlay_writer.release()
                        overlay_writer = None
                        if log:
                            log("[WARN] overlay 缓存视频创建失败。")

                    if request.enable_segmentation:
                        mask_path = cache_video_dir / "mask_cache.mp4"
                        mask_writer = cv2.VideoWriter(str(mask_path), fourcc, out_fps, (w, h))
                        if mask_writer.isOpened():
                            cache_mask_path = str(mask_path)
                        else:
                            mask_writer.release()
                            mask_writer = None
                            if log:
                                log("[WARN] mask 缓存视频创建失败。")

                do_infer = (read_count - 1) % frame_step == 0
                overlay_for_cache = frame_bgr
                mask_for_cache = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

                if do_infer:
                    processed_count += 1
                    last_result = self._infer_one_frame(
                        frame_bgr=frame_bgr,
                        request=request,
                        frame_index=read_count,
                        source_name=request.input_path.name,
                        log=log,
                    )
                    perf_model_sum += last_result.model_infer_ms
                    perf_post_sum += last_result.postprocess_ms
                    perf_viz_sum += last_result.viz_ms

                    overlay_for_cache = last_result.overlay_bgr
                    if last_result.mask_u8 is not None:
                        mask_for_cache = last_result.mask_u8

                    if on_frame and (processed_count == 1 or processed_count % preview_interval == 0):
                        on_frame(last_result)
                        last_emitted_processed_idx = processed_count

                    if log and processed_count % perf_log_interval == 0:
                        avg_model = perf_model_sum / processed_count
                        avg_post = perf_post_sum / processed_count
                        avg_viz = perf_viz_sum / processed_count
                        avg_total = avg_model + avg_post + avg_viz
                        avg_fps = 1000.0 / avg_total if avg_total > 1e-6 else 0.0
                        log(
                            "[PERF][video] "
                            f"processed={processed_count}, "
                            f"avg_infer={avg_model:.1f}ms, "
                            f"avg_post={avg_post:.1f}ms, "
                            f"avg_viz={avg_viz:.1f}ms, "
                            f"avg_total={avg_total:.1f}ms, "
                            f"avg_fps={avg_fps:.2f}"
                        )

                if overlay_writer is not None:
                    overlay_writer.write(overlay_for_cache)
                if mask_writer is not None:
                    mask_writer.write(cv2.cvtColor(mask_for_cache, cv2.COLOR_GRAY2BGR))

        finally:
            cap.release()
            if overlay_writer is not None:
                overlay_writer.release()
            if mask_writer is not None:
                mask_writer.release()

        if on_frame and last_result is not None and last_emitted_processed_idx != processed_count:
            on_frame(last_result)

        if cache_overlay_path:
            cache_video_playable = Path(cache_overlay_path).exists()

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
            cache_overlay_video=cache_overlay_path,
            cache_mask_video=cache_mask_path,
            cache_video_playable=cache_video_playable,
        )