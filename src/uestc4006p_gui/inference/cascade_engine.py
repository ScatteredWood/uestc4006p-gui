from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable
from queue import Queue
from threading import Thread

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
        self._bridge = None
        self._model_cache: dict[str, object] = {}
        self.runtime_device: str = "cpu"
        self.runtime_device_note: str = "未进行设备探测。"
        self._device_initialized = False

    def _ensure_bridge(self):
        if self._bridge is None:
            self._bridge = get_bridge_objects()
        return self._bridge

    @property
    def bridge(self):
        return self._ensure_bridge()

    @staticmethod
    def _run_with_timeout(fn, timeout_seconds: float, timeout_message: str) -> tuple[bool, str]:
        q: Queue[tuple[bool, str]] = Queue(maxsize=1)

        def _task():
            try:
                fn()
                q.put((True, ""))
            except Exception as exc:  # pragma: no cover
                q.put((False, str(exc)))

        th = Thread(target=_task, daemon=True)
        th.start()
        th.join(timeout=max(0.1, float(timeout_seconds)))
        if th.is_alive():
            return False, timeout_message
        return q.get_nowait() if not q.empty() else (False, "未知错误：自检线程未返回结果。")

    @staticmethod
    def _format_device(device: str) -> str:
        return "CUDA:0" if str(device).lower().startswith("cuda") else "CPU"

    def current_device_display(self) -> str:
        return self._format_device(self.runtime_device)

    def ensure_runtime_device(self, timeout_seconds: float = 8.0, force_refresh: bool = False) -> tuple[str, str]:
        if self._device_initialized and not force_refresh:
            return self.runtime_device, self.runtime_device_note

        try:
            import torch
        except Exception as exc:
            self.runtime_device = "cpu"
            self.runtime_device_note = f"未安装/无法导入 torch，自动回退 CPU: {exc}"
            self._device_initialized = True
            return self.runtime_device, self.runtime_device_note

        cuda_status: dict[str, object] = {"ok": False, "error": ""}

        def _probe_cuda():
            try:
                if not torch.cuda.is_available():
                    cuda_status["ok"] = False
                    cuda_status["error"] = "torch.cuda.is_available() 为 False。"
                    return
                _ = torch.zeros((1,), device="cuda:0")
                torch.cuda.synchronize()
                cuda_status["ok"] = True
            except Exception as exc:  # pragma: no cover
                cuda_status["ok"] = False
                cuda_status["error"] = str(exc)

        ok, msg = self._run_with_timeout(
            _probe_cuda,
            timeout_seconds=timeout_seconds,
            timeout_message="CUDA 探测超时，自动回退 CPU。",
        )
        if not ok:
            self.runtime_device = "cpu"
            self.runtime_device_note = msg
            self._device_initialized = True
            return self.runtime_device, self.runtime_device_note

        if bool(cuda_status.get("ok")):
            self.runtime_device = "cuda:0"
            self.runtime_device_note = "检测到可用 NVIDIA GPU，已优先使用 CUDA:0。"
        else:
            self.runtime_device = "cpu"
            fallback_reason = str(cuda_status.get("error") or "未检测到可用 CUDA。")
            self.runtime_device_note = f"自动回退 CPU：{fallback_reason}"
        self._device_initialized = True
        return self.runtime_device, self.runtime_device_note

    def check_dependencies(self, timeout_seconds: float = 10.0) -> tuple[bool, str]:
        ok, msg = self._run_with_timeout(
            self._ensure_bridge,
            timeout_seconds=timeout_seconds,
            timeout_message="推理依赖初始化超时，请检查 ultralytics/torch 依赖。",
        )
        if not ok:
            return False, msg
        device, note = self.ensure_runtime_device(timeout_seconds=min(5.0, timeout_seconds))
        return True, f"当前设备: {self._format_device(device)} | {note}"

    def startup_self_check(self, timeout_seconds: float = 12.0) -> tuple[bool, str]:
        def _check():
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            import yaml  # noqa: F401

            self._ensure_bridge()
            self.ensure_runtime_device(timeout_seconds=min(6.0, timeout_seconds), force_refresh=True)

        ok, msg = self._run_with_timeout(
            _check,
            timeout_seconds=timeout_seconds,
            timeout_message="启动自检超时，已跳过自检并允许继续使用 GUI。",
        )
        if not ok:
            return False, msg
        return True, f"当前设备: {self._format_device(self.runtime_device)} | {self.runtime_device_note}"

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
        self.ensure_runtime_device()
        model = self.bridge.YOLO(p)
        try:
            model.to(self.runtime_device)
        except Exception as exc:
            if str(self.runtime_device).startswith("cuda"):
                self.runtime_device = "cpu"
                self.runtime_device_note = f"{role}模型加载到 CUDA 失败，自动回退 CPU: {exc}"
                self._device_initialized = True
                model.to("cpu")
            else:
                raise
        self._model_cache[p] = model
        return model

    def _run_detection_only(self, image_bgr: np.ndarray, det_model: object, det_conf: float):
        t0 = perf_counter()
        det_res = det_model.predict(
            image_bgr,
            conf=float(det_conf),
            iou=0.5,
            verbose=False,
            device=self.runtime_device,
        )[0]
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
            device=self.runtime_device,
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

    def _move_cached_models_to_cpu(self) -> None:
        for model in self._model_cache.values():
            try:
                model.to("cpu")
            except Exception:
                pass

    def _infer_one_frame_guarded(
        self,
        frame_bgr: np.ndarray,
        request: RunRequest,
        frame_index: int,
        source_name: str,
        log: LogFn | None = None,
    ) -> FrameResult:
        try:
            return self._infer_one_frame(
                frame_bgr=frame_bgr,
                request=request,
                frame_index=frame_index,
                source_name=source_name,
                log=log,
            )
        except Exception as exc:
            if str(self.runtime_device).startswith("cuda"):
                self.runtime_device = "cpu"
                self.runtime_device_note = f"CUDA 推理失败后自动回退 CPU: {exc}"
                self._device_initialized = True
                self._move_cached_models_to_cpu()
                if log:
                    log(self.runtime_device_note)
                return self._infer_one_frame(
                    frame_bgr=frame_bgr,
                    request=request,
                    frame_index=frame_index,
                    source_name=source_name,
                    log=log,
                )
            raise

    def run_image(self, request: RunRequest, log: LogFn | None = None) -> RunSummary:
        t0 = perf_counter()
        self.ensure_runtime_device()
        if not request.input_path.exists():
            raise FileNotFoundError(f"输入图片不存在: {request.input_path}")

        image_bgr = cv2.imread(str(request.input_path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片: {request.input_path}")

        frame_result = self._infer_one_frame_guarded(
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
            current_device=self._format_device(self.runtime_device),
            device_note=self.runtime_device_note,
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
        self.ensure_runtime_device()
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
                    last_result = self._infer_one_frame_guarded(
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
            current_device=self._format_device(self.runtime_device),
            device_note=self.runtime_device_note,
        )
