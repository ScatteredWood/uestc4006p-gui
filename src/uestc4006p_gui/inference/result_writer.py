from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from ..core.paths import CACHE_DIR, OUTPUTS_DIR
from ..core.schemas import FrameResult, RunRequest, RunSummary


class ResultWriter:
    """缓存写入与手动保存逻辑。"""

    def __init__(self, cache_root: Path = CACHE_DIR, outputs_root: Path = OUTPUTS_DIR) -> None:
        self.cache_root = cache_root
        self.outputs_root = outputs_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_run_dir(self, root: Path, mode: str) -> Path:
        run_dir = root / f"{mode}_{self._timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _write_frame_bundle(run_dir: Path, frame: FrameResult) -> dict[str, str]:
        out: dict[str, str] = {}

        original_path = run_dir / "original.jpg"
        overlay_path = run_dir / "overlay.jpg"
        cv2.imwrite(str(original_path), frame.original_bgr)
        cv2.imwrite(str(overlay_path), frame.overlay_bgr)
        out["original"] = str(original_path)
        out["overlay"] = str(overlay_path)

        if frame.mask_u8 is not None:
            mask_path = run_dir / "mask.png"
            cv2.imwrite(str(mask_path), frame.mask_u8)
            out["mask"] = str(mask_path)

        return out

    @staticmethod
    def _copy_if_exists(src: str, dst: Path) -> str:
        if src and Path(src).exists():
            shutil.copy2(src, dst)
            return str(dst)
        return ""

    @staticmethod
    def _write_meta(run_dir: Path, request: RunRequest, summary: RunSummary, extra: dict | None = None) -> Path:
        meta = {
            "mode": request.mode,
            "input_path": str(request.input_path),
            "det_model_path": str(request.det_model_path),
            "seg_model_path": str(request.seg_model_path),
            "enable_detection": request.enable_detection,
            "enable_segmentation": request.enable_segmentation,
            "cache_only": request.cache_only,
            "params": {
                "det_conf": request.params.det_conf,
                "seg_conf": request.params.seg_conf,
                "seg_thr": request.params.seg_thr,
                "post_open": request.params.post_open,
                "post_close": request.params.post_close,
                "post_min_area": request.params.post_min_area,
                "frame_step": request.params.frame_step,
                "max_frames": request.params.max_frames,
                "preview_interval": request.params.preview_interval,
            },
            "summary": {
                "total_frames": summary.total_frames,
                "processed_frames": summary.processed_frames,
                "elapsed_seconds": summary.elapsed_seconds,
                "stopped": summary.stopped,
                "message": summary.message,
                "cache_overlay_video": summary.cache_overlay_video,
                "cache_mask_video": summary.cache_mask_video,
                "cache_video_playable": summary.cache_video_playable,
            },
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            meta.update(extra)

        meta_path = run_dir / "result_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def cache_run(self, request: RunRequest, summary: RunSummary) -> Path | None:
        """
        默认缓存逻辑：只缓存当前结果快照和摘要，不写 outputs。
        视频模式的成品缓存视频由 engine 在 cache 下直接生成。
        """
        if summary.last_frame_result is None:
            return None

        run_dir = self._create_run_dir(self.cache_root, request.mode)
        self._write_frame_bundle(run_dir, summary.last_frame_result)
        self._write_meta(run_dir, request, summary)
        return run_dir

    def save_current_result(self, request: RunRequest, summary: RunSummary) -> Path:
        """手动保存当前结果到 outputs。"""
        if summary.last_frame_result is None:
            raise RuntimeError("当前没有可保存结果。请先执行一次推理。")

        output_root = request.output_dir if request.output_dir else self.outputs_root
        output_root.mkdir(parents=True, exist_ok=True)

        run_dir = self._create_run_dir(output_root, request.mode)
        self._write_frame_bundle(run_dir, summary.last_frame_result)

        extra: dict[str, str] = {}
        if request.mode == "video":
            overlay_saved = self._copy_if_exists(
                summary.cache_overlay_video,
                run_dir / "overlay_video.mp4",
            )
            mask_saved = self._copy_if_exists(
                summary.cache_mask_video,
                run_dir / "mask_video.mp4",
            )
            if overlay_saved:
                extra["saved_overlay_video"] = overlay_saved
            if mask_saved:
                extra["saved_mask_video"] = mask_saved
            if not overlay_saved and not mask_saved:
                note_path = run_dir / "video_note.txt"
                note_path.write_text(
                    "视频模式未检测到可复制的缓存视频。\n"
                    f"输入视频: {request.input_path}\n"
                    f"当前帧: {summary.last_frame_result.frame_index}\n",
                    encoding="utf-8",
                )
                extra["video_note"] = str(note_path)

        self._write_meta(run_dir, request, summary, extra=extra)
        return run_dir