from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

RunMode = Literal["image", "video"]


@dataclass
class InferenceParams:
    det_conf: float = 0.15
    seg_conf: float = 0.10
    seg_thr: float = 0.30
    post_open: int = 0
    post_close: int = 3
    post_min_area: int = 25
    frame_step: int = 1
    max_frames: int = 0
    preview_interval: int = 5


@dataclass
class RunRequest:
    mode: RunMode
    input_path: Path
    output_dir: Path
    det_model_path: Path
    seg_model_path: Path
    enable_detection: bool = True
    enable_segmentation: bool = True
    cache_only: bool = True
    params: InferenceParams = field(default_factory=InferenceParams)


@dataclass
class FrameResult:
    mode: RunMode
    frame_index: int
    source_name: str
    original_bgr: np.ndarray
    overlay_bgr: np.ndarray
    mask_u8: np.ndarray | None = None
    det_count: int = 0
    det_conf_max: float = 0.0
    det_conf_mean: float = 0.0
    segmentation_enabled: bool = False
    model_infer_ms: float = 0.0
    postprocess_ms: float = 0.0
    viz_ms: float = 0.0
    est_fps: float = 0.0


@dataclass
class RunSummary:
    mode: RunMode
    source_path: str
    total_frames: int
    processed_frames: int
    elapsed_seconds: float
    stopped: bool
    message: str = ""
    last_frame_result: FrameResult | None = None
    cache_overlay_video: str = ""
    cache_mask_video: str = ""
    cache_video_playable: bool = False
    current_device: str = ""
    device_note: str = ""
