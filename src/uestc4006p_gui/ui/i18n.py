from __future__ import annotations

from typing import Any

LANG_ZH = "zh"
LANG_EN = "en"
SUPPORTED_LANGUAGES = {LANG_ZH, LANG_EN}

TEXTS: dict[str, dict[str, str]] = {
    "window.title": {
        LANG_ZH: "UESTC4006P GUI MVP",
        LANG_EN: "UESTC4006P GUI MVP",
    },
    "status.ready": {LANG_ZH: "就绪", LANG_EN: "Ready"},
    "status.mode_video": {
        LANG_ZH: "视频模式：可按帧预览，也可播放缓存视频。",
        LANG_EN: "Video mode: frame preview or cached video playback.",
    },
    "status.mode_image": {
        LANG_ZH: "图片模式：单图推理。",
        LANG_EN: "Image mode: single-image inference.",
    },
    "status.saved_current_result": {
        LANG_ZH: "已保存当前结果",
        LANG_EN: "Current result saved",
    },
    "group.input_model": {LANG_ZH: "输入与模型", LANG_EN: "Input && Models"},
    "group.switch": {LANG_ZH: "开关", LANG_EN: "Switches"},
    "group.params": {LANG_ZH: "参数", LANG_EN: "Parameters"},
    "group.actions": {LANG_ZH: "操作", LANG_EN: "Actions"},
    "group.result_summary": {LANG_ZH: "当前结果摘要", LANG_EN: "Current Result Summary"},
    "group.log": {LANG_ZH: "过程日志", LANG_EN: "Process Log"},
    "label.language": {LANG_ZH: "语言", LANG_EN: "Language"},
    "label.input_mode": {LANG_ZH: "输入模式", LANG_EN: "Input Mode"},
    "label.det_model_path": {LANG_ZH: "检测模型路径", LANG_EN: "Detection Model Path"},
    "label.seg_model_path": {LANG_ZH: "分割模型路径", LANG_EN: "Segmentation Model Path"},
    "label.input_file": {LANG_ZH: "输入文件", LANG_EN: "Input File"},
    "label.output_dir": {LANG_ZH: "输出目录", LANG_EN: "Output Directory"},
    "label.det_conf": {LANG_ZH: "检测置信度 (det_conf)", LANG_EN: "Detection Confidence (det_conf)"},
    "label.seg_conf": {LANG_ZH: "分割置信度 (seg_conf)", LANG_EN: "Segmentation Confidence (seg_conf)"},
    "label.seg_thr": {LANG_ZH: "分割二值阈值 (seg_thr)", LANG_EN: "Segmentation Threshold (seg_thr)"},
    "label.post_open": {LANG_ZH: "开运算核大小 (post_open)", LANG_EN: "Opening Kernel Size (post_open)"},
    "label.post_close": {LANG_ZH: "闭运算核大小 (post_close)", LANG_EN: "Closing Kernel Size (post_close)"},
    "label.post_min_area": {
        LANG_ZH: "最小连通域面积 (post_min_area)",
        LANG_EN: "Min Connected Area (post_min_area)",
    },
    "label.frame_step": {LANG_ZH: "视频抽帧步长 (frame_step)", LANG_EN: "Video Frame Step (frame_step)"},
    "label.max_frames": {LANG_ZH: "最大处理帧数 (max_frames)", LANG_EN: "Max Processed Frames (max_frames)"},
    "label.preview_interval": {
        LANG_ZH: "预览刷新间隔 (preview_interval)",
        LANG_EN: "Preview Refresh Interval (preview_interval)",
    },
    "label.preview_mode": {LANG_ZH: "预览模式", LANG_EN: "Preview Mode"},
    "combo.language.zh": {LANG_ZH: "中文", LANG_EN: "Chinese"},
    "combo.language.en": {LANG_ZH: "English", LANG_EN: "English"},
    "combo.mode.image": {LANG_ZH: "图片", LANG_EN: "Image"},
    "combo.mode.video": {LANG_ZH: "视频", LANG_EN: "Video"},
    "combo.preview.original": {LANG_ZH: "原图", LANG_EN: "Original"},
    "combo.preview.overlay": {LANG_ZH: "Overlay", LANG_EN: "Overlay"},
    "combo.preview.mask": {LANG_ZH: "Mask", LANG_EN: "Mask"},
    "combo.video_view.frame": {LANG_ZH: "单帧预览", LANG_EN: "Frame Preview"},
    "combo.video_view.video": {LANG_ZH: "缓存视频播放", LANG_EN: "Cached Video Playback"},
    "combo.video_source.overlay": {LANG_ZH: "Overlay 视频", LANG_EN: "Overlay Video"},
    "combo.video_source.mask": {LANG_ZH: "Mask 视频", LANG_EN: "Mask Video"},
    "combo.video_source.mask_unavailable": {LANG_ZH: "Mask 视频 (不可用)", LANG_EN: "Mask Video (Unavailable)"},
    "checkbox.enable_det": {LANG_ZH: "启用检测", LANG_EN: "Enable Detection"},
    "checkbox.enable_seg": {LANG_ZH: "启用分割", LANG_EN: "Enable Segmentation"},
    "checkbox.cache_only": {LANG_ZH: "仅缓存预览", LANG_EN: "Cache Preview Only"},
    "checkbox.show_result": {LANG_ZH: "显示右侧结果摘要", LANG_EN: "Show Right Result Summary"},
    "button.browse": {LANG_ZH: "浏览", LANG_EN: "Browse"},
    "button.start": {LANG_ZH: "开始", LANG_EN: "Start"},
    "button.stop": {LANG_ZH: "停止", LANG_EN: "Stop"},
    "button.manual_save": {LANG_ZH: "手动保存当前结果", LANG_EN: "Manually Save Current Result"},
    "button.open_output": {LANG_ZH: "打开输出目录", LANG_EN: "Open Output Directory"},
    "button.clear_log": {LANG_ZH: "清空日志", LANG_EN: "Clear Log"},
    "button.play": {LANG_ZH: "播放", LANG_EN: "Play"},
    "button.pause": {LANG_ZH: "暂停", LANG_EN: "Pause"},
    "placeholder.select_dir": {LANG_ZH: "选择目录", LANG_EN: "Select directory"},
    "placeholder.select_file": {LANG_ZH: "选择文件", LANG_EN: "Select file"},
    "preview.info.none": {LANG_ZH: "暂无预览", LANG_EN: "No preview"},
    "preview.info.select_start": {LANG_ZH: "请选择输入并点击开始", LANG_EN: "Select input and click Start"},
    "preview.info.qtm_unavailable": {LANG_ZH: "QtMultimedia 不可用", LANG_EN: "QtMultimedia unavailable"},
    "preview.info.qtm_unavailable_video": {
        LANG_ZH: "QtMultimedia 不可用，无法播放缓存视频。",
        LANG_EN: "QtMultimedia unavailable, cached video cannot be played.",
    },
    "preview.info.cache_unavailable": {LANG_ZH: "缓存视频不可用", LANG_EN: "Cached video unavailable"},
    "preview.info.cache_loaded": {LANG_ZH: "已加载缓存视频: {name}", LANG_EN: "Cached video loaded: {name}"},
    "preview.info.running": {LANG_ZH: "推理中...", LANG_EN: "Running inference..."},
    "preview.info.current_frame": {LANG_ZH: "当前帧: {index}", LANG_EN: "Current frame: {index}"},
    "preview.info.image_done": {LANG_ZH: "图片推理完成", LANG_EN: "Image inference completed"},
    "preview.label.select_start": {LANG_ZH: "请选择输入并点击开始", LANG_EN: "Select input and click Start"},
    "preview.label.running": {LANG_ZH: "推理中...", LANG_EN: "Running inference..."},
    "preview.label.no_mask": {LANG_ZH: "当前无 mask", LANG_EN: "No mask for current frame"},
    "preview.label.convert_failed": {LANG_ZH: "预览转换失败", LANG_EN: "Preview conversion failed"},
    "startup.default_models_yaml_missing": {
        LANG_ZH: "未找到默认模型配置 default_models.yaml。"
        "GUI 可正常启动，请在界面中手动选择模型后再运行推理。"
        "\n已检查路径: {candidates}",
        LANG_EN: "default_models.yaml was not found. "
        "The GUI can still start; please select model files manually before running inference."
        "\nChecked paths: {candidates}",
    },
    "startup.default_models_yaml_top_invalid": {
        LANG_ZH: "default_models.yaml 顶层必须是 YAML 映射（key-value）。",
        LANG_EN: "The root of default_models.yaml must be a YAML mapping (key-value).",
    },
    "startup.default_models_read_failed": {
        LANG_ZH: "读取默认模型配置失败: {error}",
        LANG_EN: "Failed to read default model config: {error}",
    },
    "startup.default_model_missing": {
        LANG_ZH: "default_models.yaml 中的默认{label}不存在: {raw}\n不影响 GUI 启动，可在界面中手动选择模型。",
        LANG_EN: "Configured default {label} in default_models.yaml does not exist: {raw}\n"
        "GUI startup is not affected; you can select the model manually.",
    },
    "startup.default_label.det_model": {LANG_ZH: "检测模型", LANG_EN: "detection model"},
    "startup.default_label.seg_model": {LANG_ZH: "分割模型", LANG_EN: "segmentation model"},
    "startup.default_output_fallback": {
        LANG_ZH: "默认输出目录不可用，已回退到 outputs: {error}",
        LANG_EN: "Default output directory is unavailable; fell back to outputs: {error}",
    },
    "startup.dependencies_not_ready": {
        LANG_ZH: "启动自检发现依赖未就绪：\n{detail}\nGUI 可继续启动，但推理可能无法开始。",
        LANG_EN: "Startup self-check found dependency issues:\n{detail}\n"
        "GUI can continue, but inference may not start.",
    },
    "startup.self_check_watchdog_timeout": {
        LANG_ZH: "启动自检看门狗超时，已跳过阻塞项并允许继续使用。",
        LANG_EN: "Startup self-check watchdog timed out. Blocking checks were skipped and GUI keeps running.",
    },
    "dialog.startup_check.title": {LANG_ZH: "启动检查提示", LANG_EN: "Startup Check Notice"},
    "dialog.startup_check.body": {
        LANG_ZH: "检测到以下问题（不会导致 GUI 直接退出）：\n\n{items}\n\n你仍可打开界面并手动选择模型；修复后重新点击“开始”。",
        LANG_EN: "The following issues were detected (the GUI will not exit directly):\n\n{items}\n\n"
        "You can still open the GUI and select model files manually; click \"Start\" again after fixing.",
    },
    "dialog.select_det_model": {LANG_ZH: "选择检测模型", LANG_EN: "Select detection model"},
    "dialog.select_seg_model": {LANG_ZH: "选择分割模型", LANG_EN: "Select segmentation model"},
    "dialog.select_output_dir": {LANG_ZH: "选择输出目录", LANG_EN: "Select output directory"},
    "dialog.select_video": {LANG_ZH: "选择视频", LANG_EN: "Select video"},
    "dialog.select_image": {LANG_ZH: "选择图片", LANG_EN: "Select image"},
    "validation.invalid_mode": {
        LANG_ZH: "任务模式为空或非法，请先选择“图片”或“视频”。",
        LANG_EN: "Task mode is empty or invalid. Please select \"Image\" or \"Video\" first.",
    },
    "validation.input_required": {LANG_ZH: "请先选择输入文件。", LANG_EN: "Please select an input file first."},
    "validation.input_not_exist": {LANG_ZH: "输入路径不存在，请重新选择。", LANG_EN: "Input path does not exist. Please reselect."},
    "validation.input_not_file": {
        LANG_ZH: "当前仅支持输入单个文件，请选择图片或视频文件。",
        LANG_EN: "Only a single file input is supported. Please select an image or video file.",
    },
    "validation.output_required": {LANG_ZH: "请先选择输出目录。", LANG_EN: "Please select an output directory first."},
    "validation.output_not_dir": {LANG_ZH: "输出路径不是目录，请重新选择。", LANG_EN: "Output path is not a directory. Please reselect."},
    "validation.output_unavailable": {LANG_ZH: "输出目录不可用：{error}", LANG_EN: "Output directory is unavailable: {error}"},
    "validation.seg_requires_det": {
        LANG_ZH: "启用分割时必须同时启用检测，请调整任务组合。",
        LANG_EN: "Segmentation requires detection to be enabled at the same time.",
    },
    "validation.need_one_task": {
        LANG_ZH: "请至少启用一个任务（检测或分割）。",
        LANG_EN: "Please enable at least one task (detection or segmentation).",
    },
    "validation.det_model_required": {
        LANG_ZH: "当前任务需要检测模型，请先选择检测模型。",
        LANG_EN: "Current task requires a detection model. Please select it first.",
    },
    "validation.det_model_invalid": {
        LANG_ZH: "检测模型路径不存在或不是文件，请重新选择。",
        LANG_EN: "Detection model path does not exist or is not a file. Please reselect.",
    },
    "validation.seg_model_required": {
        LANG_ZH: "当前任务需要分割模型，请先选择分割模型。",
        LANG_EN: "Current task requires a segmentation model. Please select it first.",
    },
    "validation.seg_model_invalid": {
        LANG_ZH: "分割模型路径不存在或不是文件，请重新选择。",
        LANG_EN: "Segmentation model path does not exist or is not a file. Please reselect.",
    },
    "validation.dependency_not_ready": {
        LANG_ZH: "推理依赖未就绪，当前无法开始推理。\n{detail}\n请修复路径或依赖后重试，GUI 不会退出。",
        LANG_EN: "Inference dependencies are not ready, so inference cannot start now.\n{detail}\n"
        "Please fix paths/dependencies and retry. The GUI will not exit.",
    },
    "dialog.cannot_start_run": {LANG_ZH: "无法开始推理", LANG_EN: "Cannot Start Inference"},
    "dialog.inference_failed": {LANG_ZH: "推理失败", LANG_EN: "Inference Failed"},
    "dialog.info": {LANG_ZH: "提示", LANG_EN: "Info"},
    "dialog.save_failed": {LANG_ZH: "保存失败", LANG_EN: "Save Failed"},
    "dialog.exit_running": {LANG_ZH: "任务仍在运行", LANG_EN: "Task Still Running"},
    "dialog.exit_failed": {LANG_ZH: "退出失败", LANG_EN: "Exit Failed"},
    "dialog.exit_confirm": {LANG_ZH: "退出确认", LANG_EN: "Exit Confirmation"},
    "dialog.cleanup_failed": {LANG_ZH: "清理失败", LANG_EN: "Cleanup Failed"},
    "message.no_result_to_save": {LANG_ZH: "当前没有可保存结果。", LANG_EN: "No current result to save."},
    "message.current_result_saved_log": {LANG_ZH: "[SAVE] 已保存: {path}", LANG_EN: "[SAVE] Saved: {path}"},
    "message.cache_snapshot_saved": {
        LANG_ZH: "[CACHE] 结果快照已缓存: {path}",
        LANG_EN: "[CACHE] Result snapshot cached: {path}",
    },
    "message.worker_started": {LANG_ZH: "后台任务启动", LANG_EN: "Background worker started"},
    "message.current_device_log": {LANG_ZH: "[DEVICE] 当前设备: {device}", LANG_EN: "[DEVICE] Current device: {device}"},
    "message.device_note_log": {LANG_ZH: "[DEVICE] 说明: {note}", LANG_EN: "[DEVICE] Note: {note}"},
    "summary.empty": {LANG_ZH: "暂无结果。运行一次推理后会在这里显示摘要。", LANG_EN: "No results yet. Run inference once to show summary here."},
    "summary.input_file": {LANG_ZH: "输入文件: {name}", LANG_EN: "Input file: {name}"},
    "summary.mode": {LANG_ZH: "模式: {mode}", LANG_EN: "Mode: {mode}"},
    "summary.mode.video": {LANG_ZH: "视频", LANG_EN: "Video"},
    "summary.mode.image": {LANG_ZH: "图片", LANG_EN: "Image"},
    "summary.current_device": {LANG_ZH: "当前设备: {device}", LANG_EN: "Current device: {device}"},
    "summary.device_note": {LANG_ZH: "设备说明: {note}", LANG_EN: "Device note: {note}"},
    "summary.current_frame": {LANG_ZH: "当前帧号: {index}", LANG_EN: "Current frame: {index}"},
    "summary.det_count": {LANG_ZH: "检测框数量: {count}", LANG_EN: "Detections: {count}"},
    "summary.det_conf_max": {LANG_ZH: "最高检测置信度: {value:.3f}", LANG_EN: "Max detection confidence: {value:.3f}"},
    "summary.det_conf_mean": {LANG_ZH: "平均检测置信度: {value:.3f}", LANG_EN: "Mean detection confidence: {value:.3f}"},
    "summary.seg_enabled": {LANG_ZH: "启用分割: {value}", LANG_EN: "Segmentation enabled: {value}"},
    "summary.yes": {LANG_ZH: "是", LANG_EN: "Yes"},
    "summary.no": {LANG_ZH: "否", LANG_EN: "No"},
    "summary.current_proc": {LANG_ZH: "当前处理耗时: {value:.1f} ms", LANG_EN: "Current processing time: {value:.1f} ms"},
    "summary.est_fps": {LANG_ZH: "估算 FPS: {value:.2f}", LANG_EN: "Estimated FPS: {value:.2f}"},
    "summary.preview_mode": {LANG_ZH: "当前预览模式: {mode}", LANG_EN: "Current preview mode: {mode}"},
    "summary.section_perf": {LANG_ZH: "性能分解:", LANG_EN: "Performance Breakdown:"},
    "summary.perf_model": {LANG_ZH: "- 模型推理: {value:.1f} ms", LANG_EN: "- Model inference: {value:.1f} ms"},
    "summary.perf_post": {LANG_ZH: "- 后处理: {value:.1f} ms", LANG_EN: "- Postprocess: {value:.1f} ms"},
    "summary.perf_viz": {LANG_ZH: "- 可视化合成: {value:.1f} ms", LANG_EN: "- Visualization compose: {value:.1f} ms"},
    "summary.perf_ui_convert": {LANG_ZH: "- UI 图像转换: {value:.1f} ms", LANG_EN: "- UI image conversion: {value:.1f} ms"},
    "summary.perf_ui_refresh": {LANG_ZH: "- UI 刷新: {value:.1f} ms", LANG_EN: "- UI refresh: {value:.1f} ms"},
    "summary.perf_write": {LANG_ZH: "- 写盘耗时({action}): {value:.1f} ms", LANG_EN: "- Disk write ({action}): {value:.1f} ms"},
    "summary.section_video_cache": {LANG_ZH: "视频缓存:", LANG_EN: "Video Cache:"},
    "summary.overlay_cache": {LANG_ZH: "- overlay 缓存视频: {value}", LANG_EN: "- Overlay cache video: {value}"},
    "summary.mask_cache": {LANG_ZH: "- mask 缓存视频: {value}", LANG_EN: "- Mask cache video: {value}"},
    "summary.not_generated": {LANG_ZH: "未生成", LANG_EN: "Not generated"},
    "summary.playable": {LANG_ZH: "- 当前可播放状态: {value}", LANG_EN: "- Current playable state: {value}"},
    "summary.playable_yes": {LANG_ZH: "可播放", LANG_EN: "Playable"},
    "summary.playable_no": {LANG_ZH: "不可播放", LANG_EN: "Not playable"},
    "cleanup.remove_failed": {LANG_ZH: "删除 {name} 失败（已重试 {retries} 次）: {error}", LANG_EN: "Failed to remove {name} (retried {retries} times): {error}"},
    "exit.running_prompt": {
        LANG_ZH: "后台推理仍在运行，是否请求停止并继续退出？",
        LANG_EN: "Background inference is still running. Request stop and continue exiting?",
    },
    "exit.thread_not_stopped": {
        LANG_ZH: "推理线程尚未停止，请稍后再试。",
        LANG_EN: "Inference thread has not stopped yet. Please try again later.",
    },
    "exit.confirm_text": {
        LANG_ZH: "退出前如何处理缓存和日志？",
        LANG_EN: "How should cache and logs be handled before exit?",
    },
    "exit.confirm_info": {
        LANG_ZH: "仅会处理 GUI 仓库下的 cache 和 logs，不会删除 outputs。",
        LANG_EN: "Only cache and logs under the GUI repository will be handled; outputs will not be deleted.",
    },
    "exit.btn.delete": {LANG_ZH: "删除 cache 和 logs 后退出", LANG_EN: "Delete cache and logs, then exit"},
    "exit.btn.keep": {LANG_ZH: "保留 cache 和 logs 后退出", LANG_EN: "Keep cache and logs, then exit"},
    "exit.btn.cancel": {LANG_ZH: "取消", LANG_EN: "Cancel"},
    "exit.cleanup_failed_hint": {
        LANG_ZH: "{errors}\n\n可先关闭占用缓存视频的程序，再手动清理 cache。",
        LANG_EN: "{errors}\n\nClose programs that occupy cached video files, then clean cache manually.",
    },
    "model.weights_filter": {LANG_ZH: "PyTorch Weights (*.pt)", LANG_EN: "PyTorch Weights (*.pt)"},
    "model.video_filter": {LANG_ZH: "Video Files (*.mp4 *.avi *.mov *.mkv)", LANG_EN: "Video Files (*.mp4 *.avi *.mov *.mkv)"},
    "model.image_filter": {
        LANG_ZH: "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        LANG_EN: "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
    },
}

HELP_TEXTS_BY_LANG: dict[str, dict[str, str]] = {
    LANG_ZH: {
        "input_mode": "输入模式：图片用于单图推理，视频用于逐帧推理演示。",
        "input_path": "输入路径：图片模式选择单张图片，视频模式选择单个视频文件。",
        "output_dir": "输出目录：仅在手动保存结果时写入该目录。",
        "enable_detection": "启用检测：关闭后不执行检测推理，仅显示原图。",
        "det_model_path": "检测模型路径：检测框模型权重路径（.pt）。",
        "seg_model_path": "分割模型路径：裂缝分割模型权重路径（.pt）。",
        "enable_segmentation": "启用分割：勾选后在检测基础上执行分割；不勾选时仅显示检测框。",
        "cache_only": "仅缓存预览：结果默认写入 cache 用于演示，不自动写入 outputs。",
        "start_run": "开始：启动后台推理线程。",
        "stop_run": "停止：请求后台线程安全停止。",
        "manual_save": "手动保存当前结果：将当前预览结果保存到输出目录。",
        "open_output": "打开输出目录：在系统文件管理器中打开输出目录。",
        "clear_log": "清空日志：仅清空界面日志显示，不删除磁盘日志文件。",
        "det_conf": "检测置信度阈值。值越低越容易保留检测框，但误检可能增多。",
        "seg_conf": "分割置信度阈值。值越低越容易保留分割结果，但噪声可能增多。",
        "seg_thr": "分割二值阈值。值越高 mask 越保守，值越低覆盖越大。",
        "post_open": "开运算核大小。用于去除小噪点，0 表示关闭。",
        "post_close": "闭运算核大小。用于连接细小断裂，0 表示关闭。",
        "post_min_area": "最小连通域面积。小于该面积的碎片将被过滤。",
        "frame_step": "视频抽帧步长。每隔 N 帧处理一次，N 越大越快。",
        "max_frames": "最大处理帧数。0 表示不限制，适合演示时快速截断。",
        "preview_interval": "预览刷新间隔。每处理 N 个有效帧刷新一次界面，值越大越流畅。",
        "toggle_result_panel": "显示结果摘要区：控制右侧摘要区域显示/隐藏。",
        "video_view_mode": "视频预览模式：在“单帧预览”和“缓存视频播放”之间切换。",
        "video_source": "视频播放源：选择播放 overlay 或 mask 缓存视频。",
    },
    LANG_EN: {
        "input_mode": "Input mode: image mode for single-image inference, video mode for frame-by-frame inference.",
        "input_path": "Input path: choose one image in image mode or one video file in video mode.",
        "output_dir": "Output directory: written only when manually saving results.",
        "enable_detection": "Enable detection: if disabled, detection inference is skipped and only original image is shown.",
        "det_model_path": "Detection model path: weights path for detection model (.pt).",
        "seg_model_path": "Segmentation model path: weights path for crack segmentation model (.pt).",
        "enable_segmentation": "Enable segmentation: runs on top of detection when enabled; otherwise only boxes are shown.",
        "cache_only": "Cache preview only: results are written to cache for demonstration and not auto-saved to outputs.",
        "start_run": "Start: start the background inference thread.",
        "stop_run": "Stop: request the background thread to stop safely.",
        "manual_save": "Manual save: save current preview result to output directory.",
        "open_output": "Open output directory in system file manager.",
        "clear_log": "Clear log: clear UI log view only; does not delete disk log files.",
        "det_conf": "Detection confidence threshold. Lower values keep more boxes but may increase false positives.",
        "seg_conf": "Segmentation confidence threshold. Lower values keep more masks but may increase noise.",
        "seg_thr": "Segmentation binarization threshold. Higher values make masks more conservative.",
        "post_open": "Opening kernel size for removing small noise. 0 disables it.",
        "post_close": "Closing kernel size for connecting small breaks. 0 disables it.",
        "post_min_area": "Minimum connected component area. Smaller fragments are filtered out.",
        "frame_step": "Video frame step. Process every N frames; larger N is faster.",
        "max_frames": "Maximum processed frames. 0 means unlimited.",
        "preview_interval": "Preview refresh interval. Refresh UI every N valid frames.",
        "toggle_result_panel": "Show/hide the right summary panel.",
        "video_view_mode": "Video preview mode: switch between frame preview and cached video playback.",
        "video_source": "Video source: choose overlay or mask cached video.",
    },
}


def normalize_language(value: str | None) -> str:
    if not value:
        return LANG_ZH
    value = value.strip().lower()
    if value.startswith("en"):
        return LANG_EN
    if value.startswith("zh"):
        return LANG_ZH
    return LANG_ZH


def tr(language: str, key: str, **kwargs: Any) -> str:
    lang = normalize_language(language)
    entry = TEXTS.get(key)
    if not entry:
        return key
    template = entry.get(lang) or entry.get(LANG_ZH) or key
    if kwargs:
        return template.format(**kwargs)
    return template


def help_text(language: str, key: str) -> str:
    lang = normalize_language(language)
    return HELP_TEXTS_BY_LANG.get(lang, {}).get(key, HELP_TEXTS_BY_LANG[LANG_ZH].get(key, ""))
