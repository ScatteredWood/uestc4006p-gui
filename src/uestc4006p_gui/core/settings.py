from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import InferenceParams


@dataclass(frozen=True)
class UiDefaults:
    enable_detection: bool = True
    enable_segmentation: bool = True
    cache_only: bool = True
    input_mode: str = "image"
    params: InferenceParams = field(default_factory=InferenceParams)


UI_DEFAULTS = UiDefaults()


HELP_TEXTS: dict[str, str] = {
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
}