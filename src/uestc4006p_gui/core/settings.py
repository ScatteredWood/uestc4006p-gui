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


# 所有关键控件文案：用于 tooltip + statusTip。
HELP_TEXTS: dict[str, str] = {
    "input_mode": "输入模式：图片用于单图推理，视频用于按帧推理演示。",
    "input_path": "输入路径：图片模式请选择单张图片；视频模式请选择单个视频文件。",
    "output_dir": "输出目录：手动保存结果时写入该目录，默认指向 outputs。",
    "enable_detection": "启用检测：勾选后执行检测框推理；不勾选时仅显示原图。",
    "det_model_path": "检测模型路径。用于产生病害检测框，通常选择 det 的 best.pt。",
    "seg_model_path": "分割模型路径。用于检测框内裂缝分割，通常选择 seg 的 best.pt。",
    "enable_segmentation": "启用分割：勾选后会在检测结果基础上继续执行裂缝分割；不勾选时仅显示检测框。",
    "cache_only": "仅缓存预览：结果默认只保存在缓存中用于界面展示，不自动写入正式输出目录。",
    "start_run": "开始按钮：启动后台推理任务，界面可继续操作。",
    "stop_run": "停止按钮：向后台任务发送停止信号，安全结束当前处理。",
    "manual_save": "手动保存当前结果：将当前预览结果写入 outputs 目录，便于汇报留档。",
    "open_output": "打开输出目录：快速打开当前输出目录，查看导出结果。",
    "clear_log": "清空日志：只清空界面日志显示，不删除日志文件。",
    "det_conf": "det_conf：检测置信度阈值。值越低越容易保留检测框，但误检可能增多。",
    "seg_conf": "seg_conf：分割实例置信度阈值。值越低越容易保留分割结果，但噪声可能增多。",
    "seg_thr": "seg_thr：分割二值化阈值。值越高，mask 更保守；值越低，mask 覆盖更大。",
    "post_open": "post_open：形态学开运算核大小。用于去除小噪点，0 表示关闭。",
    "post_close": "post_close：形态学闭运算核大小。用于连接细小断裂，0 表示关闭。",
    "post_min_area": "post_min_area：最小连通域面积。小于该面积的碎片会被过滤。",
    "frame_step": "frame_step：视频每隔 N 帧处理一次。值越大越快，但结果更稀疏。",
    "max_frames": "max_frames：视频最多处理帧数。0 表示不限制，适合演示时快速截断。",
}
