from __future__ import annotations

import gc
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
import shutil

import cv2
import yaml
from PySide6.QtCore import QSettings, Qt, QThread, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.paths import CACHE_DIR, DEFAULT_MODELS_YAML, LOGS_DIR, OUTPUTS_DIR, ensure_runtime_dirs
from ..core.schemas import FrameResult, InferenceParams, RunRequest, RunSummary
from ..core.settings import HELP_TEXTS, UI_DEFAULTS
from ..inference.cascade_engine import CascadeEngine
from ..inference.result_writer import ResultWriter
from .worker import InferenceWorker

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget

    QT_MEDIA_AVAILABLE = True
except Exception:  # pragma: no cover
    QAudioOutput = None
    QMediaPlayer = None
    QVideoWidget = None
    QT_MEDIA_AVAILABLE = False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()

        self.engine = CascadeEngine()
        self.writer = ResultWriter()

        self._thread: QThread | None = None
        self._worker: InferenceWorker | None = None
        self._is_running = False

        self.current_request: RunRequest | None = None
        self.current_summary: RunSummary | None = None
        self.current_frame: FrameResult | None = None

        self.cached_overlay_video: str = ""
        self.cached_mask_video: str = ""
        self.current_play_video_path: str = ""
        self._slider_dragging = False

        self.last_ui_convert_ms: float = 0.0
        self.last_ui_refresh_ms: float = 0.0
        self.last_write_ms: float = 0.0
        self.last_write_action: str = "none"

        self.log_file = LOGS_DIR / "gui.log"
        self.audio_output = None
        self.media_player = None
        self.settings = QSettings("UESTC4006P", "GUI")
        self._cleanup_retry_queue: list[Path] = []

        self.setWindowTitle("UESTC4006P GUI MVP")
        self.resize(1680, 920)

        self._setup_ui()
        self._init_video_player()
        self._load_defaults_from_yaml()
        self._load_path_history()
        self._on_mode_changed()
        self._set_running(False)
        self._update_result_summary()

    def _setup_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        self.left_panel = self._build_left_panel()
        self.center_panel = self._build_center_panel()
        self.result_panel = self._build_result_panel()

        self.left_panel.setFixedWidth(420)
        self.left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.left_panel.setMinimumHeight(0)
        self.result_panel.setFixedWidth(340)
        self.result_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.result_panel.setMinimumHeight(0)
        result_policy = self.result_panel.sizePolicy()
        result_policy.setRetainSizeWhenHidden(True)
        self.result_panel.setSizePolicy(result_policy)

        top_layout.addWidget(self.left_panel, 0)
        top_layout.addWidget(self.center_panel, 1)
        top_layout.addWidget(self.result_panel, 0)

        log_panel = self._build_log_panel()
        log_panel.setFixedHeight(120)
        log_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        top_widget = QWidget(root)
        top_widget.setLayout(top_layout)
        top_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root_layout.addWidget(top_widget, 1)
        root_layout.addWidget(log_panel, 0)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(260)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("就绪")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        group_input = QGroupBox("输入与模型", panel)
        group_input.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        form_input = QFormLayout(group_input)
        form_input.setLabelAlignment(Qt.AlignRight)
        form_input.setHorizontalSpacing(8)
        form_input.setVerticalSpacing(6)
        form_input.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.mode_combo = QComboBox(group_input)
        self.mode_combo.addItem("图片", "image")
        self.mode_combo.addItem("视频", "video")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._apply_help(self.mode_combo, "input_mode")
        form_input.addRow("输入模式", self.mode_combo)

        self.det_model_edit, det_btn = self._make_path_row(select_dir=False)
        self._apply_help(self.det_model_edit, "det_model_path")
        self._apply_help(det_btn, "det_model_path")
        det_btn.clicked.connect(
            lambda: self._browse_file(
                self.det_model_edit,
                "选择检测模型",
                "PyTorch Weights (*.pt)",
                "last_det_model_dir",
                "det_model_path",
            )
        )
        form_input.addRow("检测模型路径", self._join_row(self.det_model_edit, det_btn))

        self.seg_model_edit, seg_btn = self._make_path_row(select_dir=False)
        self._apply_help(self.seg_model_edit, "seg_model_path")
        self._apply_help(seg_btn, "seg_model_path")
        seg_btn.clicked.connect(
            lambda: self._browse_file(
                self.seg_model_edit,
                "选择分割模型",
                "PyTorch Weights (*.pt)",
                "last_seg_model_dir",
                "seg_model_path",
            )
        )
        form_input.addRow("分割模型路径", self._join_row(self.seg_model_edit, seg_btn))

        self.input_path_edit, input_btn = self._make_path_row(select_dir=False)
        self._apply_help(self.input_path_edit, "input_path")
        self._apply_help(input_btn, "input_path")
        input_btn.clicked.connect(self._browse_input_by_mode)
        form_input.addRow("输入文件", self._join_row(self.input_path_edit, input_btn))

        self.output_dir_edit, output_btn = self._make_path_row(select_dir=True)
        self._apply_help(self.output_dir_edit, "output_dir")
        self._apply_help(output_btn, "output_dir")
        output_btn.clicked.connect(
            lambda: self._browse_dir(self.output_dir_edit, "选择输出目录", "last_output_dir", "output_dir")
        )
        form_input.addRow("输出目录", self._join_row(self.output_dir_edit, output_btn))

        group_switch = QGroupBox("开关", panel)
        group_switch.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        switch_layout = QVBoxLayout(group_switch)
        switch_layout.setContentsMargins(8, 4, 8, 4)
        switch_layout.setSpacing(3)

        self.enable_det_check = QCheckBox("启用检测", group_switch)
        self.enable_det_check.setChecked(UI_DEFAULTS.enable_detection)
        self._apply_help(self.enable_det_check, "enable_detection")
        switch_layout.addWidget(self.enable_det_check)

        self.enable_seg_check = QCheckBox("启用分割", group_switch)
        self.enable_seg_check.setChecked(UI_DEFAULTS.enable_segmentation)
        self.enable_seg_check.toggled.connect(self._on_seg_toggled)
        self._apply_help(self.enable_seg_check, "enable_segmentation")
        switch_layout.addWidget(self.enable_seg_check)

        self.cache_only_check = QCheckBox("仅缓存预览", group_switch)
        self.cache_only_check.setChecked(UI_DEFAULTS.cache_only)
        self._apply_help(self.cache_only_check, "cache_only")
        switch_layout.addWidget(self.cache_only_check)
        group_params = QGroupBox("参数", panel)
        group_params.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        form_params = QFormLayout(group_params)
        form_params.setLabelAlignment(Qt.AlignRight)
        form_params.setHorizontalSpacing(8)
        form_params.setVerticalSpacing(6)
        form_params.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.det_conf_spin = QDoubleSpinBox(group_params)
        self.det_conf_spin.setRange(0.01, 1.0)
        self.det_conf_spin.setSingleStep(0.01)
        self.det_conf_spin.setValue(UI_DEFAULTS.params.det_conf)
        self._apply_help(self.det_conf_spin, "det_conf")
        form_params.addRow("检测置信度 (det_conf)", self.det_conf_spin)

        self.seg_conf_spin = QDoubleSpinBox(group_params)
        self.seg_conf_spin.setRange(0.01, 1.0)
        self.seg_conf_spin.setSingleStep(0.01)
        self.seg_conf_spin.setValue(UI_DEFAULTS.params.seg_conf)
        self._apply_help(self.seg_conf_spin, "seg_conf")
        form_params.addRow("分割置信度 (seg_conf)", self.seg_conf_spin)

        self.seg_thr_spin = QDoubleSpinBox(group_params)
        self.seg_thr_spin.setRange(0.01, 1.0)
        self.seg_thr_spin.setSingleStep(0.01)
        self.seg_thr_spin.setValue(UI_DEFAULTS.params.seg_thr)
        self._apply_help(self.seg_thr_spin, "seg_thr")
        form_params.addRow("分割二值阈值 (seg_thr)", self.seg_thr_spin)

        self.post_open_spin = QSpinBox(group_params)
        self.post_open_spin.setRange(0, 31)
        self.post_open_spin.setValue(UI_DEFAULTS.params.post_open)
        self._apply_help(self.post_open_spin, "post_open")
        form_params.addRow("开运算核大小 (post_open)", self.post_open_spin)

        self.post_close_spin = QSpinBox(group_params)
        self.post_close_spin.setRange(0, 31)
        self.post_close_spin.setValue(UI_DEFAULTS.params.post_close)
        self._apply_help(self.post_close_spin, "post_close")
        form_params.addRow("闭运算核大小 (post_close)", self.post_close_spin)

        self.post_min_area_spin = QSpinBox(group_params)
        self.post_min_area_spin.setRange(0, 100000)
        self.post_min_area_spin.setValue(UI_DEFAULTS.params.post_min_area)
        self._apply_help(self.post_min_area_spin, "post_min_area")
        form_params.addRow("最小连通域面积 (post_min_area)", self.post_min_area_spin)

        self.frame_step_spin = QSpinBox(group_params)
        self.frame_step_spin.setRange(1, 999)
        self.frame_step_spin.setValue(UI_DEFAULTS.params.frame_step)
        self._apply_help(self.frame_step_spin, "frame_step")
        form_params.addRow("视频抽帧步长 (frame_step)", self.frame_step_spin)

        self.max_frames_spin = QSpinBox(group_params)
        self.max_frames_spin.setRange(0, 999999)
        self.max_frames_spin.setValue(UI_DEFAULTS.params.max_frames)
        self._apply_help(self.max_frames_spin, "max_frames")
        form_params.addRow("最大处理帧数 (max_frames)", self.max_frames_spin)

        self.preview_interval_spin = QSpinBox(group_params)
        self.preview_interval_spin.setRange(1, 120)
        self.preview_interval_spin.setValue(UI_DEFAULTS.params.preview_interval)
        self._apply_help(self.preview_interval_spin, "preview_interval")
        form_params.addRow("预览刷新间隔 (preview_interval)", self.preview_interval_spin)

        group_actions = QGroupBox("操作", panel)
        group_actions.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        actions_layout = QVBoxLayout(group_actions)
        actions_layout.setContentsMargins(8, 8, 8, 8)
        actions_layout.setSpacing(6)

        self.start_btn = QPushButton("开始", group_actions)
        self.start_btn.clicked.connect(self._start_run)
        self._apply_help(self.start_btn, "start_run")
        actions_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止", group_actions)
        self.stop_btn.clicked.connect(self._stop_run)
        self._apply_help(self.stop_btn, "stop_run")
        actions_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("手动保存当前结果", group_actions)
        self.save_btn.clicked.connect(self._save_current_result)
        self._apply_help(self.save_btn, "manual_save")
        actions_layout.addWidget(self.save_btn)

        self.open_output_btn = QPushButton("打开输出目录", group_actions)
        self.open_output_btn.clicked.connect(self._open_output_dir)
        self._apply_help(self.open_output_btn, "open_output")
        actions_layout.addWidget(self.open_output_btn)

        self.clear_log_btn = QPushButton("清空日志", group_actions)
        self.clear_log_btn.clicked.connect(self._clear_logs)
        self._apply_help(self.clear_log_btn, "clear_log")
        actions_layout.addWidget(self.clear_log_btn)

        layout.addWidget(group_input)
        layout.addWidget(group_switch)
        layout.addWidget(group_params)
        layout.addWidget(group_actions)
        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)

        top.addWidget(QLabel("预览模式", panel))
        self.preview_combo = QComboBox(panel)
        self.preview_combo.addItem("原图", "original")
        self.preview_combo.addItem("Overlay", "overlay")
        self.preview_combo.addItem("Mask", "mask")
        self.preview_combo.currentIndexChanged.connect(self._on_preview_mode_changed)
        top.addWidget(self.preview_combo)

        self.video_view_combo = QComboBox(panel)
        self.video_view_combo.addItem("单帧预览", "frame")
        self.video_view_combo.addItem("缓存视频播放", "video")
        self.video_view_combo.currentIndexChanged.connect(self._on_video_view_changed)
        self._apply_help(self.video_view_combo, "video_view_mode")
        top.addWidget(self.video_view_combo)

        self.video_source_combo = QComboBox(panel)
        self.video_source_combo.addItem("Overlay 视频", "overlay")
        self.video_source_combo.addItem("Mask 视频", "mask")
        self.video_source_combo.currentIndexChanged.connect(self._on_video_source_changed)
        self._apply_help(self.video_source_combo, "video_source")
        top.addWidget(self.video_source_combo)

        self.show_result_check = QCheckBox("显示右侧结果摘要", panel)
        self.show_result_check.setChecked(True)
        self._apply_help(self.show_result_check, "toggle_result_panel")
        self.show_result_check.setFixedHeight(self.show_result_check.sizeHint().height())
        self.show_result_check.toggled.connect(self._toggle_result_panel)
        top.addWidget(self.show_result_check)

        top.addStretch(1)
        self.preview_info = QLabel("暂无预览", panel)
        self.preview_info.setWordWrap(False)
        self.preview_info.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.preview_info.setMinimumWidth(220)
        self.preview_info.setFixedHeight(self.preview_info.sizeHint().height())
        top.addWidget(self.preview_info)

        self.preview_stack = QStackedWidget(panel)

        frame_page = QWidget(panel)
        frame_layout = QVBoxLayout(frame_page)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel(frame_page)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("请选择输入并点击开始")
        self.preview_label.setWordWrap(False)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        scroll = QScrollArea(frame_page)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setWidget(self.preview_label)
        frame_layout.addWidget(scroll)

        video_page = QWidget(panel)
        video_layout = QVBoxLayout(video_page)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(6)

        if QT_MEDIA_AVAILABLE and QVideoWidget is not None:
            self.video_widget = QVideoWidget(video_page)
        else:
            self.video_widget = QLabel("QtMultimedia 不可用，无法播放缓存视频。", video_page)
            self.video_widget.setAlignment(Qt.AlignCenter)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_widget, 1)

        controls = QHBoxLayout()
        self.play_pause_btn = QPushButton("播放", video_page)
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        controls.addWidget(self.play_pause_btn)

        self.stop_video_btn = QPushButton("停止", video_page)
        self.stop_video_btn.clicked.connect(self._stop_playback)
        controls.addWidget(self.stop_video_btn)

        self.video_slider = QSlider(Qt.Horizontal, video_page)
        self.video_slider.setRange(0, 0)
        self.video_slider.sliderPressed.connect(self._on_slider_pressed)
        self.video_slider.sliderReleased.connect(self._on_slider_released)
        self.video_slider.sliderMoved.connect(self._seek_position)
        controls.addWidget(self.video_slider, 1)

        self.video_time_label = QLabel("00:00 / 00:00", video_page)
        controls.addWidget(self.video_time_label)

        video_layout.addLayout(controls)

        self.preview_stack.addWidget(frame_page)
        self.preview_stack.addWidget(video_page)

        layout.addLayout(top)
        self.preview_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.preview_stack, 1)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QGroupBox("当前结果摘要", self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        self.result_text = QTextEdit(panel)
        self.result_text.setReadOnly(True)
        self.result_text.setLineWrapMode(QTextEdit.NoWrap)
        self.result_text.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.result_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_text.setMinimumHeight(0)
        layout.addWidget(self.result_text)
        return panel

    def _build_log_panel(self) -> QWidget:
        panel = QGroupBox("过程日志", self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        self.log_text = QTextEdit(panel)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return panel

    def _init_video_player(self) -> None:
        if not QT_MEDIA_AVAILABLE or QMediaPlayer is None or QAudioOutput is None:
            self._set_video_controls_enabled(False)
            self.preview_info.setText("QtMultimedia 不可用")
            return

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.0)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        self.media_player.positionChanged.connect(self._on_player_position_changed)
        self.media_player.durationChanged.connect(self._on_player_duration_changed)
        self.media_player.playbackStateChanged.connect(self._on_player_state_changed)

        self._set_video_controls_enabled(False)

    @staticmethod
    def _make_path_row(select_dir: bool = False):
        edit = QLineEdit()
        edit.setPlaceholderText("选择目录" if select_dir else "选择文件")
        btn = QPushButton("浏览")
        return edit, btn

    @staticmethod
    def _join_row(edit: QLineEdit, btn: QPushButton) -> QWidget:
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(btn, 0)
        return box

    @staticmethod
    def _apply_help(widget: QWidget, key: str) -> None:
        text = HELP_TEXTS.get(key, "")
        if text:
            widget.setToolTip(text)
            widget.setStatusTip(text)

    @staticmethod
    def _safe_existing_dir(path_text: str) -> Path:
        text = path_text.strip()
        if text:
            p = Path(text).expanduser()
            if p.is_file():
                p = p.parent
            if p.exists() and p.is_dir():
                return p
        return Path.home()

    def _set_path_history(self, dir_key: str, value: Path) -> None:
        target_dir = value if value.is_dir() else value.parent
        self.settings.setValue(f"paths/{dir_key}", str(target_dir))

    def _set_path_value(self, value_key: str, value: Path) -> None:
        self.settings.setValue(f"paths/{value_key}", str(value))

    def _dialog_start_dir(self, edit: QLineEdit, dir_key: str) -> str:
        saved_dir = str(self.settings.value(f"paths/{dir_key}", "", type=str) or "")
        if saved_dir:
            p = Path(saved_dir)
            if p.exists() and p.is_dir():
                return str(p)
        return str(self._safe_existing_dir(edit.text()))

    def _load_path_history(self) -> None:
        det_model = str(self.settings.value("paths/det_model_path", "", type=str) or "").strip()
        seg_model = str(self.settings.value("paths/seg_model_path", "", type=str) or "").strip()
        input_path = str(self.settings.value("paths/input_path", "", type=str) or "").strip()
        output_dir = str(self.settings.value("paths/output_dir", "", type=str) or "").strip()

        self.det_model_edit.setText(det_model if det_model and Path(det_model).exists() else "")
        self.seg_model_edit.setText(seg_model if seg_model and Path(seg_model).exists() else "")
        self.input_path_edit.setText(input_path if input_path and Path(input_path).exists() else "")

        if output_dir and Path(output_dir).exists() and Path(output_dir).is_dir():
            self.output_dir_edit.setText(output_dir)
        elif not self.output_dir_edit.text().strip() or not Path(self.output_dir_edit.text().strip()).exists():
            self.output_dir_edit.setText(str(Path.home()))

    def _persist_current_paths(self) -> None:
        mapping = [
            (self.det_model_edit, "last_det_model_dir", "det_model_path"),
            (self.seg_model_edit, "last_seg_model_dir", "seg_model_path"),
            (self.input_path_edit, "last_input_dir", "input_path"),
            (self.output_dir_edit, "last_output_dir", "output_dir"),
        ]
        for edit, dir_key, value_key in mapping:
            text = edit.text().strip()
            if not text:
                continue
            p = Path(text)
            self._set_path_history(dir_key, p)
            self._set_path_value(value_key, p)

    def _load_defaults_from_yaml(self) -> None:
        self.det_model_edit.clear()
        self.seg_model_edit.clear()
        self.output_dir_edit.setText(str(OUTPUTS_DIR))
        if not DEFAULT_MODELS_YAML.exists():
            return
        try:
            cfg = yaml.safe_load(DEFAULT_MODELS_YAML.read_text(encoding="utf-8")) or {}
            out_cfg = str(cfg.get("default_output_dir", OUTPUTS_DIR)).strip()
            out_path = Path(out_cfg) if out_cfg else OUTPUTS_DIR
            self.output_dir_edit.setText(str(out_path if out_path.exists() else Path.home()))
        except Exception as exc:
            self._append_log(f"[WARN] 读取 default_models.yaml 失败: {exc}")

    def _browse_input_by_mode(self) -> None:
        mode = self.mode_combo.currentData()
        if mode == "video":
            self._browse_file(
                self.input_path_edit,
                "选择视频",
                "Video Files (*.mp4 *.avi *.mov *.mkv)",
                "last_input_dir",
                "input_path",
            )
        else:
            self._browse_file(
                self.input_path_edit,
                "选择图片",
                "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
                "last_input_dir",
                "input_path",
            )

    def _browse_file(
        self,
        edit: QLineEdit,
        title: str,
        pattern: str,
        dir_key: str,
        value_key: str,
    ) -> None:
        start_dir = self._dialog_start_dir(edit, dir_key)
        f, _ = QFileDialog.getOpenFileName(self, title, start_dir, pattern)
        if f:
            target = Path(f)
            edit.setText(str(target))
            self._set_path_history(dir_key, target)
            self._set_path_value(value_key, target)

    def _browse_dir(self, edit: QLineEdit, title: str, dir_key: str, value_key: str) -> None:
        start_dir = self._dialog_start_dir(edit, dir_key)
        d = QFileDialog.getExistingDirectory(self, title, start_dir)
        if d:
            target = Path(d)
            edit.setText(str(target))
            self._set_path_history(dir_key, target)
            self._set_path_value(value_key, target)

    def _on_mode_changed(self) -> None:
        is_video = self.mode_combo.currentData() == "video"
        self.frame_step_spin.setEnabled(is_video)
        self.max_frames_spin.setEnabled(is_video)
        self.preview_interval_spin.setEnabled(is_video)

        self.video_view_combo.setVisible(is_video)
        self.video_source_combo.setVisible(is_video)

        if is_video:
            if self.video_view_combo.currentData() == "video":
                self.preview_stack.setCurrentIndex(1)
                self._load_selected_cached_video()
            else:
                self.preview_stack.setCurrentIndex(0)
            self.statusBar().showMessage("视频模式：可按帧预览，也可播放缓存视频。")
        else:
            self.video_view_combo.setCurrentIndex(0)
            self.preview_stack.setCurrentIndex(0)
            self._release_video_playback_handles()
            self.statusBar().showMessage("图片模式：单图推理。")

    def _on_seg_toggled(self, enabled: bool) -> None:
        self.seg_model_edit.setEnabled(enabled)
        self.seg_conf_spin.setEnabled(enabled)
        self.seg_thr_spin.setEnabled(enabled)
        self.post_open_spin.setEnabled(enabled)
        self.post_close_spin.setEnabled(enabled)
        self.post_min_area_spin.setEnabled(enabled)

    def _toggle_result_panel(self, checked: bool) -> None:
        self.result_panel.setVisible(checked)

    def _on_preview_mode_changed(self) -> None:
        self._refresh_preview()
        self._update_result_summary()

    def _on_video_view_changed(self) -> None:
        mode = self.video_view_combo.currentData()
        if mode == "video":
            self.preview_stack.setCurrentIndex(1)
            self._load_selected_cached_video()
        else:
            self.preview_stack.setCurrentIndex(0)
            self._release_video_playback_handles()

    def _on_video_source_changed(self) -> None:
        if self.video_view_combo.currentData() == "video":
            self._load_selected_cached_video()

    def _release_video_playback_handles(self) -> None:
        if self.media_player is not None:
            try:
                self.media_player.stop()
            except Exception:
                pass
            try:
                self.media_player.setSource(QUrl())
            except Exception:
                pass
        self.current_play_video_path = ""
        self._slider_dragging = False
        self.video_slider.blockSignals(True)
        self.video_slider.setRange(0, 0)
        self.video_slider.setValue(0)
        self.video_slider.blockSignals(False)
        self.video_time_label.setText("00:00 / 00:00")
        self.play_pause_btn.setText("播放")
        self._set_video_controls_enabled(False)

    @staticmethod
    def _flush_pending_handles() -> None:
        QApplication.processEvents()
        gc.collect()

    def _release_media_before_cleanup(self) -> None:
        self._release_video_playback_handles()
        self._flush_pending_handles()

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        self.play_pause_btn.setEnabled(enabled)
        self.stop_video_btn.setEnabled(enabled)
        self.video_slider.setEnabled(enabled)
        self.video_source_combo.setEnabled(enabled)

    def _refresh_video_sources(self) -> None:
        has_mask = bool(self.cached_mask_video and Path(self.cached_mask_video).exists())
        self.video_source_combo.setItemText(1, "Mask 视频" if has_mask else "Mask 视频 (不可用)")
        if not has_mask and self.video_source_combo.currentData() == "mask":
            self.video_source_combo.setCurrentIndex(0)

    def _apply_default_preview_after_finish(self, summary: RunSummary) -> None:
        preferred_preview = "original"
        if self.current_frame is not None:
            if self.current_frame.overlay_bgr is not None and self.current_frame.overlay_bgr.size > 0:
                preferred_preview = "overlay"
            elif self.current_frame.mask_u8 is not None and self.current_frame.mask_u8.size > 0:
                preferred_preview = "mask"

        preview_idx = self.preview_combo.findData(preferred_preview)
        if preview_idx >= 0 and self.preview_combo.currentIndex() != preview_idx:
            self.preview_combo.setCurrentIndex(preview_idx)

        if summary.mode != "video":
            return
        if summary.cache_overlay_video and Path(summary.cache_overlay_video).exists():
            preferred_video_source = "overlay"
        elif summary.cache_mask_video and Path(summary.cache_mask_video).exists():
            preferred_video_source = "mask"
        else:
            preferred_video_source = "overlay"
        source_idx = self.video_source_combo.findData(preferred_video_source)
        if source_idx >= 0 and self.video_source_combo.currentIndex() != source_idx:
            self.video_source_combo.setCurrentIndex(source_idx)

    def _load_selected_cached_video(self) -> None:
        if self.media_player is None:
            self._set_video_controls_enabled(False)
            self.preview_info.setText("QtMultimedia 不可用")
            return

        source = self.video_source_combo.currentData()
        target = self.cached_overlay_video if source == "overlay" else self.cached_mask_video

        if not target or not Path(target).exists():
            self._set_video_controls_enabled(False)
            self.preview_info.setText("缓存视频不可用")
            self._release_video_playback_handles()
            self._update_result_summary()
            return

        if self.current_play_video_path != target:
            self._release_video_playback_handles()
            self.current_play_video_path = target
            self.media_player.setSource(QUrl.fromLocalFile(target))
            self.media_player.pause()
        self._set_video_controls_enabled(True)
        self.preview_info.setText(f"已加载缓存视频: {Path(target).name}")
        self._update_result_summary()

    def _toggle_play_pause(self) -> None:
        if self.media_player is None:
            return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def _stop_playback(self) -> None:
        self._release_video_playback_handles()

    def _on_player_position_changed(self, position: int) -> None:
        if self.media_player is None:
            return
        if not self._slider_dragging:
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(position)
            self.video_slider.blockSignals(False)
        self.video_time_label.setText(f"{self._fmt_ms(position)} / {self._fmt_ms(self.media_player.duration())}")

    def _on_player_duration_changed(self, duration: int) -> None:
        if self.media_player is None:
            return
        self.video_slider.setRange(0, max(0, duration))
        self.video_time_label.setText(f"{self._fmt_ms(self.media_player.position())} / {self._fmt_ms(duration)}")

    def _on_player_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.play_pause_btn.setText("暂停" if state == QMediaPlayer.PlayingState else "播放")

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True

    def _on_slider_released(self) -> None:
        if self.media_player is None:
            return
        self._slider_dragging = False
        self.media_player.setPosition(self.video_slider.value())

    def _seek_position(self, value: int) -> None:
        if self.media_player is None:
            return
        if self._slider_dragging:
            self.video_time_label.setText(f"{self._fmt_ms(value)} / {self._fmt_ms(self.media_player.duration())}")

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        total_sec = max(0, int(ms // 1000))
        m, s = divmod(total_sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _validate_request_inputs(self) -> tuple[bool, str]:
        mode = self.mode_combo.currentData()
        if mode not in {"image", "video"}:
            return False, "任务模式为空或非法，请先选择“图片”或“视频”。"

        input_text = self.input_path_edit.text().strip()
        if not input_text:
            return False, "请先选择输入文件。"
        input_path = Path(input_text)
        if not input_path.exists():
            return False, "输入路径不存在，请重新选择。"
        if not input_path.is_file():
            return False, "当前仅支持输入单个文件，请选择图片或视频文件。"

        output_text = self.output_dir_edit.text().strip()
        if not output_text:
            return False, "请先选择输出目录。"
        output_dir = Path(output_text)
        if output_dir.exists() and not output_dir.is_dir():
            return False, "输出路径不是目录，请重新选择。"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"输出目录不可用：{exc}"

        if self.enable_seg_check.isChecked() and not self.enable_det_check.isChecked():
            return False, "启用分割时必须同时启用检测，请调整任务组合。"

        if not self.enable_det_check.isChecked() and not self.enable_seg_check.isChecked():
            return False, "请至少启用一个任务（检测或分割）。"

        if self.enable_det_check.isChecked():
            det_text = self.det_model_edit.text().strip()
            if not det_text:
                return False, "当前任务需要检测模型，请先选择检测模型。"
            det_model_path = Path(det_text)
            if not det_model_path.exists() or not det_model_path.is_file():
                return False, "检测模型路径不存在或不是文件，请重新选择。"

        if self.enable_seg_check.isChecked():
            seg_text = self.seg_model_edit.text().strip()
            if not seg_text:
                return False, "当前任务需要分割模型，请先选择分割模型。"
            seg_model_path = Path(seg_text)
            if not seg_model_path.exists() or not seg_model_path.is_file():
                return False, "分割模型路径不存在或不是文件，请重新选择。"

        return True, ""

    def _collect_request(self) -> RunRequest | None:
        ok, message = self._validate_request_inputs()
        if not ok:
            QMessageBox.warning(self, "无法开始推理", message)
            return None

        mode = self.mode_combo.currentData()
        input_path = Path(self.input_path_edit.text().strip())
        output_dir = Path(self.output_dir_edit.text().strip())
        det_model_path = Path(self.det_model_edit.text().strip())
        seg_model_path = Path(self.seg_model_edit.text().strip())

        output_dir.mkdir(parents=True, exist_ok=True)
        params = InferenceParams(
            det_conf=float(self.det_conf_spin.value()),
            seg_conf=float(self.seg_conf_spin.value()),
            seg_thr=float(self.seg_thr_spin.value()),
            post_open=int(self.post_open_spin.value()),
            post_close=int(self.post_close_spin.value()),
            post_min_area=int(self.post_min_area_spin.value()),
            frame_step=int(self.frame_step_spin.value()),
            max_frames=int(self.max_frames_spin.value()),
            preview_interval=int(self.preview_interval_spin.value()),
        )
        return RunRequest(
            mode=mode,
            input_path=input_path,
            output_dir=output_dir,
            det_model_path=det_model_path,
            seg_model_path=seg_model_path,
            enable_detection=self.enable_det_check.isChecked(),
            enable_segmentation=self.enable_seg_check.isChecked() and self.enable_det_check.isChecked(),
            cache_only=self.cache_only_check.isChecked(),
            params=params,
        )
    def _start_run(self) -> None:
        if self._is_running:
            return

        request = self._collect_request()
        if request is None:
            return

        self._persist_current_paths()
        self.current_request = request
        self.current_summary = None
        self.current_frame = None
        self.cached_overlay_video = ""
        self.cached_mask_video = ""
        self.current_play_video_path = ""
        self._release_video_playback_handles()

        self.progress_bar.setValue(0)
        self.preview_info.setText("推理中...")
        self.preview_label.setText("推理中...")
        self._append_log(
            f"[RUN] start mode={request.mode}, input={request.input_path.name}, cache_only={request.cache_only}"
        )

        self._set_running(True)

        self._thread = QThread(self)
        self._worker = InferenceWorker(self.engine, request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.sig_log.connect(self._append_log)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_frame.connect(self._on_frame)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_failed.connect(self._on_failed)
        self._worker.sig_finished.connect(self._thread.quit)
        self._worker.sig_failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    def _stop_run(self) -> None:
        if self._worker is not None:
            self._append_log("[RUN] stop requested")
            self._worker.request_stop()

    def _on_progress(self, value: int, total: int) -> None:
        total_safe = max(1, int(total))
        progress = int(min(value, total_safe) * 100 / total_safe)
        self.progress_bar.setValue(progress)

    def _on_frame(self, frame: FrameResult) -> None:
        self.current_frame = frame
        if self.current_summary is not None:
            self.current_summary.last_frame_result = frame

        t0 = perf_counter()
        self._refresh_preview()
        self.last_ui_refresh_ms = (perf_counter() - t0) * 1000.0

        if frame.mode == "video":
            self.preview_info.setText(f"当前帧: {frame.frame_index}")
            interval = max(1, int(self.preview_interval_spin.value()))
            if frame.frame_index % max(10, interval * 5) == 0:
                self._append_log(
                    "[PERF][ui] "
                    f"frame={frame.frame_index}, "
                    f"model={frame.model_infer_ms:.1f}ms, "
                    f"post={frame.postprocess_ms:.1f}ms, "
                    f"viz={frame.viz_ms:.1f}ms, "
                    f"ui_convert={self.last_ui_convert_ms:.1f}ms, "
                    f"ui_refresh={self.last_ui_refresh_ms:.1f}ms"
                )
        else:
            self.preview_info.setText("图片推理完成")

        self._update_result_summary()

    def _on_finished(self, summary: RunSummary) -> None:
        self.current_summary = summary
        if summary.last_frame_result is not None:
            self.current_frame = summary.last_frame_result

        self.cached_overlay_video = summary.cache_overlay_video or ""
        self.cached_mask_video = summary.cache_mask_video or ""
        self._refresh_video_sources()
        self._apply_default_preview_after_finish(summary)

        if self.current_request is not None and self.current_request.cache_only:
            t0 = perf_counter()
            cache_dir = self.writer.cache_run(self.current_request, summary)
            self.last_write_ms = (perf_counter() - t0) * 1000.0
            self.last_write_action = "cache_run"
            if cache_dir is not None:
                self._append_log(f"[CACHE] 结果快照已缓存: {cache_dir}")

        if summary.mode == "video" and self.video_view_combo.currentData() == "video":
            self._load_selected_cached_video()

        self.progress_bar.setValue(100)
        self._append_log(
            f"[DONE] {summary.message} | processed={summary.processed_frames}/{summary.total_frames} | "
            f"elapsed={summary.elapsed_seconds:.2f}s"
        )
        if summary.cache_overlay_video:
            self._append_log(f"[CACHE][VIDEO] overlay: {summary.cache_overlay_video}")
        if summary.cache_mask_video:
            self._append_log(f"[CACHE][VIDEO] mask: {summary.cache_mask_video}")

        self._refresh_preview()
        self._update_result_summary()
        self._set_running(False)

    def _on_failed(self, message: str) -> None:
        self._append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "推理失败", message)
        self._set_running(False)

    def _on_thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _set_running(self, running: bool) -> None:
        self._is_running = running

        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

        for w in [
            self.mode_combo,
            self.det_model_edit,
            self.seg_model_edit,
            self.input_path_edit,
            self.output_dir_edit,
            self.enable_det_check,
            self.enable_seg_check,
            self.cache_only_check,
            self.det_conf_spin,
            self.seg_conf_spin,
            self.seg_thr_spin,
            self.post_open_spin,
            self.post_close_spin,
            self.post_min_area_spin,
            self.frame_step_spin,
            self.max_frames_spin,
            self.preview_interval_spin,
        ]:
            w.setEnabled(not running)

    def _save_current_result(self) -> None:
        if self.current_request is None or self.current_summary is None:
            QMessageBox.information(self, "提示", "当前没有可保存结果。")
            return

        try:
            t0 = perf_counter()
            run_dir = self.writer.save_current_result(self.current_request, self.current_summary)
            self.last_write_ms = (perf_counter() - t0) * 1000.0
            self.last_write_action = "manual_save"
            self._append_log(f"[SAVE] 已保存: {run_dir}")
            self.statusBar().showMessage("已保存当前结果", 3000)
            self._update_result_summary()
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def _open_output_dir(self) -> None:
        out_dir = Path(self.output_dir_edit.text().strip() or str(OUTPUTS_DIR))
        out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    def _clear_logs(self) -> None:
        self.log_text.clear()

    def _append_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self.log_text.append(line)
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _refresh_preview(self) -> None:
        if self.current_frame is None:
            return

        mode = self.preview_combo.currentData()
        if mode == "original":
            arr = self.current_frame.original_bgr
        elif mode == "mask":
            if self.current_frame.mask_u8 is None:
                self.preview_label.setText("当前无 mask")
                return
            arr = self.current_frame.mask_u8
        else:
            arr = self.current_frame.overlay_bgr

        pixmap = self._to_pixmap(arr)
        if pixmap.isNull():
            self.preview_label.setText("预览转换失败")
            return

        target_size = self.preview_label.size()
        if target_size.width() > 20 and target_size.height() > 20:
            pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)

    def _to_pixmap(self, img) -> QPixmap:
        t0 = perf_counter()
        if img is None:
            return QPixmap()

        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg)
        self.last_ui_convert_ms = (perf_counter() - t0) * 1000.0
        return pm
    def _update_result_summary(self) -> None:
        if self.current_request is None and self.current_frame is None and self.current_summary is None:
            self.result_text.setPlainText("暂无结果。运行一次推理后会在这里显示摘要。")
            return

        lines: list[str] = []

        if self.current_request is not None:
            lines.append(f"输入文件: {self.current_request.input_path.name}")
            lines.append(f"模式: {'视频' if self.current_request.mode == 'video' else '图片'}")

        if self.current_frame is not None:
            if self.current_frame.mode == "video":
                lines.append(f"当前帧号: {self.current_frame.frame_index}")
            lines.append(f"检测框数量: {self.current_frame.det_count}")
            lines.append(f"最高检测置信度: {self.current_frame.det_conf_max:.3f}")
            lines.append(f"平均检测置信度: {self.current_frame.det_conf_mean:.3f}")
            lines.append(f"启用分割: {'是' if self.current_frame.segmentation_enabled else '否'}")

            total_proc = (
                self.current_frame.model_infer_ms
                + self.current_frame.postprocess_ms
                + self.current_frame.viz_ms
            )
            lines.append(f"当前处理耗时: {total_proc:.1f} ms")
            lines.append(f"估算 FPS: {self.current_frame.est_fps:.2f}")

        preview_map = {"original": "原图", "overlay": "overlay", "mask": "mask"}
        lines.append(f"当前预览模式: {preview_map.get(self.preview_combo.currentData(), '原图')}")

        lines.append("")
        lines.append("性能分解:")
        if self.current_frame is not None:
            lines.append(f"- 模型推理: {self.current_frame.model_infer_ms:.1f} ms")
            lines.append(f"- 后处理: {self.current_frame.postprocess_ms:.1f} ms")
            lines.append(f"- 可视化合成: {self.current_frame.viz_ms:.1f} ms")
        lines.append(f"- UI 图像转换: {self.last_ui_convert_ms:.1f} ms")
        lines.append(f"- UI 刷新: {self.last_ui_refresh_ms:.1f} ms")
        lines.append(f"- 写盘耗时({self.last_write_action}): {self.last_write_ms:.1f} ms")

        if self.current_summary is not None and self.current_summary.mode == "video":
            lines.append("")
            lines.append("视频缓存:")
            lines.append(f"- overlay 缓存视频: {self.current_summary.cache_overlay_video or '未生成'}")
            lines.append(f"- mask 缓存视频: {self.current_summary.cache_mask_video or '未生成'}")
            lines.append(f"- 当前可播放状态: {'可播放' if self.current_summary.cache_video_playable else '不可播放'}")

        self.result_text.setPlainText("\n".join(lines))

    def _cleanup_cache_and_logs(self) -> list[str]:
        errors: list[str] = []
        targets = [(CACHE_DIR, "cache"), (LOGS_DIR, "logs")]
        self._release_media_before_cleanup()
        self.cached_overlay_video = ""
        self.cached_mask_video = ""
        for path, name in targets:
            err = self._remove_tree_with_retry(path, name)
            if err:
                errors.append(err)
        return errors

    def _remove_tree_with_retry(self, path: Path, name: str) -> str | None:
        retries = 5
        last_exc: Exception | None = None
        for i in range(retries):
            try:
                if path.exists():
                    shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                return None
            except Exception as exc:
                last_exc = exc
                self._release_media_before_cleanup()
                sleep(0.15 * (i + 1))
        if path not in self._cleanup_retry_queue:
            self._cleanup_retry_queue.append(path)
        return f"删除 {name} 失败（已重试 {retries} 次）: {last_exc}"

    def _drain_cleanup_retry_queue(self) -> list[str]:
        pending = list(self._cleanup_retry_queue)
        self._cleanup_retry_queue.clear()
        errors: list[str] = []
        for path in pending:
            err = self._remove_tree_with_retry(path, path.name)
            if err:
                errors.append(err)
        return errors

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_running:
            ret = QMessageBox.question(
                self,
                "任务仍在运行",
                "后台推理仍在运行，是否请求停止并继续退出？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                event.ignore()
                return
            if self._worker is not None:
                self._worker.request_stop()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(3000)
            if self._thread is not None and self._thread.isRunning():
                QMessageBox.warning(self, "退出失败", "推理线程尚未停止，请稍后再试。")
                event.ignore()
                return

        box = QMessageBox(self)
        box.setWindowTitle("退出确认")
        box.setIcon(QMessageBox.Question)
        box.setText("退出前如何处理缓存和日志？")
        box.setInformativeText("仅会处理 GUI 仓库下的 cache 和 logs，不会删除 outputs。")

        delete_btn = box.addButton("删除 cache 和 logs 后退出", QMessageBox.AcceptRole)
        keep_btn = box.addButton("保留 cache 和 logs 后退出", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(keep_btn)

        box.exec()
        clicked = box.clickedButton()

        if clicked == cancel_btn:
            event.ignore()
            return

        if clicked == delete_btn:
            errors = self._cleanup_cache_and_logs()
            errors.extend(self._drain_cleanup_retry_queue())
            if errors:
                QMessageBox.warning(
                    self,
                    "清理失败",
                    "\n".join(errors) + "\n\n可先关闭占用缓存视频的程序，再手动清理 cache。",
                )

        self._persist_current_paths()
        self._release_media_before_cleanup()
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.preview_stack.currentIndex() == 0:
            self._refresh_preview()

    def append_startup_hint(self, text: str) -> None:
        self._append_log(text)
