from __future__ import annotations

import gc
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import perf_counter, sleep
import shutil

import cv2
import yaml
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl
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

from ..core.paths import (
    CACHE_DIR,
    DEFAULT_MODELS_YAML,
    DEFAULT_MODELS_YAML_CANDIDATES,
    LOGS_DIR,
    OUTPUTS_DIR,
    ensure_runtime_dirs,
    resolve_configured_path,
)
from ..core.schemas import FrameResult, InferenceParams, RunRequest, RunSummary
from ..core.settings import UI_DEFAULTS
from ..inference.cascade_engine import CascadeEngine
from ..inference.result_writer import ResultWriter
from .i18n import LANG_EN, LANG_ZH, help_text, normalize_language, tr
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
        self.language = normalize_language(self.settings.value("ui/language", LANG_ZH, type=str))
        self._help_bindings: list[tuple[QWidget, str]] = []
        self._cleanup_retry_queue: list[Path] = []
        self._startup_notices: list[str] = []
        self._updating_language_combo = False
        self._startup_check_result: tuple[bool, str] | None = None
        self._startup_check_timer: QTimer | None = None
        self._startup_check_started_at: float = 0.0

        self.setWindowTitle(self._tr("window.title"))
        self.resize(1680, 920)

        self._setup_ui()
        self._init_video_player()
        self._load_defaults_from_yaml()
        self._load_path_history()
        self._collect_runtime_startup_notices()
        self._on_mode_changed()
        self._set_running(False)
        self._update_result_summary()
        self.retranslate_ui()
        self._show_startup_notices()
        self._launch_startup_self_check()

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
        self.statusBar().showMessage(self._tr("status.ready"))

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.group_input = QGroupBox(panel)
        self.group_input.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.form_input = QFormLayout(self.group_input)
        self.form_input.setLabelAlignment(Qt.AlignRight)
        self.form_input.setHorizontalSpacing(8)
        self.form_input.setVerticalSpacing(6)
        self.form_input.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.language_combo = QComboBox(self.group_input)
        self.language_combo.addItem("", LANG_ZH)
        self.language_combo.addItem("", LANG_EN)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.label_language = QLabel(self._tr("label.language"), self.group_input)
        self.form_input.addRow(self.label_language, self.language_combo)

        self.mode_combo = QComboBox(self.group_input)
        self.mode_combo.addItem("", "image")
        self.mode_combo.addItem("", "video")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._bind_help(self.mode_combo, "input_mode")
        self.label_input_mode = QLabel(self._tr("label.input_mode"), self.group_input)
        self.form_input.addRow(self.label_input_mode, self.mode_combo)

        self.det_model_edit, self.det_model_btn = self._make_path_row(select_dir=False)
        self._bind_help(self.det_model_edit, "det_model_path")
        self._bind_help(self.det_model_btn, "det_model_path")
        self.det_model_btn.clicked.connect(
            lambda: self._browse_file(
                self.det_model_edit,
                "dialog.select_det_model",
                self._tr("model.weights_filter"),
                "last_det_model_dir",
                "det_model_path",
            )
        )
        self.det_model_row = self._join_row(self.det_model_edit, self.det_model_btn)
        self.label_det_model_path = QLabel(self._tr("label.det_model_path"), self.group_input)
        self.form_input.addRow(self.label_det_model_path, self.det_model_row)

        self.seg_model_edit, self.seg_model_btn = self._make_path_row(select_dir=False)
        self._bind_help(self.seg_model_edit, "seg_model_path")
        self._bind_help(self.seg_model_btn, "seg_model_path")
        self.seg_model_btn.clicked.connect(
            lambda: self._browse_file(
                self.seg_model_edit,
                "dialog.select_seg_model",
                self._tr("model.weights_filter"),
                "last_seg_model_dir",
                "seg_model_path",
            )
        )
        self.seg_model_row = self._join_row(self.seg_model_edit, self.seg_model_btn)
        self.label_seg_model_path = QLabel(self._tr("label.seg_model_path"), self.group_input)
        self.form_input.addRow(self.label_seg_model_path, self.seg_model_row)

        self.input_path_edit, self.input_path_btn = self._make_path_row(select_dir=False)
        self._bind_help(self.input_path_edit, "input_path")
        self._bind_help(self.input_path_btn, "input_path")
        self.input_path_btn.clicked.connect(self._browse_input_by_mode)
        self.input_path_row = self._join_row(self.input_path_edit, self.input_path_btn)
        self.label_input_file = QLabel(self._tr("label.input_file"), self.group_input)
        self.form_input.addRow(self.label_input_file, self.input_path_row)

        self.output_dir_edit, self.output_dir_btn = self._make_path_row(select_dir=True)
        self._bind_help(self.output_dir_edit, "output_dir")
        self._bind_help(self.output_dir_btn, "output_dir")
        self.output_dir_btn.clicked.connect(
            lambda: self._browse_dir(
                self.output_dir_edit,
                "dialog.select_output_dir",
                "last_output_dir",
                "output_dir",
            )
        )
        self.output_dir_row = self._join_row(self.output_dir_edit, self.output_dir_btn)
        self.label_output_dir = QLabel(self._tr("label.output_dir"), self.group_input)
        self.form_input.addRow(self.label_output_dir, self.output_dir_row)

        self.group_switch = QGroupBox(panel)
        self.group_switch.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        switch_layout = QVBoxLayout(self.group_switch)
        switch_layout.setContentsMargins(8, 4, 8, 4)
        switch_layout.setSpacing(3)

        self.enable_det_check = QCheckBox(self.group_switch)
        self.enable_det_check.setChecked(UI_DEFAULTS.enable_detection)
        self._bind_help(self.enable_det_check, "enable_detection")
        switch_layout.addWidget(self.enable_det_check)

        self.enable_seg_check = QCheckBox(self.group_switch)
        self.enable_seg_check.setChecked(UI_DEFAULTS.enable_segmentation)
        self.enable_seg_check.toggled.connect(self._on_seg_toggled)
        self._bind_help(self.enable_seg_check, "enable_segmentation")
        switch_layout.addWidget(self.enable_seg_check)

        self.cache_only_check = QCheckBox(self.group_switch)
        self.cache_only_check.setChecked(UI_DEFAULTS.cache_only)
        self._bind_help(self.cache_only_check, "cache_only")
        switch_layout.addWidget(self.cache_only_check)
        self.group_params = QGroupBox(panel)
        self.group_params.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.form_params = QFormLayout(self.group_params)
        self.form_params.setLabelAlignment(Qt.AlignRight)
        self.form_params.setHorizontalSpacing(8)
        self.form_params.setVerticalSpacing(6)
        self.form_params.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.det_conf_spin = QDoubleSpinBox(self.group_params)
        self.det_conf_spin.setRange(0.01, 1.0)
        self.det_conf_spin.setSingleStep(0.01)
        self.det_conf_spin.setValue(UI_DEFAULTS.params.det_conf)
        self._bind_help(self.det_conf_spin, "det_conf")
        self.label_det_conf = QLabel(self._tr("label.det_conf"), self.group_params)
        self.form_params.addRow(self.label_det_conf, self.det_conf_spin)

        self.seg_conf_spin = QDoubleSpinBox(self.group_params)
        self.seg_conf_spin.setRange(0.01, 1.0)
        self.seg_conf_spin.setSingleStep(0.01)
        self.seg_conf_spin.setValue(UI_DEFAULTS.params.seg_conf)
        self._bind_help(self.seg_conf_spin, "seg_conf")
        self.label_seg_conf = QLabel(self._tr("label.seg_conf"), self.group_params)
        self.form_params.addRow(self.label_seg_conf, self.seg_conf_spin)

        self.seg_thr_spin = QDoubleSpinBox(self.group_params)
        self.seg_thr_spin.setRange(0.01, 1.0)
        self.seg_thr_spin.setSingleStep(0.01)
        self.seg_thr_spin.setValue(UI_DEFAULTS.params.seg_thr)
        self._bind_help(self.seg_thr_spin, "seg_thr")
        self.label_seg_thr = QLabel(self._tr("label.seg_thr"), self.group_params)
        self.form_params.addRow(self.label_seg_thr, self.seg_thr_spin)

        self.post_open_spin = QSpinBox(self.group_params)
        self.post_open_spin.setRange(0, 31)
        self.post_open_spin.setValue(UI_DEFAULTS.params.post_open)
        self._bind_help(self.post_open_spin, "post_open")
        self.label_post_open = QLabel(self._tr("label.post_open"), self.group_params)
        self.form_params.addRow(self.label_post_open, self.post_open_spin)

        self.post_close_spin = QSpinBox(self.group_params)
        self.post_close_spin.setRange(0, 31)
        self.post_close_spin.setValue(UI_DEFAULTS.params.post_close)
        self._bind_help(self.post_close_spin, "post_close")
        self.label_post_close = QLabel(self._tr("label.post_close"), self.group_params)
        self.form_params.addRow(self.label_post_close, self.post_close_spin)

        self.post_min_area_spin = QSpinBox(self.group_params)
        self.post_min_area_spin.setRange(0, 100000)
        self.post_min_area_spin.setValue(UI_DEFAULTS.params.post_min_area)
        self._bind_help(self.post_min_area_spin, "post_min_area")
        self.label_post_min_area = QLabel(self._tr("label.post_min_area"), self.group_params)
        self.form_params.addRow(self.label_post_min_area, self.post_min_area_spin)

        self.frame_step_spin = QSpinBox(self.group_params)
        self.frame_step_spin.setRange(1, 999)
        self.frame_step_spin.setValue(UI_DEFAULTS.params.frame_step)
        self._bind_help(self.frame_step_spin, "frame_step")
        self.label_frame_step = QLabel(self._tr("label.frame_step"), self.group_params)
        self.form_params.addRow(self.label_frame_step, self.frame_step_spin)

        self.max_frames_spin = QSpinBox(self.group_params)
        self.max_frames_spin.setRange(0, 999999)
        self.max_frames_spin.setValue(UI_DEFAULTS.params.max_frames)
        self._bind_help(self.max_frames_spin, "max_frames")
        self.label_max_frames = QLabel(self._tr("label.max_frames"), self.group_params)
        self.form_params.addRow(self.label_max_frames, self.max_frames_spin)

        self.preview_interval_spin = QSpinBox(self.group_params)
        self.preview_interval_spin.setRange(1, 120)
        self.preview_interval_spin.setValue(UI_DEFAULTS.params.preview_interval)
        self._bind_help(self.preview_interval_spin, "preview_interval")
        self.label_preview_interval = QLabel(self._tr("label.preview_interval"), self.group_params)
        self.form_params.addRow(self.label_preview_interval, self.preview_interval_spin)

        self.group_actions = QGroupBox(panel)
        self.group_actions.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        actions_layout = QVBoxLayout(self.group_actions)
        actions_layout.setContentsMargins(8, 8, 8, 8)
        actions_layout.setSpacing(6)

        self.start_btn = QPushButton(self.group_actions)
        self.start_btn.clicked.connect(self._start_run)
        self._bind_help(self.start_btn, "start_run")
        actions_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton(self.group_actions)
        self.stop_btn.clicked.connect(self._stop_run)
        self._bind_help(self.stop_btn, "stop_run")
        actions_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton(self.group_actions)
        self.save_btn.clicked.connect(self._save_current_result)
        self._bind_help(self.save_btn, "manual_save")
        actions_layout.addWidget(self.save_btn)

        self.open_output_btn = QPushButton(self.group_actions)
        self.open_output_btn.clicked.connect(self._open_output_dir)
        self._bind_help(self.open_output_btn, "open_output")
        actions_layout.addWidget(self.open_output_btn)

        self.clear_log_btn = QPushButton(self.group_actions)
        self.clear_log_btn.clicked.connect(self._clear_logs)
        self._bind_help(self.clear_log_btn, "clear_log")
        actions_layout.addWidget(self.clear_log_btn)

        layout.addWidget(self.group_input)
        layout.addWidget(self.group_switch)
        layout.addWidget(self.group_params)
        layout.addWidget(self.group_actions)
        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)

        self.preview_mode_label = QLabel(panel)
        top.addWidget(self.preview_mode_label)
        self.preview_combo = QComboBox(panel)
        self.preview_combo.addItem("", "original")
        self.preview_combo.addItem("", "overlay")
        self.preview_combo.addItem("", "mask")
        self.preview_combo.currentIndexChanged.connect(self._on_preview_mode_changed)
        top.addWidget(self.preview_combo)

        self.video_view_combo = QComboBox(panel)
        self.video_view_combo.addItem("", "frame")
        self.video_view_combo.addItem("", "video")
        self.video_view_combo.currentIndexChanged.connect(self._on_video_view_changed)
        self._bind_help(self.video_view_combo, "video_view_mode")
        top.addWidget(self.video_view_combo)

        self.video_source_combo = QComboBox(panel)
        self.video_source_combo.addItem("", "overlay")
        self.video_source_combo.addItem("", "mask")
        self.video_source_combo.currentIndexChanged.connect(self._on_video_source_changed)
        self._bind_help(self.video_source_combo, "video_source")
        top.addWidget(self.video_source_combo)

        self.show_result_check = QCheckBox(panel)
        self.show_result_check.setChecked(True)
        self._bind_help(self.show_result_check, "toggle_result_panel")
        self.show_result_check.setFixedHeight(self.show_result_check.sizeHint().height())
        self.show_result_check.toggled.connect(self._toggle_result_panel)
        top.addWidget(self.show_result_check)

        top.addStretch(1)
        self.preview_info = QLabel(self._tr("preview.info.none"), panel)
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
        self.preview_label.setText(self._tr("preview.label.select_start"))
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
            self.video_widget = QLabel(self._tr("preview.info.qtm_unavailable_video"), video_page)
            self.video_widget.setAlignment(Qt.AlignCenter)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_widget, 1)

        controls = QHBoxLayout()
        self.play_pause_btn = QPushButton(self._tr("button.play"), video_page)
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        controls.addWidget(self.play_pause_btn)

        self.stop_video_btn = QPushButton(self._tr("button.stop"), video_page)
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
        panel = QGroupBox(self._tr("group.result_summary"), self)
        self.result_group = panel
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
        panel = QGroupBox(self._tr("group.log"), self)
        self.log_group = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        self.log_text = QTextEdit(panel)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return panel

    def _init_video_player(self) -> None:
        if not QT_MEDIA_AVAILABLE or QMediaPlayer is None or QAudioOutput is None:
            self._set_video_controls_enabled(False)
            self.preview_info.setText(self._tr("preview.info.qtm_unavailable"))
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

    def _make_path_row(self, select_dir: bool = False):
        edit = QLineEdit()
        edit.setPlaceholderText(
            self._tr("placeholder.select_dir") if select_dir else self._tr("placeholder.select_file")
        )
        btn = QPushButton(self._tr("button.browse"))
        return edit, btn

    @staticmethod
    def _join_row(edit: QLineEdit, btn: QPushButton) -> QWidget:
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(btn, 0)
        return box

    def _bind_help(self, widget: QWidget, key: str) -> None:
        self._help_bindings.append((widget, key))
        self._apply_help(widget, key)

    def _apply_help(self, widget: QWidget, key: str) -> None:
        text = help_text(self.language, key)
        if text:
            widget.setToolTip(text)
            widget.setStatusTip(text)
        else:
            widget.setToolTip("")
            widget.setStatusTip("")

    def _tr(self, key: str, **kwargs) -> str:
        return tr(self.language, key, **kwargs)

    @staticmethod
    def _set_combo_item_text_by_data(combo: QComboBox, value: str, text: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setItemText(idx, text)

    def _on_language_changed(self) -> None:
        if self._updating_language_combo:
            return
        target = normalize_language(str(self.language_combo.currentData() or LANG_ZH))
        if target == self.language:
            return
        self.language = target
        self.settings.setValue("ui/language", self.language)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self._tr("window.title"))
        self.group_input.setTitle(self._tr("group.input_model"))
        self.group_switch.setTitle(self._tr("group.switch"))
        self.group_params.setTitle(self._tr("group.params"))
        self.group_actions.setTitle(self._tr("group.actions"))
        self.result_group.setTitle(self._tr("group.result_summary"))
        self.log_group.setTitle(self._tr("group.log"))

        self._updating_language_combo = True
        self._set_combo_item_text_by_data(self.language_combo, LANG_ZH, self._tr("combo.language.zh"))
        self._set_combo_item_text_by_data(self.language_combo, LANG_EN, self._tr("combo.language.en"))
        lang_index = self.language_combo.findData(self.language)
        if lang_index >= 0:
            self.language_combo.setCurrentIndex(lang_index)
        self._updating_language_combo = False

        self._set_combo_item_text_by_data(self.mode_combo, "image", self._tr("combo.mode.image"))
        self._set_combo_item_text_by_data(self.mode_combo, "video", self._tr("combo.mode.video"))
        self._set_combo_item_text_by_data(self.preview_combo, "original", self._tr("combo.preview.original"))
        self._set_combo_item_text_by_data(self.preview_combo, "overlay", self._tr("combo.preview.overlay"))
        self._set_combo_item_text_by_data(self.preview_combo, "mask", self._tr("combo.preview.mask"))
        self._set_combo_item_text_by_data(self.video_view_combo, "frame", self._tr("combo.video_view.frame"))
        self._set_combo_item_text_by_data(self.video_view_combo, "video", self._tr("combo.video_view.video"))
        self._set_combo_item_text_by_data(
            self.video_source_combo, "overlay", self._tr("combo.video_source.overlay")
        )
        self._set_combo_item_text_by_data(self.video_source_combo, "mask", self._tr("combo.video_source.mask"))
        self._refresh_video_sources()

        self.label_language.setText(self._tr("label.language"))
        self.label_input_mode.setText(self._tr("label.input_mode"))
        self.label_det_model_path.setText(self._tr("label.det_model_path"))
        self.label_seg_model_path.setText(self._tr("label.seg_model_path"))
        self.label_input_file.setText(self._tr("label.input_file"))
        self.label_output_dir.setText(self._tr("label.output_dir"))

        self.label_det_conf.setText(self._tr("label.det_conf"))
        self.label_seg_conf.setText(self._tr("label.seg_conf"))
        self.label_seg_thr.setText(self._tr("label.seg_thr"))
        self.label_post_open.setText(self._tr("label.post_open"))
        self.label_post_close.setText(self._tr("label.post_close"))
        self.label_post_min_area.setText(self._tr("label.post_min_area"))
        self.label_frame_step.setText(self._tr("label.frame_step"))
        self.label_max_frames.setText(self._tr("label.max_frames"))
        self.label_preview_interval.setText(self._tr("label.preview_interval"))

        self.det_model_edit.setPlaceholderText(self._tr("placeholder.select_file"))
        self.seg_model_edit.setPlaceholderText(self._tr("placeholder.select_file"))
        self.input_path_edit.setPlaceholderText(self._tr("placeholder.select_file"))
        self.output_dir_edit.setPlaceholderText(self._tr("placeholder.select_dir"))

        self.det_model_btn.setText(self._tr("button.browse"))
        self.seg_model_btn.setText(self._tr("button.browse"))
        self.input_path_btn.setText(self._tr("button.browse"))
        self.output_dir_btn.setText(self._tr("button.browse"))

        self.enable_det_check.setText(self._tr("checkbox.enable_det"))
        self.enable_seg_check.setText(self._tr("checkbox.enable_seg"))
        self.cache_only_check.setText(self._tr("checkbox.cache_only"))
        self.start_btn.setText(self._tr("button.start"))
        self.stop_btn.setText(self._tr("button.stop"))
        self.save_btn.setText(self._tr("button.manual_save"))
        self.open_output_btn.setText(self._tr("button.open_output"))
        self.clear_log_btn.setText(self._tr("button.clear_log"))
        self.preview_mode_label.setText(self._tr("label.preview_mode"))
        self.show_result_check.setText(self._tr("checkbox.show_result"))
        self.stop_video_btn.setText(self._tr("button.stop"))
        if self.media_player is not None and self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.play_pause_btn.setText(self._tr("button.pause"))
        else:
            self.play_pause_btn.setText(self._tr("button.play"))
        if isinstance(self.video_widget, QLabel):
            self.video_widget.setText(self._tr("preview.info.qtm_unavailable_video"))

        for widget, key in self._help_bindings:
            self._apply_help(widget, key)

        if self._is_running:
            self.preview_info.setText(self._tr("preview.info.running"))
            self.preview_label.setText(self._tr("preview.label.running"))
        elif self.current_frame is None and self.preview_label.pixmap() is None:
            self.preview_info.setText(self._tr("preview.info.none"))
            self.preview_label.setText(self._tr("preview.label.select_start"))
        elif self.current_frame is not None and self.current_frame.mode == "video":
            self.preview_info.setText(self._tr("preview.info.current_frame", index=self.current_frame.frame_index))
        elif self.current_frame is not None:
            self.preview_info.setText(self._tr("preview.info.image_done"))

        self._on_mode_changed()
        self._update_result_summary()

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
            candidates = " | ".join(str(p) for p in DEFAULT_MODELS_YAML_CANDIDATES)
            self._startup_notices.append(
                self._tr("startup.default_models_yaml_missing", candidates=candidates)
            )
            return

        try:
            cfg = yaml.safe_load(DEFAULT_MODELS_YAML.read_text(encoding="utf-8")) or {}
            if not isinstance(cfg, dict):
                raise ValueError(self._tr("startup.default_models_yaml_top_invalid"))
        except Exception as exc:
            self._startup_notices.append(self._tr("startup.default_models_read_failed", error=exc))
            return

        for key, edit, label_key in (
            ("det_model", self.det_model_edit, "startup.default_label.det_model"),
            ("seg_model", self.seg_model_edit, "startup.default_label.seg_model"),
        ):
            raw = str(cfg.get(key, "") or "").strip()
            if not raw:
                continue
            path = resolve_configured_path(raw, base_dir=DEFAULT_MODELS_YAML.parent)
            if path is not None and path.exists() and path.is_file():
                edit.setText(str(path))
            else:
                self._startup_notices.append(
                    self._tr("startup.default_model_missing", label=self._tr(label_key), raw=raw)
                )

        out_raw = str(cfg.get("default_output_dir", "") or "").strip()
        out_path = (
            resolve_configured_path(out_raw, base_dir=DEFAULT_MODELS_YAML.parent)
            if out_raw
            else OUTPUTS_DIR
        )
        if out_path is None:
            out_path = OUTPUTS_DIR
        try:
            out_path.mkdir(parents=True, exist_ok=True)
            self.output_dir_edit.setText(str(out_path))
        except Exception as exc:
            self.output_dir_edit.setText(str(OUTPUTS_DIR))
            self._startup_notices.append(self._tr("startup.default_output_fallback", error=exc))

    def _collect_runtime_startup_notices(self) -> None:
        # 轻量静态检查放在这里；重依赖自检放到异步线程，避免阻塞主界面启动。
        return

    def _show_startup_notices(self) -> None:
        if not self._startup_notices:
            return

        for item in self._startup_notices:
            self._append_log(f"[STARTUP][WARN] {item}")

        lines = [f"{idx + 1}. {item}" for idx, item in enumerate(self._startup_notices)]
        QMessageBox.warning(
            self,
            self._tr("dialog.startup_check.title"),
            self._tr("dialog.startup_check.body", items="\n\n".join(lines)),
        )

    def _launch_startup_self_check(self) -> None:
        self._startup_check_result = None
        self._startup_check_started_at = perf_counter()

        def _worker() -> None:
            self._startup_check_result = self.engine.startup_self_check(timeout_seconds=10.0)

        Thread(target=_worker, daemon=True).start()
        self._startup_check_timer = QTimer(self)
        self._startup_check_timer.setInterval(200)
        self._startup_check_timer.timeout.connect(self._poll_startup_self_check)
        self._startup_check_timer.start()

    def _poll_startup_self_check(self) -> None:
        if self._startup_check_result is not None:
            if self._startup_check_timer is not None:
                self._startup_check_timer.stop()
            ok, detail = self._startup_check_result
            if ok:
                self._append_log(f"[STARTUP] {detail}")
            else:
                self._append_log(f"[STARTUP][WARN] {self._tr('startup.dependencies_not_ready', detail=detail)}")
            return

        if perf_counter() - self._startup_check_started_at > 15.0:
            if self._startup_check_timer is not None:
                self._startup_check_timer.stop()
            self._append_log(f"[STARTUP][WARN] {self._tr('startup.self_check_watchdog_timeout')}")

    def _browse_input_by_mode(self) -> None:
        mode = self.mode_combo.currentData()
        if mode == "video":
            self._browse_file(
                self.input_path_edit,
                "dialog.select_video",
                self._tr("model.video_filter"),
                "last_input_dir",
                "input_path",
            )
        else:
            self._browse_file(
                self.input_path_edit,
                "dialog.select_image",
                self._tr("model.image_filter"),
                "last_input_dir",
                "input_path",
            )

    def _browse_file(
        self,
        edit: QLineEdit,
        title_key: str,
        pattern: str,
        dir_key: str,
        value_key: str,
    ) -> None:
        start_dir = self._dialog_start_dir(edit, dir_key)
        f, _ = QFileDialog.getOpenFileName(self, self._tr(title_key), start_dir, pattern)
        if f:
            target = Path(f)
            edit.setText(str(target))
            self._set_path_history(dir_key, target)
            self._set_path_value(value_key, target)

    def _browse_dir(self, edit: QLineEdit, title_key: str, dir_key: str, value_key: str) -> None:
        start_dir = self._dialog_start_dir(edit, dir_key)
        d = QFileDialog.getExistingDirectory(self, self._tr(title_key), start_dir)
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
            self.statusBar().showMessage(self._tr("status.mode_video"))
        else:
            self.video_view_combo.setCurrentIndex(0)
            self.preview_stack.setCurrentIndex(0)
            self._release_video_playback_handles()
            self.statusBar().showMessage(self._tr("status.mode_image"))

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
        self.play_pause_btn.setText(self._tr("button.play"))
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
        self.video_source_combo.setItemText(
            1,
            self._tr("combo.video_source.mask") if has_mask else self._tr("combo.video_source.mask_unavailable"),
        )
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
            self.preview_info.setText(self._tr("preview.info.qtm_unavailable"))
            return

        source = self.video_source_combo.currentData()
        target = self.cached_overlay_video if source == "overlay" else self.cached_mask_video

        if not target or not Path(target).exists():
            self._set_video_controls_enabled(False)
            self.preview_info.setText(self._tr("preview.info.cache_unavailable"))
            self._release_video_playback_handles()
            self._update_result_summary()
            return

        if self.current_play_video_path != target:
            self._release_video_playback_handles()
            self.current_play_video_path = target
            self.media_player.setSource(QUrl.fromLocalFile(target))
            self.media_player.pause()
        self._set_video_controls_enabled(True)
        self.preview_info.setText(self._tr("preview.info.cache_loaded", name=Path(target).name))
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
        self.play_pause_btn.setText(
            self._tr("button.pause") if state == QMediaPlayer.PlayingState else self._tr("button.play")
        )

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
            return False, self._tr("validation.invalid_mode")

        input_text = self.input_path_edit.text().strip()
        if not input_text:
            return False, self._tr("validation.input_required")
        input_path = Path(input_text)
        if not input_path.exists():
            return False, self._tr("validation.input_not_exist")
        if not input_path.is_file():
            return False, self._tr("validation.input_not_file")

        output_text = self.output_dir_edit.text().strip()
        if not output_text:
            return False, self._tr("validation.output_required")
        output_dir = Path(output_text)
        if output_dir.exists() and not output_dir.is_dir():
            return False, self._tr("validation.output_not_dir")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, self._tr("validation.output_unavailable", error=exc)

        if self.enable_seg_check.isChecked() and not self.enable_det_check.isChecked():
            return False, self._tr("validation.seg_requires_det")

        if not self.enable_det_check.isChecked() and not self.enable_seg_check.isChecked():
            return False, self._tr("validation.need_one_task")

        if self.enable_det_check.isChecked():
            det_text = self.det_model_edit.text().strip()
            if not det_text:
                return False, self._tr("validation.det_model_required")
            det_model_path = Path(det_text)
            if not det_model_path.exists() or not det_model_path.is_file():
                return False, self._tr("validation.det_model_invalid")

        if self.enable_seg_check.isChecked():
            seg_text = self.seg_model_edit.text().strip()
            if not seg_text:
                return False, self._tr("validation.seg_model_required")
            seg_model_path = Path(seg_text)
            if not seg_model_path.exists() or not seg_model_path.is_file():
                return False, self._tr("validation.seg_model_invalid")

        dep_ok, dep_msg = self.engine.check_dependencies()
        if not dep_ok:
            return (
                False,
                self._tr("validation.dependency_not_ready", detail=dep_msg),
            )

        return True, ""

    def _collect_request(self) -> RunRequest | None:
        ok, message = self._validate_request_inputs()
        if not ok:
            QMessageBox.warning(self, self._tr("dialog.cannot_start_run"), message)
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
        self.preview_info.setText(self._tr("preview.info.running"))
        self.preview_label.setText(self._tr("preview.label.running"))
        self._append_log(
            f"[RUN] start mode={request.mode}, input={request.input_path.name}, cache_only={request.cache_only}"
        )
        self._append_log(
            self._tr("message.current_device_log", device=self.engine.current_device_display())
        )
        if self.engine.runtime_device_note:
            self._append_log(self._tr("message.device_note_log", note=self.engine.runtime_device_note))

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
            self.preview_info.setText(self._tr("preview.info.current_frame", index=frame.frame_index))
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
            self.preview_info.setText(self._tr("preview.info.image_done"))

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
                self._append_log(self._tr("message.cache_snapshot_saved", path=cache_dir))

        if summary.mode == "video" and self.video_view_combo.currentData() == "video":
            self._load_selected_cached_video()

        self.progress_bar.setValue(100)
        self._append_log(
            f"[DONE] {summary.message} | processed={summary.processed_frames}/{summary.total_frames} | "
            f"elapsed={summary.elapsed_seconds:.2f}s"
        )
        if summary.current_device:
            self._append_log(self._tr("message.current_device_log", device=summary.current_device))
        if summary.device_note:
            self._append_log(self._tr("message.device_note_log", note=summary.device_note))
        if summary.cache_overlay_video:
            self._append_log(f"[CACHE][VIDEO] overlay: {summary.cache_overlay_video}")
        if summary.cache_mask_video:
            self._append_log(f"[CACHE][VIDEO] mask: {summary.cache_mask_video}")

        self._refresh_preview()
        self._update_result_summary()
        self._set_running(False)

    def _on_failed(self, message: str) -> None:
        self._append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, self._tr("dialog.inference_failed"), message)
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
            QMessageBox.information(self, self._tr("dialog.info"), self._tr("message.no_result_to_save"))
            return

        try:
            t0 = perf_counter()
            run_dir = self.writer.save_current_result(self.current_request, self.current_summary)
            self.last_write_ms = (perf_counter() - t0) * 1000.0
            self.last_write_action = "manual_save"
            self._append_log(self._tr("message.current_result_saved_log", path=run_dir))
            self.statusBar().showMessage(self._tr("status.saved_current_result"), 3000)
            self._update_result_summary()
        except Exception as exc:
            QMessageBox.critical(self, self._tr("dialog.save_failed"), str(exc))

    def _open_output_dir(self) -> None:
        out_dir = Path(self.output_dir_edit.text().strip() or str(OUTPUTS_DIR))
        out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    def _clear_logs(self) -> None:
        self.log_text.clear()

    def _append_log(self, message: str) -> None:
        if message == "后台任务启动":
            message = self._tr("message.worker_started")
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
                self.preview_label.setText(self._tr("preview.label.no_mask"))
                return
            arr = self.current_frame.mask_u8
        else:
            arr = self.current_frame.overlay_bgr

        pixmap = self._to_pixmap(arr)
        if pixmap.isNull():
            self.preview_label.setText(self._tr("preview.label.convert_failed"))
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
            self.result_text.setPlainText(self._tr("summary.empty"))
            return

        lines: list[str] = []

        if self.current_request is not None:
            lines.append(self._tr("summary.input_file", name=self.current_request.input_path.name))
            lines.append(
                self._tr(
                    "summary.mode",
                    mode=self._tr(
                        "summary.mode.video"
                        if self.current_request.mode == "video"
                        else "summary.mode.image"
                    ),
                )
            )
        if self.current_summary is not None and self.current_summary.current_device:
            lines.append(self._tr("summary.current_device", device=self.current_summary.current_device))
            if self.current_summary.device_note:
                lines.append(self._tr("summary.device_note", note=self.current_summary.device_note))

        if self.current_frame is not None:
            if self.current_frame.mode == "video":
                lines.append(self._tr("summary.current_frame", index=self.current_frame.frame_index))
            lines.append(self._tr("summary.det_count", count=self.current_frame.det_count))
            lines.append(self._tr("summary.det_conf_max", value=self.current_frame.det_conf_max))
            lines.append(self._tr("summary.det_conf_mean", value=self.current_frame.det_conf_mean))
            lines.append(
                self._tr(
                    "summary.seg_enabled",
                    value=self._tr("summary.yes" if self.current_frame.segmentation_enabled else "summary.no"),
                )
            )

            total_proc = (
                self.current_frame.model_infer_ms
                + self.current_frame.postprocess_ms
                + self.current_frame.viz_ms
            )
            lines.append(self._tr("summary.current_proc", value=total_proc))
            lines.append(self._tr("summary.est_fps", value=self.current_frame.est_fps))

        preview_map = {
            "original": self._tr("combo.preview.original"),
            "overlay": self._tr("combo.preview.overlay"),
            "mask": self._tr("combo.preview.mask"),
        }
        lines.append(
            self._tr(
                "summary.preview_mode",
                mode=preview_map.get(self.preview_combo.currentData(), self._tr("combo.preview.original")),
            )
        )

        lines.append("")
        lines.append(self._tr("summary.section_perf"))
        if self.current_frame is not None:
            lines.append(self._tr("summary.perf_model", value=self.current_frame.model_infer_ms))
            lines.append(self._tr("summary.perf_post", value=self.current_frame.postprocess_ms))
            lines.append(self._tr("summary.perf_viz", value=self.current_frame.viz_ms))
        lines.append(self._tr("summary.perf_ui_convert", value=self.last_ui_convert_ms))
        lines.append(self._tr("summary.perf_ui_refresh", value=self.last_ui_refresh_ms))
        lines.append(self._tr("summary.perf_write", action=self.last_write_action, value=self.last_write_ms))

        if self.current_summary is not None and self.current_summary.mode == "video":
            lines.append("")
            lines.append(self._tr("summary.section_video_cache"))
            lines.append(
                self._tr(
                    "summary.overlay_cache",
                    value=self.current_summary.cache_overlay_video or self._tr("summary.not_generated"),
                )
            )
            lines.append(
                self._tr(
                    "summary.mask_cache",
                    value=self.current_summary.cache_mask_video or self._tr("summary.not_generated"),
                )
            )
            lines.append(
                self._tr(
                    "summary.playable",
                    value=self._tr("summary.playable_yes" if self.current_summary.cache_video_playable else "summary.playable_no"),
                )
            )

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
        return self._tr("cleanup.remove_failed", name=name, retries=retries, error=last_exc)

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
                self._tr("dialog.exit_running"),
                self._tr("exit.running_prompt"),
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
                QMessageBox.warning(self, self._tr("dialog.exit_failed"), self._tr("exit.thread_not_stopped"))
                event.ignore()
                return

        box = QMessageBox(self)
        box.setWindowTitle(self._tr("dialog.exit_confirm"))
        box.setIcon(QMessageBox.Question)
        box.setText(self._tr("exit.confirm_text"))
        box.setInformativeText(self._tr("exit.confirm_info"))

        delete_btn = box.addButton(self._tr("exit.btn.delete"), QMessageBox.AcceptRole)
        keep_btn = box.addButton(self._tr("exit.btn.keep"), QMessageBox.DestructiveRole)
        cancel_btn = box.addButton(self._tr("exit.btn.cancel"), QMessageBox.RejectRole)
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
                    self._tr("dialog.cleanup_failed"),
                    self._tr("exit.cleanup_failed_hint", errors="\n".join(errors)),
                )

        self._persist_current_paths()
        self._release_media_before_cleanup()
        if self._startup_check_timer is not None:
            self._startup_check_timer.stop()
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.preview_stack.currentIndex() == 0:
            self._refresh_preview()

    def append_startup_hint(self, text: str) -> None:
        self._append_log(text)
