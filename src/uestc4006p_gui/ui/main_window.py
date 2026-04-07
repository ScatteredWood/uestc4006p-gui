from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import cv2
import yaml
from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
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
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.paths import DEFAULT_MODELS_YAML, LOGS_DIR, OUTPUTS_DIR, ensure_runtime_dirs
from ..core.schemas import FrameResult, InferenceParams, RunRequest, RunSummary
from ..core.settings import HELP_TEXTS, UI_DEFAULTS
from ..inference.cascade_engine import CascadeEngine
from ..inference.result_writer import ResultWriter
from .worker import InferenceWorker


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

        self.log_file = LOGS_DIR / "gui.log"
        self.setWindowTitle("UESTC4006P GUI MVP - 演示版")
        self.resize(1580, 900)

        self._setup_ui()
        self._load_defaults_from_yaml()
        self._on_mode_changed()
        self._set_running(False)

    def _setup_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        root_layout.addWidget(left_panel, 3)
        root_layout.addWidget(center_panel, 5)
        root_layout.addWidget(right_panel, 3)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(260)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("就绪")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        group_input = QGroupBox("输入与模型", panel)
        form_input = QFormLayout(group_input)

        self.mode_combo = QComboBox(group_input)
        self.mode_combo.addItem("图片", "image")
        self.mode_combo.addItem("视频", "video")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._apply_help(self.mode_combo, "input_mode")
        form_input.addRow("输入模式", self.mode_combo)

        self.det_model_edit, det_btn = self._make_path_row(select_dir=False)
        self._apply_help(self.det_model_edit, "det_model_path")
        self._apply_help(det_btn, "det_model_path")
        det_btn.clicked.connect(lambda: self._browse_file(self.det_model_edit, "选择检测模型", "PyTorch Weights (*.pt)"))
        form_input.addRow("检测模型", self._join_row(self.det_model_edit, det_btn))

        self.seg_model_edit, seg_btn = self._make_path_row(select_dir=False)
        self._apply_help(self.seg_model_edit, "seg_model_path")
        self._apply_help(seg_btn, "seg_model_path")
        seg_btn.clicked.connect(lambda: self._browse_file(self.seg_model_edit, "选择分割模型", "PyTorch Weights (*.pt)"))
        form_input.addRow("分割模型", self._join_row(self.seg_model_edit, seg_btn))

        self.input_path_edit, input_btn = self._make_path_row(select_dir=False)
        input_btn.clicked.connect(self._browse_input_by_mode)
        self._apply_help(self.input_path_edit, "input_path")
        self._apply_help(input_btn, "input_path")
        form_input.addRow("输入文件", self._join_row(self.input_path_edit, input_btn))

        self.output_dir_edit, output_btn = self._make_path_row(select_dir=True)
        output_btn.clicked.connect(lambda: self._browse_dir(self.output_dir_edit, "选择输出目录"))
        self._apply_help(self.output_dir_edit, "output_dir")
        self._apply_help(output_btn, "output_dir")
        form_input.addRow("输出目录", self._join_row(self.output_dir_edit, output_btn))

        group_switch = QGroupBox("开关", panel)
        switch_layout = QVBoxLayout(group_switch)

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
        form_params = QFormLayout(group_params)

        self.det_conf_spin = QDoubleSpinBox(group_params)
        self.det_conf_spin.setRange(0.01, 1.0)
        self.det_conf_spin.setSingleStep(0.01)
        self.det_conf_spin.setValue(UI_DEFAULTS.params.det_conf)
        self._apply_help(self.det_conf_spin, "det_conf")
        form_params.addRow("det_conf", self.det_conf_spin)

        self.seg_conf_spin = QDoubleSpinBox(group_params)
        self.seg_conf_spin.setRange(0.01, 1.0)
        self.seg_conf_spin.setSingleStep(0.01)
        self.seg_conf_spin.setValue(UI_DEFAULTS.params.seg_conf)
        self._apply_help(self.seg_conf_spin, "seg_conf")
        form_params.addRow("seg_conf", self.seg_conf_spin)

        self.seg_thr_spin = QDoubleSpinBox(group_params)
        self.seg_thr_spin.setRange(0.01, 1.0)
        self.seg_thr_spin.setSingleStep(0.01)
        self.seg_thr_spin.setValue(UI_DEFAULTS.params.seg_thr)
        self._apply_help(self.seg_thr_spin, "seg_thr")
        form_params.addRow("seg_thr", self.seg_thr_spin)

        self.post_open_spin = QSpinBox(group_params)
        self.post_open_spin.setRange(0, 31)
        self.post_open_spin.setValue(UI_DEFAULTS.params.post_open)
        self._apply_help(self.post_open_spin, "post_open")
        form_params.addRow("post_open", self.post_open_spin)

        self.post_close_spin = QSpinBox(group_params)
        self.post_close_spin.setRange(0, 31)
        self.post_close_spin.setValue(UI_DEFAULTS.params.post_close)
        self._apply_help(self.post_close_spin, "post_close")
        form_params.addRow("post_close", self.post_close_spin)

        self.post_min_area_spin = QSpinBox(group_params)
        self.post_min_area_spin.setRange(0, 10000)
        self.post_min_area_spin.setValue(UI_DEFAULTS.params.post_min_area)
        self._apply_help(self.post_min_area_spin, "post_min_area")
        form_params.addRow("post_min_area", self.post_min_area_spin)

        self.frame_step_spin = QSpinBox(group_params)
        self.frame_step_spin.setRange(1, 999)
        self.frame_step_spin.setValue(UI_DEFAULTS.params.frame_step)
        self._apply_help(self.frame_step_spin, "frame_step")
        form_params.addRow("frame_step", self.frame_step_spin)

        self.max_frames_spin = QSpinBox(group_params)
        self.max_frames_spin.setRange(0, 999999)
        self.max_frames_spin.setValue(UI_DEFAULTS.params.max_frames)
        self._apply_help(self.max_frames_spin, "max_frames")
        form_params.addRow("max_frames", self.max_frames_spin)

        group_actions = QGroupBox("操作", panel)
        actions_layout = QVBoxLayout(group_actions)

        self.start_btn = QPushButton("开始推理", group_actions)
        self.start_btn.clicked.connect(self._start_run)
        self._apply_help(self.start_btn, "start_run")
        actions_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止", group_actions)
        self.stop_btn.clicked.connect(self._stop_run)
        self._apply_help(self.stop_btn, "stop_run")
        actions_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("手动保存当前结果", group_actions)
        self._apply_help(self.save_btn, "manual_save")
        self.save_btn.clicked.connect(self._save_current_result)
        actions_layout.addWidget(self.save_btn)

        self.open_output_btn = QPushButton("打开输出目录", group_actions)
        self.open_output_btn.clicked.connect(self._open_output_dir)
        self._apply_help(self.open_output_btn, "open_output")
        actions_layout.addWidget(self.open_output_btn)

        self.clear_log_btn = QPushButton("清空日志", group_actions)
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
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

        top = QHBoxLayout()
        top.addWidget(QLabel("预览视图", panel))

        self.preview_combo = QComboBox(panel)
        self.preview_combo.addItem("原图", "original")
        self.preview_combo.addItem("overlay", "overlay")
        self.preview_combo.addItem("mask", "mask")
        self.preview_combo.currentIndexChanged.connect(self._refresh_preview)
        top.addWidget(self.preview_combo)
        top.addStretch(1)

        self.preview_info = QLabel("当前无预览结果", panel)
        top.addWidget(self.preview_info)

        self.preview_label = QLabel(panel)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("请先选择输入并点击“开始推理”")

        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.preview_label)

        layout.addLayout(top)
        layout.addWidget(scroll)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        title = QLabel("日志", panel)
        layout.addWidget(title)

        self.log_text = QTextEdit(panel)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return panel

    @staticmethod
    def _make_path_row(select_dir: bool = False):
        edit = QLineEdit()
        edit.setPlaceholderText("请选择目录" if select_dir else "请选择文件")
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

    def _load_defaults_from_yaml(self) -> None:
        if not DEFAULT_MODELS_YAML.exists():
            self.output_dir_edit.setText(str(OUTPUTS_DIR))
            return
        try:
            cfg = yaml.safe_load(DEFAULT_MODELS_YAML.read_text(encoding="utf-8")) or {}
            self.det_model_edit.setText(str(cfg.get("det_model", "")))
            self.seg_model_edit.setText(str(cfg.get("seg_model", "")))
            self.output_dir_edit.setText(str(cfg.get("default_output_dir", OUTPUTS_DIR)))
        except Exception as exc:
            self._append_log(f"[WARN] 读取 default_models.yaml 失败: {exc}")
            self.output_dir_edit.setText(str(OUTPUTS_DIR))

    def _browse_input_by_mode(self) -> None:
        mode = self.mode_combo.currentData()
        if mode == "video":
            self._browse_file(self.input_path_edit, "选择视频", "Video Files (*.mp4 *.avi *.mov)")
        else:
            self._browse_file(self.input_path_edit, "选择图片", "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")

    def _browse_file(self, edit: QLineEdit, title: str, pattern: str) -> None:
        f, _ = QFileDialog.getOpenFileName(self, title, str(Path.cwd()), pattern)
        if f:
            edit.setText(f)

    def _browse_dir(self, edit: QLineEdit, title: str) -> None:
        d = QFileDialog.getExistingDirectory(self, title, str(Path.cwd()))
        if d:
            edit.setText(d)

    def _on_mode_changed(self) -> None:
        is_video = self.mode_combo.currentData() == "video"
        self.frame_step_spin.setEnabled(is_video)
        self.max_frames_spin.setEnabled(is_video)
        if is_video:
            self.statusBar().showMessage("当前模式：视频。可使用 frame_step 和 max_frames 控制演示速度。")
        else:
            self.statusBar().showMessage("当前模式：图片。将进行单图推理与预览。")

    def _on_seg_toggled(self, enabled: bool) -> None:
        self.seg_model_edit.setEnabled(enabled)
        self.seg_conf_spin.setEnabled(enabled)
        self.seg_thr_spin.setEnabled(enabled)
        self.post_open_spin.setEnabled(enabled)
        self.post_close_spin.setEnabled(enabled)
        self.post_min_area_spin.setEnabled(enabled)

    def _collect_request(self) -> RunRequest | None:
        mode = self.mode_combo.currentData()
        input_path = Path(self.input_path_edit.text().strip())
        output_dir = Path(self.output_dir_edit.text().strip() or str(OUTPUTS_DIR))
        det_model_path = Path(self.det_model_edit.text().strip())
        seg_model_path = Path(self.seg_model_edit.text().strip())

        if not input_path.exists():
            QMessageBox.warning(self, "输入错误", "输入路径不存在，请重新选择。")
            return None
        if self.enable_det_check.isChecked() and not det_model_path.exists():
            QMessageBox.warning(self, "模型错误", "检测模型路径不存在，请检查。")
            return None
        if self.enable_seg_check.isChecked() and not seg_model_path.exists():
            QMessageBox.warning(self, "模型错误", "分割模型路径不存在，请检查。")
            return None

        if self.enable_seg_check.isChecked() and not self.enable_det_check.isChecked():
            self._append_log("[WARN] 检测关闭时无法执行级联分割，已自动关闭分割。")

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
            self._append_log("[INFO] 当前任务仍在运行中。")
            return

        request = self._collect_request()
        if request is None:
            return

        self.current_request = request
        self.current_summary = None
        self.current_frame = None
        self.preview_label.setText("推理中，请稍候...")
        self.preview_label.setPixmap(QPixmap())
        self.preview_info.setText("准备启动")

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
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._worker.deleteLater)

        self._set_running(True)
        self._append_log(f"[RUN] 启动任务，模式={request.mode}, 输入={request.input_path}")
        self.progress_bar.setRange(0, 0)
        self._thread.start()

    def _stop_run(self) -> None:
        if not self._is_running or self._worker is None:
            return
        self._append_log("[INFO] 已请求停止，等待当前步骤安全结束...")
        self._worker.request_stop()

    def _on_progress(self, cur: int, total: int) -> None:
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(cur)
        self.statusBar().showMessage(f"视频进度: {cur}/{total}")

    def _on_frame(self, frame_result_obj: object) -> None:
        frame_result = frame_result_obj  # Qt 信号统一 object，运行时是 FrameResult
        self.current_frame = frame_result
        self.preview_info.setText(f"当前帧: {frame_result.frame_index}")
        self._refresh_preview()

    def _on_finished(self, summary_obj: object) -> None:
        summary = summary_obj  # Qt 信号统一 object，运行时是 RunSummary
        self.current_summary = summary
        self._append_log(
            f"[DONE] {summary.message} | processed={summary.processed_frames}, "
            f"elapsed={summary.elapsed_seconds:.2f}s"
        )

        if self.current_request is not None:
            try:
                if self.current_request.cache_only:
                    cache_dir = self.writer.cache_run(self.current_request, summary)
                    if cache_dir is not None:
                        self._append_log(f"[CACHE] 结果已缓存到: {cache_dir}")
                else:
                    out_dir = self.writer.save_current_result(self.current_request, summary)
                    self._append_log(f"[SAVE] 已自动保存到: {out_dir}")
            except Exception as exc:
                self._append_log(f"[WARN] 自动写出失败: {exc}")

        self.statusBar().showMessage(summary.message)
        if self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(self.progress_bar.maximum())

    def _on_failed(self, err_msg: str) -> None:
        self._append_log(f"[ERROR] {err_msg}")
        self.statusBar().showMessage("任务失败")
        QMessageBox.critical(self, "推理失败", err_msg)

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_running(False)
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.save_btn.setEnabled(not running)

    def _save_current_result(self) -> None:
        if self.current_request is None or self.current_summary is None:
            QMessageBox.information(self, "提示", "当前没有可保存结果，请先运行一次推理。")
            return
        try:
            out_dir = self.writer.save_current_result(self.current_request, self.current_summary)
            self._append_log(f"[SAVE] 手动保存成功: {out_dir}")
            self.statusBar().showMessage(f"已保存到: {out_dir}")
        except Exception as exc:
            self._append_log(f"[ERROR] 手动保存失败: {exc}")
            QMessageBox.critical(self, "保存失败", str(exc))

    def _open_output_dir(self) -> None:
        target = Path(self.output_dir_edit.text().strip() or str(OUTPUTS_DIR))
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _append_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self.log_text.append(line)
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + os.linesep)
        except Exception:
            pass

    def _refresh_preview(self) -> None:
        if self.current_frame is None:
            return

        mode = self.preview_combo.currentData()
        title = "预览"
        img = None
        is_mask = False

        if mode == "original":
            img = self.current_frame.original_bgr
            title = "原图"
        elif mode == "overlay":
            img = self.current_frame.overlay_bgr
            title = "overlay"
        elif mode == "mask":
            if self.current_frame.mask_u8 is None:
                self.preview_label.setText("当前没有 mask（可能未启用分割）")
                self.preview_label.setPixmap(QPixmap())
                self.preview_info.setText("mask 不可用")
                return
            img = self.current_frame.mask_u8
            is_mask = True
            title = "mask"

        if img is None:
            return

        pix = self._to_pixmap(img, is_mask=is_mask)
        # 按当前预览区域缩放，保持比例，便于演示。
        target_w = max(200, self.preview_label.width() - 12)
        target_h = max(200, self.preview_label.height() - 12)
        pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)
        self.preview_label.setText("")
        self.preview_info.setText(f"{title} | 帧 {self.current_frame.frame_index}")

    @staticmethod
    def _to_pixmap(img, is_mask: bool = False) -> QPixmap:
        if is_mask:
            if len(img.shape) == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_preview()

    def append_startup_hint(self, hint: str) -> None:
        self._append_log(hint)
