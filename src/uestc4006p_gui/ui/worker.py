from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from ..core.schemas import RunRequest
from ..inference.cascade_engine import CascadeEngine


class InferenceWorker(QObject):
    """后台推理 worker，运行在 QThread 中。"""

    sig_log = Signal(str)
    sig_progress = Signal(int, int)
    sig_frame = Signal(object)  # FrameResult
    sig_finished = Signal(object)  # RunSummary
    sig_failed = Signal(str)

    def __init__(self, engine: CascadeEngine, request: RunRequest) -> None:
        super().__init__()
        self.engine = engine
        self.request = request
        self._stop_event = Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    @Slot()
    def run(self) -> None:
        try:
            self.sig_log.emit("后台任务启动")
            if self.request.mode == "image":
                summary = self.engine.run_image(
                    self.request,
                    log=self.sig_log.emit,
                )
                if summary.last_frame_result is not None:
                    self.sig_frame.emit(summary.last_frame_result)
            else:
                summary = self.engine.run_video(
                    self.request,
                    on_progress=self.sig_progress.emit,
                    on_frame=self.sig_frame.emit,
                    should_stop=self._stop_event.is_set,
                    log=self.sig_log.emit,
                )
            self.sig_finished.emit(summary)
        except Exception as exc:  # pragma: no cover
            self.sig_failed.emit(str(exc))