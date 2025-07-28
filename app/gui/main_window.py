from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox, QCheckBox, QGroupBox,
    QSizePolicy
)

from app.audio.capture import record_block, TARGET_SR
from app.asr.transcribe import ASREngine, ASRConfig
from app.asr.translate import translate_to_de
from app.io.autosave import AutoSaver

import sounddevice as sd

AUTOSAVE_INTERVAL_SEC_DEFAULT = 15 * 60

@dataclass
class DeviceInfo:
    index: int
    name: str
    api: str
    max_in: int
    max_out: int

    def display(self) -> str:
        return f"{self.index}: {self.name} | {self.api} | in={self.max_in} out={self.max_out}"

def list_devices() -> List[DeviceInfo]:
    out: List[DeviceInfo] = []
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        api_name = apis[d["hostapi"]]["name"]
        out.append(DeviceInfo(
            index=i,
            name=d["name"],
            api=api_name,
            max_in=d["max_input_channels"],
            max_out=d["max_output_channels"],
        ))
    return out

class MicWorker(QThread):
    line_ready = Signal(str)
    status_text = Signal(str)

    def __init__(self, mic_index: int, translate_flag: Callable[[], bool], save_root: Path, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.mic_index = mic_index
        self.translate_flag = translate_flag
        self._running = False
        self._paused = False
        self.asr = ASREngine(ASRConfig())
        self.saver = AutoSaver(Path(save_root), interval_minutes=15)

    def start_run(self):
        self._running = True
        self._paused = False
        if not self.isRunning():
            self.start()

    def pause(self):
        self._paused = True
        self.status_text.emit("Paused.")

    def resume(self):
        self._paused = False
        self.status_text.emit("Resumed.")

    def stop_run(self):
        self._running = False

    def run(self):
        self.status_text.emit("Listening…")
        try:
            while self._running:
                if self._paused:
                    self.msleep(50)
                    continue
                audio16 = record_block(self.mic_index, seconds=2.0).squeeze(-1)
                text, lang, _ = self.asr.transcribe(audio16, TARGET_SR)
                if not text:
                    continue
                lang_up = (lang or "").upper()
                line = f"[{time.strftime('%H:%M:%S')}] Speaker 1 (You) ({lang_up}): {text}"
                tr = ""
                if self.translate_flag() and lang in ("pl", "sk"):
                    try:
                        tr = translate_to_de(text, lang)
                    except Exception:
                        tr = ""
                if tr:
                    line += f"\n               → DE: {tr}"
                self.line_ready.emit(line)
                self.saver.write(line)
        finally:
            self.saver.close()
            self.status_text.emit("Stopped.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multilingual Live Transcription — MVP GUI")
        self.resize(1000, 700)
        self.devices: List[DeviceInfo] = list_devices()
        self.mic_combo = QComboBox()
        self.sys_combo = QComboBox()
        self._populate_device_combos()
        self.chk_translate = QCheckBox("Show PL/SK → DE translations")
        self.chk_translate.setChecked(True)
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.status_label = QLabel("Idle.")
        self.countdown_label = QLabel("Next autosave in —")
        self.saving_label = QLabel("")
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setLineWrapMode(QTextEdit.NoWrap)
        self.transcript.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        device_box = QGroupBox("Audio Sources")
        dev_layout = QVBoxLayout()
        dev_layout.addWidget(QLabel("Microphone:"))
        dev_layout.addWidget(self.mic_combo)
        dev_layout.addWidget(QLabel("System / Loopback (optional, ignored for now):"))
        dev_layout.addWidget(self.sys_combo)
        dev_layout.addWidget(self.chk_translate)
        device_box.setLayout(dev_layout)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        status_row.addWidget(self.countdown_label)
        status_row.addSpacing(12)
        status_row.addWidget(self.saving_label)
        central = QWidget()
        root = QVBoxLayout(central)
        root.addWidget(device_box)
        root.addLayout(btn_row)
        root.addWidget(self.transcript)
        root.addLayout(status_row)
        self.setCentralWidget(central)
        self.autosave_interval = AUTOSAVE_INTERVAL_SEC_DEFAULT
        self.seconds_left = self.autosave_interval
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._on_tick)
        self.worker: Optional[MicWorker] = None
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause_resume)
        self.btn_stop.clicked.connect(self._on_stop)
        self._update_countdown_label()

    def _populate_device_combos(self):
        self.mic_combo.clear()
        self.sys_combo.clear()
        if not self.devices:
            self.mic_combo.addItem("No devices found")
            self.sys_combo.addItem("No devices found")
            self.mic_combo.setEnabled(False)
            self.sys_combo.setEnabled(False)
            return
        for d in self.devices:
            if d.max_in > 0:
                self.mic_combo.addItem(d.display(), d.index)
        for d in self.devices:
            if d.max_out > 0 and "WASAPI" in d.api:
                self.sys_combo.addItem(d.display(), d.index)
        if self.sys_combo.count() == 0:
            for d in self.devices:
                if d.max_out > 0:
                    self.sys_combo.addItem(d.display(), d.index)
        self._try_select(self.mic_combo, 23)
        self._try_select(self.sys_combo, 18)

    @staticmethod
    def _try_select(combo: QComboBox, want_idx: int):
        for i in range(combo.count()):
            if combo.itemData(i) == want_idx:
                combo.setCurrentIndex(i)
                break

    def _update_countdown_label(self):
        m = self.seconds_left // 60
        s = self.seconds_left % 60
        self.countdown_label.setText(f"Next autosave in {m}:{s:02d}")

    @Slot()
    def _on_start(self):
        if self.worker and self.worker.isRunning():
            return
        mic_idx = self.mic_combo.currentData()
        self.transcript.append(f"# Session started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.transcript.append(f"# Mic device index: {mic_idx}")
        self.transcript.append("")
        self.worker = MicWorker(
            mic_index=mic_idx,
            translate_flag=lambda: self.chk_translate.isChecked(),
            save_root=Path("."),
        )
        self.worker.line_ready.connect(self._on_line)
        self.worker.status_text.connect(self._on_status)
        self.seconds_left = self.autosave_interval
        self._update_countdown_label()
        self.worker.start_run()
        self.timer.start()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)

    @Slot()
    def _on_pause_resume(self):
        if not self.worker:
            return
        if self.btn_pause.text() == "Pause":
            self.worker.pause()
            self.btn_pause.setText("Resume")
        else:
            self.worker.resume()
            self.btn_pause.setText("Pause")

    @Slot()
    def _on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_run()
            self.worker.wait(2000)
        self.timer.stop()
        self.status_label.setText("Idle.")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    @Slot(str)
    def _on_line(self, text: str):
        if not self.chk_translate.isChecked():
            lines = [ln for ln in text.splitlines() if "→ DE:" not in ln]
            text = "\n".join(lines)
        self.transcript.append(text)

    @Slot(str)
    def _on_status(self, text: str):
        self.status_label.setText(text)

    @Slot()
    def _on_tick(self):
        if self.seconds_left > 0:
            self.seconds_left -= 1
            self._update_countdown_label()
        else:
            self.saving_label.setText("Saving…")
            QTimer.singleShot(600, lambda: self.saving_label.setText(""))
            self.seconds_left = self.autosave_interval
            self._update_countdown_label()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop_run()
            self.worker.wait(1500)
        event.accept()

def run():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
