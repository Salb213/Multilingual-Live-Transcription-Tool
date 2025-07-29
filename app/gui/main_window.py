from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import sounddevice as sd
from PySide6.QtCore import QThread, Signal, Slot, QTimer, Qt, QObject
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

from app.audio.capture import record_block, TARGET_SR
from app.asr.transcribe import ASREngine
from app.asr.translate import translate_to_de
from app.io.autosave import AutoSaver

AUTOSAVE_INTERVAL_SEC_DEFAULT = 15 * 60

class DeviceInfo:
    def __init__(self, index: int, name: str, api: str, max_in: int, max_out: int):
        self.index = index
        self.name = name
        self.api = api
        self.max_in = max_in
        self.max_out = max_out

    def display(self) -> str:
        return f"{self.index}: {self.name} | {self.api} | in={self.max_in} out={self.max_out}"

def list_devices() -> List[DeviceInfo]:
    out: List[DeviceInfo] = []
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        api_name = apis[d["hostapi"]]["name"]
        out.append(DeviceInfo(i, d["name"], api_name, d["max_input_channels"], d["max_output_channels"]))
    return out

class MicWorker(QThread):
    line_ready = Signal(str)
    status_text = Signal(str)

    def __init__(
        self,
        mic_index: int,
        translate_flag: Callable[[], bool],
        force_de_flag: Callable[[], bool],
        save_root: Path,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.mic_index = mic_index
        self.translate_flag = translate_flag
        self.force_de_flag = force_de_flag
        self._running = False
        self._paused = False
        self.asr: Optional[ASREngine] = None
        self.saver = AutoSaver(save_root, interval_minutes=15)

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
        try:
            if self.asr is None:
                self.status_text.emit("Loading ASR model (medium)…")
                self.asr = ASREngine()
                self.status_text.emit("Listening…")
            while self._running:
                if self._paused:
                    self.msleep(50)
                    continue
                audio16 = record_block(self.mic_index, seconds=4.0).squeeze(-1)
                text, lang, prob = self.asr.transcribe(audio16, TARGET_SR, forced_lang=None)
                if self.force_de_flag() and (lang not in ("de", "en", "pl", "sk") or prob < 0.60):
                    text, lang, prob = self.asr.transcribe(audio16, TARGET_SR, forced_lang="de")
                if not text:
                    continue
                lang_up = lang.upper()
                line = f"[{time.strftime('%H:%M:%S')}] Speaker 1 (You) ({lang_up}): {text}"
                if self.translate_flag() and lang in ("pl", "sk"):
                    try:
                        tr = translate_to_de(text, lang)
                        if tr:
                            line += f"\n               → DE: {tr}"
                    except Exception:
                        pass
                self.line_ready.emit(line)
                self.saver.write(line)
        finally:
            self.saver.close()
            self.status_text.emit("Stopped.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multilingual Live Transcription")
        self.resize(1000, 700)
        self.devices = list_devices()
        self.mic_combo = QComboBox()
        self.sys_combo = QComboBox()
        self.chk_translate = QCheckBox("Show PL/SK → DE translations")
        self.chk_translate.setChecked(True)
        self.chk_force_de = QCheckBox("Force German if unsure")
        self.chk_force_de.setChecked(True)
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.status_label = QLabel("Idle.")
        self.countdown_label = QLabel("Next autosave in —")
        self.saving_label = QLabel("")
        self.transcript = QTextEdit(readOnly=True, lineWrapMode=QTextEdit.NoWrap)
        self.transcript.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._populate_device_combos()
        self._build_layout()
        self.autosave_interval = AUTOSAVE_INTERVAL_SEC_DEFAULT
        self.seconds_left = self.autosave_interval
        self.timer = QTimer(interval=1000, timeout=self._tick)
        self.worker: Optional[MicWorker] = None
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._pause_resume)
        self.btn_stop.clicked.connect(self._stop)
        self._update_countdown()

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
        for i in range(self.mic_combo.count()):
            if self.mic_combo.itemData(i) == 23:
                self.mic_combo.setCurrentIndex(i)
                break

    def _build_layout(self):
        device_box = QGroupBox("Audio Sources")
        dev_layout = QVBoxLayout()
        dev_layout.addWidget(QLabel("Microphone:"))
        dev_layout.addWidget(self.mic_combo)
        dev_layout.addWidget(QLabel("System output (ignored for now):"))
        dev_layout.addWidget(self.sys_combo)
        dev_layout.addWidget(self.chk_translate)
        dev_layout.addWidget(self.chk_force_de)
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

    def _update_countdown(self):
        m, s = divmod(self.seconds_left, 60)
        self.countdown_label.setText(f"Next autosave in {m}:{s:02d}")

    @Slot()
    def _start(self):
        if self.worker and self.worker.isRunning():
            return
        mic_idx = self.mic_combo.currentData()
        self.transcript.append(f"# Started {time.strftime('%Y-%m-%d %H:%M:%S')}  (mic {mic_idx})\n")
        self.worker = MicWorker(
            mic_index=mic_idx,
            translate_flag=lambda: self.chk_translate.isChecked(),
            force_de_flag=lambda: self.chk_force_de.isChecked(),
            save_root=Path("."),
        )
        self.worker.line_ready.connect(self._append_line)
        self.worker.status_text.connect(self.status_label.setText)
        self.worker.start_run()
        self.seconds_left = self.autosave_interval
        self._update_countdown()
        self.timer.start()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)

    @Slot()
    def _pause_resume(self):
        if not self.worker:
            return
        if self.btn_pause.text() == "Pause":
            self.worker.pause()
            self.btn_pause.setText("Resume")
        else:
            self.worker.resume()
            self.btn_pause.setText("Pause")

    @Slot()
    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_run()
            self.worker.wait(2000)
        self.timer.stop()
        self.status_label.setText("Idle.")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    @Slot(str)
    def _append_line(self, text: str):
        if not self.chk_translate.isChecked():
            text = "\n".join(ln for ln in text.splitlines() if "→ DE:" not in ln)
        self.transcript.append(text)

    @Slot()
    def _tick(self):
        if self.seconds_left > 0:
            self.seconds_left -= 1
            self._update_countdown()
        else:
            self.saving_label.setText("Saving…")
            QTimer.singleShot(800, lambda: self.saving_label.clear())
            self.seconds_left = self.autosave_interval
            self._update_countdown()

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
