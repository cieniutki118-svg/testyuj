"""Główne okno aplikacji."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    Qt, Slot, QThread, QMetaObject, QTimer, Q_ARG
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QCheckBox, QSpinBox, QGroupBox, QFormLayout, QSplitter,
    QTextEdit, QMessageBox, QScrollArea
)

from core.audio_in import (
    AudioCapture, list_input_devices, list_output_devices, find_usb_input
)
from core.demo_source import DemoSource
from core.worker import ProcessingWorker
from ui.waterfall import Waterfall
from ui.level_meter import LevelMeter


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SP5MIG Signal Hunter")
        self.resize(1500, 950)

        self.fs = 48000
        self.blocksize = 1024
        self._closing = False
        self.capture = None
        self.demo = None

        # wątek przetwarzania
        self.proc_thread = QThread(self)
        self.worker = ProcessingWorker(self.fs)
        self.worker.moveToThread(self.proc_thread)
        self.worker.spectrum_ready.connect(self._on_spectrum_ready)
        self.worker.level_ready.connect(self._on_level_ready)
        self.worker.text_ready.connect(self._append_text)
        self.worker.wpm_ready.connect(self._on_wpm_ready)
        self.worker.recording_started.connect(self._on_rec_started)
        self.worker.recording_stopped.connect(self._on_rec_stopped)
        self.worker.shutdown_done.connect(self._on_shutdown_done)
        self.proc_thread.start()

        self._build_ui()
        self._populate_devices()
        self._auto_start()

    # --------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)

        # źródło
        gb_src = QGroupBox("Źródło sygnału")
        f_src = QFormLayout(gb_src)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Urządzenie audio (USB)", "device")
        self.source_combo.addItem("DEMO (syntetyczny CW)", "demo")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        f_src.addRow("Źródło:", self.source_combo)

        self.device_combo = QComboBox()
        f_src.addRow("Wejście:", self.device_combo)
        self.btn_refresh = QPushButton("Odśwież listę wejść")
        self.btn_refresh.clicked.connect(self._populate_devices)
        f_src.addRow(self.btn_refresh)

        self.btn_start = QPushButton("Start nasłuchu")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self._toggle_stream)
        f_src.addRow(self.btn_start)
        lv.addWidget(gb_src)

        # CW
        gb_cw = QGroupBox("CW / Morse")
        f_cw = QFormLayout(gb_cw)
        self.spin_tone = QSpinBox()
        self.spin_tone.setRange(200, 1500)
        self.spin_tone.setSingleStep(10)
        self.spin_tone.setValue(700)
        self.spin_tone.setSuffix(" Hz")
        self.spin_tone.valueChanged.connect(self._on_tone_changed)
        f_cw.addRow("Ton CW:", self.spin_tone)
        self.lbl_wpm = QLabel("WPM: -- (auto)")
        f_cw.addRow(self.lbl_wpm)
        self.chk_morse = QCheckBox("Dekoduj CW")
        self.chk_morse.setChecked(True)
        self.chk_morse.toggled.connect(self._on_morse_toggle)
        f_cw.addRow(self.chk_morse)
        lv.addWidget(gb_cw)

        # DSP
        gb_dsp = QGroupBox("DSP")
        v_dsp = QVBoxLayout(gb_dsp)
        self.chk_dsp = QCheckBox("Włącz DSP (bandpass + spectral sub.)")
        self.chk_dsp.setChecked(True)
        self.chk_dsp.toggled.connect(self._on_dsp_toggle)
        v_dsp.addWidget(self.chk_dsp)
        lv.addWidget(gb_dsp)

        # Odsłuch
        gb_out = QGroupBox("Odsłuch (RAW/DSP)")
        f_out = QFormLayout(gb_out)
        self.out_combo = QComboBox()
        f_out.addRow("Wyjście:", self.out_combo)
        self.btn_refresh_out = QPushButton("Odśwież listę wyjść")
        self.btn_refresh_out.clicked.connect(self._populate_outputs)
        f_out.addRow(self.btn_refresh_out)
        self.chk_out = QCheckBox("Włącz odsłuch")
        self.chk_out.toggled.connect(self._on_output_toggle)
        f_out.addRow(self.chk_out)
        lv.addWidget(gb_out)

        # Nagrywanie
        gb_rec = QGroupBox("Nagrywanie")
        v_rec = QVBoxLayout(gb_rec)
        self.btn_rec = QPushButton("● Nagrywaj do WAV")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self._toggle_record)
        v_rec.addWidget(self.btn_rec)
        self.lbl_rec = QLabel("Brak nagrania")
        self.lbl_rec.setWordWrap(True)
        v_rec.addWidget(self.lbl_rec)
        lv.addWidget(gb_rec)

        # Widok
        gb_view = QGroupBox("Widok")
        v_view = QVBoxLayout(gb_view)
        self.btn_reset_view = QPushButton("Reset zoom")
        self.btn_reset_view.clicked.connect(self._reset_zoom)
        v_view.addWidget(self.btn_reset_view)
        lv.addWidget(gb_view)

        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setFixedWidth(360)

        # prawa strona
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)

        self.glw = pg.GraphicsLayoutWidget()
        self.spectrum_plot = self.glw.addPlot(row=0, col=0)
        self.spectrum_plot.setLabel("left", "Amplituda", units="dBFS")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.25)
        self.spectrum_plot.setYRange(-110, 0)
        self.spectrum_plot.setXRange(0, self.fs / 2, padding=0)
        self.waterfall_plot = self.glw.addPlot(row=1, col=0)
        self.waterfall_plot.setXLink(self.spectrum_plot)
        self.glw.ci.layout.setRowStretchFactor(0, 1)
        self.glw.ci.layout.setRowStretchFactor(1, 2)

        self.curve = self.spectrum_plot.plot(
            pen=pg.mkPen("#39d353", width=1.5))
        self.tone_marker = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#ff5555", width=1, style=Qt.DashLine))
        self.spectrum_plot.addItem(self.tone_marker)
        self.spectrum_plot.scene().sigMouseClicked.connect(
            self._on_spectrum_clicked)

        self.waterfall = Waterfall(self.waterfall_plot, n_rows=280,
                                   n_bins=512, fs=self.fs)
        rv.addWidget(self.glw, stretch=1)

        self.level = LevelMeter()
        rv.addWidget(self.level)

        gb_text = QGroupBox("Zdekodowany tekst CW")
        tv = QVBoxLayout(gb_text)
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Consolas", 28, QFont.Bold))
        self.text_display.setStyleSheet(
            "background-color:#050505; color:#7ee787;"
            " border:1px solid #1f6feb;")
        self.text_display.setFixedHeight(140)
        tv.addWidget(self.text_display)
        row = QHBoxLayout()
        self.btn_clear = QPushButton("Wyczyść tekst")
        self.btn_clear.clicked.connect(self._clear_text)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        tv.addLayout(row)
        rv.addWidget(gb_text)

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ----------------------------------------------------- urządzenia
    def _populate_devices(self):
        self.device_combo.clear()
        self._devices = list_input_devices()
        preferred = 0
        usb = find_usb_input()
        for i, d in enumerate(self._devices):
            tag = " [USB]" if d.is_usb else ""
            self.device_combo.addItem(f"{d.index}: {d.name}{tag}", d.index)
            if usb and d.index == usb.index:
                preferred = i
        if self._devices:
            self.device_combo.setCurrentIndex(preferred)

    def _populate_outputs(self):
        self.out_combo.clear()
        self.out_combo.addItem("Domyślne", None)
        for idx, name in list_output_devices():
            self.out_combo.addItem(f"{idx}: {name}", idx)

    def _selected_device(self):
        idx = self.device_combo.currentData()
        if idx is None:
            return None
        for d in self._devices:
            if d.index == idx:
                return d
        return None

    # ----------------------------------------------------- źródło
    def _on_source_changed(self, _):
        mode = self.source_combo.currentData()
        is_demo = (mode == "demo")
        self.device_combo.setEnabled(not is_demo)
        self.btn_refresh.setEnabled(not is_demo)
        # restart strumienia
        if self.btn_start.isChecked():
            self._stop_stream()
            self._start_stream()

    def _auto_start(self):
        self._populate_outputs()
        self._start_stream()

    def _toggle_stream(self, checked):
        if checked:
            self._start_stream()
        else:
            self._stop_stream()

    def _start_stream(self):
        self._stop_stream()
        mode = self.source_combo.currentData()
        if mode == "demo":
            self.demo = DemoSource(self.fs, self.blocksize)
            self.demo.block_ready.connect(self.worker.process_block)
            self.demo.start()
        else:
            dev = self._selected_device()
            if dev is None:
                QMessageBox.warning(self, "Brak wejścia",
                                    "Nie znaleziono urządzenia wejściowego.")
                self.btn_start.setChecked(False)
                return
            self.capture = AudioCapture(dev.index, self.fs, self.blocksize)
            self.capture.block_ready.connect(self.worker.process_block)
            self.capture.error.connect(self._on_capture_error)
            self.capture.start()
        self.btn_start.setChecked(True)
        self.btn_start.setText("Stop nasłuchu")

    def _stop_stream(self):
        if self.capture is not None:
            try:
                self.capture.stop()
            except Exception:
                pass
            self.capture = None
        if self.demo is not None:
            try:
                self.demo.stop()
            except Exception:
                pass
            self.demo = None
        self.btn_start.setChecked(False)
        self.btn_start.setText("Start nasłuchu")

    def _on_capture_error(self, msg):
        QMessageBox.warning(self, "Błąd wejścia audio", msg)

    # ----------------------------------------------------- sloty workera
    @Slot(object, object)
    def _on_spectrum_ready(self, freqs, spec_db):
        self.curve.setData(freqs, spec_db)
        self.waterfall.update_frame(freqs, spec_db, self.blocksize / self.fs)

    @Slot(float, float, float)
    def _on_level_ready(self, level_db, snr_db, peak_hz):
        self.level.update_values(level_db, snr_db, peak_hz)

    @Slot(float)
    def _on_wpm_ready(self, wpm):
        self.lbl_wpm.setText(f"WPM: {wpm:4.1f} (auto)")

    @Slot(str)
    def _on_rec_started(self, path):
        self.lbl_rec.setText(f"Nagrywam: {path}")
        self.btn_rec.setChecked(True)
        self.btn_rec.setText("■ Zatrzymaj nagrywanie")

    @Slot(str)
    def _on_rec_stopped(self, path):
        self.lbl_rec.setText(f"Zapisano: {path}" if path else "Brak nagrania")
        self.btn_rec.setChecked(False)
        self.btn_rec.setText("● Nagrywaj do WAV")

    # ----------------------------------------------------- UI -> worker
    def _on_tone_changed(self, val):
        self.tone_marker.setValue(float(val))
        QMetaObject.invokeMethod(self.worker, "set_tone",
                                 Qt.QueuedConnection,
                                 Q_ARG(float, float(val)))

    def _on_dsp_toggle(self, checked):
        QMetaObject.invokeMethod(self.worker, "set_dsp_enabled",
                                 Qt.QueuedConnection,
                                 Q_ARG(bool, bool(checked)))

    def _on_morse_toggle(self, checked):
        QMetaObject.invokeMethod(self.worker, "set_morse_enabled",
                                 Qt.QueuedConnection,
                                 Q_ARG(bool, bool(checked)))

    def _on_output_toggle(self, checked):
        dev = self.out_combo.currentData()
        if dev is not None:
            QMetaObject.invokeMethod(self.worker, "set_output_device",
                                     Qt.QueuedConnection,
                                     Q_ARG(int, int(dev)))
        QMetaObject.invokeMethod(self.worker, "set_output_enabled",
                                 Qt.QueuedConnection,
                                 Q_ARG(bool, bool(checked)))

    def _toggle_record(self, checked):
        if checked:
            QMetaObject.invokeMethod(self.worker, "start_recording",
                                     Qt.QueuedConnection)
        else:
            QMetaObject.invokeMethod(self.worker, "stop_recording",
                                     Qt.QueuedConnection)

    def _clear_text(self):
        self.text_display.clear()
        QMetaObject.invokeMethod(self.worker, "reset_morse",
                                 Qt.QueuedConnection)

    def _on_spectrum_clicked(self, ev):
        if ev.double():
            return
        pos = self.spectrum_plot.vb.mapSceneToView(ev.scenePos())
        f = pos.x()
        if 100 <= f <= self.fs / 2 - 100:
            self.spin_tone.setValue(int(f))

    def _reset_zoom(self):
        self.spectrum_plot.setXRange(0, self.fs / 2, padding=0)
        self.spectrum_plot.setYRange(-110, 0, padding=0)
        self.waterfall.reset()

    def _append_text(self, s):
        cur = self.text_display.toPlainText() + s
        if len(cur) > 4000:
            cur = cur[-4000:]
        self.text_display.setPlainText(cur)
        self.text_display.moveCursor(self.text_display.textCursor().End)

    # ----------------------------------------------------- zamykanie
    def _on_shutdown_done(self):
        if self.proc_thread.isRunning():
            self.proc_thread.quit()
            self.proc_thread.wait(3000)
        self.close()

    def _force_close(self):
        if self.proc_thread.isRunning():
            self.proc_thread.quit()
            self.proc_thread.wait(5000)
        self.close()

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._closing = True
        event.ignore()

        try:
            self._stop_stream()
        except Exception:
            pass

        QMetaObject.invokeMethod(self.worker, "shutdown", Qt.QueuedConnection)
        QTimer.singleShot(4000, self._force_close)
