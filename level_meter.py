"""Wskaźnik poziomu + SNR + peak."""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar


class LevelMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(-80, 0)
        self.level_bar.setValue(-80)
        self.level_bar.setTextVisible(True)
        self.level_bar.setFormat("Poziom: %v dB")
        self.level_bar.setStyleSheet(
            "QProgressBar { background-color:#101010; color:#d0d0d0;"
            " border:1px solid #303030; height:22px; text-align:center; }"
            "QProgressBar::chunk { background-color: qlineargradient("
            "x1:0,y1:0,x2:1,y2:0, stop:0 #00c853, stop:0.6 #ffd600,"
            " stop:1 #d50000); }")
        self.snr_label = QLabel("SNR: -- dB")
        self.snr_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size:16px; color:#7ee787;")
        self.peak_label = QLabel("Peak: -- Hz")
        self.peak_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size:16px; color:#79c0ff;")
        row = QHBoxLayout()
        row.addWidget(self.snr_label)
        row.addWidget(self.peak_label)
        row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.level_bar)
        layout.addLayout(row)

    def update_values(self, level_db, snr_db, peak_hz):
        self.level_bar.setValue(int(max(-80, min(0, level_db))))
        self.snr_label.setText(f"SNR: {snr_db:5.1f} dB")
        self.peak_label.setText(f"Peak: {peak_hz:7.0f} Hz")
