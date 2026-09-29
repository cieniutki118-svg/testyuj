import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6.QtWidgets import QApplication
import pyqtgraph as pg

from ui.main_window import MainWindow


def main():
    pg.setConfigOption("background", "#0a0a0a")
    pg.setConfigOption("foreground", "#d0d0d0")
    pg.setConfigOption("antialias", True)

    app = QApplication(sys.argv)
    app.setApplicationName("SP5MIG Signal Hunter")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
