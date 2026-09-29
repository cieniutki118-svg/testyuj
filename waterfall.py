"""Waterfall na ImageItem."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF


class Waterfall:
    def __init__(self, plot_item, n_rows=280, n_bins=512, fs=48000,
                 db_min=-100, db_max=-20):
        self.plot = plot_item
        self.n_rows = n_rows
        self.n_bins = n_bins
        self.fs = fs
        self.db_min = db_min
        self.db_max = db_max
        self.data = np.full((n_rows, n_bins), db_min, dtype=np.float32)

        self.image = pg.ImageItem()
        self.image.setColorMap(pg.colormap.get("turbo"))
        self.image.setLevels([db_min, db_max])
        self.plot.addItem(self.image)
        self.plot.setLabel("bottom", "Czestotliwosc", units="Hz")
        self.plot.setLabel("left", "Czas")
        self.plot.invertY(True)
        self.plot.setMouseEnabled(x=True, y=False)
        self._block_time = 1.0

    def reset(self):
        self.data[:] = self.db_min
        self._redraw()

    def _redraw(self):
        y_max = self.n_rows * self._block_time
        self.image.setImage(self.data, autoLevels=False)
        self.image.setRect(QRectF(0.0, 0.0, self.fs / 2.0, y_max))

    def update_frame(self, freqs, spec_db, block_time):
        self._block_time = block_time
        if len(spec_db) != self.n_bins:
            idx = np.linspace(0, len(spec_db) - 1, self.n_bins)
            row = np.interp(idx, np.arange(len(spec_db)), spec_db
                            ).astype(np.float32)
        else:
            row = spec_db.astype(np.float32)
        self.data[1:] = self.data[:-1]
        self.data[0] = row
        self._redraw()
