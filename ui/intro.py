import random
import time

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
from PySide2.QtCore import Qt, QTimer, QRectF, Signal


class Intro(QWidget):
    cue = Signal()        # time for her to teleport in
    finished = Signal()

    DUR = 2.6

    def __init__(self, name):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.name = name
        self.t0 = time.time()
        self.cued = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def start(self):
        self.t0 = time.time()
        self.show()
        self.timer.start(16)

    def _tick(self):
        t = time.time() - self.t0
        if t >= 1.0 and not self.cued:
            self.cued = True
            self.cue.emit()
        if t >= self.DUR:
            self.timer.stop()
            self.close()
            self.finished.emit()
            return
        self.update()

    def _edges(self, rng, inten):
        W, H, step = self.width(), self.height(), 24
        amp = 4 + 26 * inten
        path = QPainterPath()
        for x in range(0, W, step):
            y = rng.uniform(0, amp)
            path.moveTo(x, 0) if x == 0 else path.lineTo(x, y)
        path.moveTo(0, H)
        for x in range(0, W, step):
            path.lineTo(x, H - rng.uniform(0, amp))
        path.moveTo(0, 0)
        for y in range(0, H, step):
            path.lineTo(rng.uniform(0, amp), y)
        path.moveTo(W, 0)
        for y in range(0, H, step):
            path.lineTo(W - rng.uniform(0, amp), y)
        return path

    def paintEvent(self, _):
        t = time.time() - self.t0
        W, H = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if t < 1.5:
            inten = min(1.0, t / 0.9) if t < 1.0 else max(0.0, 1 - (t - 1.0) / 0.5)
            path = self._edges(random.Random(int(t * 22)), inten)
            p.setBrush(Qt.NoBrush)
            for w, a in ((12, 50), (5, 120), (2, 255)):
                p.setPen(QPen(QColor(255, 32, 64, a), w))
                p.drawPath(path)
        if 1.0 <= t < 1.2:
            p.fillRect(self.rect(), QColor(255, 255, 255, int(255 * (1 - (t - 1.0) / 0.2))))
        if t >= 1.0:
            alpha = 255 if t < 2.2 else max(0, int(255 * (1 - (t - 2.2) / 0.4)))
            cx, cy = W // 2, H // 2 - 140
            p.setBrush(QColor(16, 22, 38, min(235, alpha)))
            p.setPen(QPen(QColor(255, 43, 214, alpha), 2))
            p.drawRoundedRect(QRectF(cx - 150, cy - 50, 300, 100), 12, 12)
            p.setPen(QColor(230, 250, 255, alpha))
            p.setFont(QFont("Pyidaungsu", 22, QFont.Bold))
            p.drawText(QRectF(cx - 150, cy - 46, 300, 44), Qt.AlignCenter, "htut A.I")
            msg = f"online. hi, {self.name}."
            n = max(0, int((t - 1.1) * 22))
            p.setPen(QColor(255, 155, 240, alpha))
            p.setFont(QFont("Pyidaungsu", 14))
            p.drawText(QRectF(cx - 150, cy, 300, 40), Qt.AlignCenter, msg[:n])