import math, random
from PySide2.QtWidgets import QWidget
from PySide2.QtGui import QPainter, QColor, QPen, QFont
from PySide2.QtCore import QTimer, Qt, QRectF, Signal
from ui.sprite_sheet import load_sheets

STATES = ("idle", "thinking", "talking", "happy", "sleepy", "tease")
COLORS = {"idle": "#00f0ff", "thinking": "#ffd400", "talking": "#ff2bd6",
          "happy": "#39ff14", "sleepy": "#6a7bb0", "tease": "#ff6ec7"}
SPRITE_FOR = {"idle": "idle", "thinking": "glitch", "talking": "idle",
              "happy": "dance", "sleepy": "sleep", "tease": "grin"}

class CharacterWidget(QWidget):
    poked = Signal()

    def __init__(self, sprite_dir, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 300)
        self.sheets = load_sheets(sprite_dir)
        self.state, self.tick = "idle", 0
        self.blink_left = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(80)

    def set_state(self, s):
        if s != self.state:
            self.state, self.tick = s, 0

    def mousePressEvent(self, e):
        self.poked.emit()

    def _step(self):
        self.tick += 1
        if self.state == "idle" and self.blink_left == 0 and random.random() < 0.02:
            self.blink_left = 3
        if self.blink_left:
            self.blink_left -= 1
        self.update()

    def _current_sheet(self):
        name = SPRITE_FOR.get(self.state, "idle")
        if self.state == "idle" and self.blink_left and "blink" in self.sheets:
            name = "blink"
        return self.sheets.get(name)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)
        sheet = self._current_sheet()
        if not sheet:
            self._draw_fallback(p, math.sin(self.tick / 7) * 5)
            return
        pm = sheet.frames[(self.tick // 2) % len(sheet.frames)]
        scale = 1.0 + (0.03 * math.sin(self.tick) if self.state == "talking" else 0)
        pm = pm.scaled(int(self.width() * scale), int(self.height() * scale),
                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
        glow = QColor(COLORS[self.state]); glow.setAlpha(40)
        p.setPen(Qt.NoPen); p.setBrush(glow)
        p.drawEllipse(QRectF(10, self.height() - 40, self.width() - 20, 28))
        p.drawPixmap((self.width() - pm.width()) // 2, self.height() - pm.height(), pm)