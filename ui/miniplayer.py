from PySide2.QtWidgets import QWidget, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PySide2.QtGui import QPainter, QColor, QPen, QFontMetrics, QGuiApplication
from PySide2.QtCore import Qt, QTimer, QRectF, QRect, Signal

from ui.pet.ledges import register_seat, unregister_seat


def fmt_ms(ms):
    s = max(0, int(ms)) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


class MiniPlayer(QWidget):
    closed = Signal()
    KEY = "miniplayer"

    def __init__(self, music):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.music = music
        self.setFixedSize(340, 136)
        self.drag = None
        self.placed = False
        self.seeking = False
        self._last_title = None

        self.cover = QLabel()
        self.cover.setFixedSize(56, 56)
        self.cover.setScaledContents(True)
        self.title = QLabel("Nothing playing")
        self.title.setStyleSheet("color:#e6faff; font-size:14px; font-weight:bold; background:transparent;")
        self.sub = QLabel("Now playing")
        self.sub.setStyleSheet("color:#ff9bf0; font-size:12px; background:transparent;")
        self.close_btn = QPushButton("x")
        self.close_btn.setFixedSize(26, 26)

        text = QVBoxLayout()
        text.addWidget(self.title)
        text.addWidget(self.sub)
        top = QHBoxLayout()
        top.addWidget(self.cover)
        top.addLayout(text, 1)
        top.addWidget(self.close_btn, 0, Qt.AlignTop)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1)
        self.slider.setPageStep(10000)
        self.cur = QLabel("00:00")
        self.total = QLabel("00:00")
        for l in (self.cur, self.total):
            l.setStyleSheet("color:#9fdcff; font-size:11px; background:transparent;")
        self.prev = QPushButton("Prev")
        self.play = QPushButton("Play")
        self.next = QPushButton("Next")
        for b in (self.prev, self.play, self.next):
            b.setFixedWidth(60)
        bottom = QHBoxLayout()
        bottom.addWidget(self.cur)
        bottom.addStretch(1)
        for b in (self.prev, self.play, self.next):
            bottom.addWidget(b)
        bottom.addStretch(1)
        bottom.addWidget(self.total)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.addLayout(top)
        root.addWidget(self.slider)
        root.addLayout(bottom)

        self.close_btn.clicked.connect(self.hide_player)
        self.prev.clicked.connect(self.music.prev_song)
        self.next.clicked.connect(lambda: self.music.next_song(False))
        self.play.clicked.connect(self.music.toggle)
        self.slider.sliderPressed.connect(lambda: setattr(self, "seeking", True))
        self.slider.sliderReleased.connect(self._release)
        self.slider.actionTriggered.connect(lambda _a: QTimer.singleShot(0, self._seek))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(300)

        register_seat(self.KEY, self._seat_rect, int(self.winId()))

    # ---------- seat for the pet ----------
    def _seat_rect(self):
        if not self.isVisible():
            return None
        g = self.geometry()
        return QRect(g.left() + 8, g.top() + 2, g.width() - 16, g.height())

    # ---------- control ----------
    def show_song(self, title, pm):
        if not self.placed:
            scr = QGuiApplication.primaryScreen().availableGeometry()
            self.move(scr.right() - self.width() - 30, scr.bottom() - self.height() - 30)
            self.placed = True
        self.show()
        self.raise_()
        self._refresh(force=True)

    def hide_player(self):
        self.hide()
        self.closed.emit()

    def _seek(self):
        self.music.seek(self.slider.value())

    def _release(self):
        self._seek()
        self.seeking = False

    def _refresh(self, force=False):
        if not self.isVisible():
            return
        s = self.music.snapshot()
        if force or s["title"] != self._last_title:
            self._last_title = s["title"]
            text = s["title"] or "Nothing playing"
            self.title.setText(QFontMetrics(self.title.font()).elidedText(text, Qt.ElideRight, 215))
            if s["pixmap"] is not None:
                self.cover.setPixmap(s["pixmap"])
            else:
                self.cover.clear()
        self.play.setText("Pause" if s["playing"] else "Play")
        self.sub.setText("Now playing" if s["playing"] else "Paused")
        if s["length"] > 0:
            self.slider.setMaximum(s["length"])
            self.total.setText(fmt_ms(s["length"]))
        if not self.seeking:
            self.slider.setValue(max(0, s["t"]))
        self.cur.setText(fmt_ms(s["t"]))

    # ---------- window ----------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(16, 22, 38, 245))
        p.setPen(QPen(QColor("#ff2bd6"), 2))
        p.drawRoundedRect(QRectF(2, 2, self.width() - 4, self.height() - 4), 12, 12)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drag = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self.drag is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPos() - self.drag)

    def mouseReleaseEvent(self, e):
        self.drag = None

    def closeEvent(self, e):
        unregister_seat(self.KEY)
        super().closeEvent(e)