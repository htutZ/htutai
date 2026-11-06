from PySide2.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PySide2.QtGui import QPainter, QColor, QPen, QGuiApplication
from PySide2.QtCore import Qt, QTimer, QRectF, Signal


class Card(QWidget):
    clicked = Signal(str)

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.pet = None
        self.setFixedWidth(300)

        self.thumb = QLabel()
        self.thumb.setFixedSize(54, 54)
        self.thumb.setScaledContents(True)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setStyleSheet("color:#e6faff; font-size:14px; font-weight:bold; background:transparent;")
        self.sub = QLabel()
        self.sub.setStyleSheet("color:#ff9bf0; font-size:12px; background:transparent;")
        self.row = QHBoxLayout()

        text = QVBoxLayout()
        text.addWidget(self.title)
        text.addWidget(self.sub)
        top = QHBoxLayout()
        top.addWidget(self.thumb)
        top.addLayout(text, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.addLayout(top)
        root.addLayout(self.row)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.follow = QTimer(self)
        self.follow.timeout.connect(self._follow)
        self.follow.start(40)

    def show_card(self, title, sub="", pixmap=None, buttons=(), ms=6000):
        self.title.setText(title)
        self.sub.setText(sub)
        self.sub.setVisible(bool(sub))
        if pixmap is not None and not pixmap.isNull():
            self.thumb.setPixmap(pixmap)
            self.thumb.show()
        else:
            self.thumb.hide()
        while self.row.count():
            item = self.row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for cid, label in buttons:
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, c=cid: self._click(c))
            self.row.addWidget(b)
        self.adjustSize()
        self.show()
        self._place()
        if ms > 0:
            self.timer.start(ms)
        else:
            self.timer.stop()

    def hide_card(self):
        self.timer.stop()
        self.hide()

    def _click(self, cid):
        self.clicked.emit(cid)
        self.hide_card()

    def _follow(self):
        if self.isVisible():
            self._place()

    def _place(self):
        if not self.pet:
            return
        g = self.pet.geometry()
        scr = (QGuiApplication.screenAt(g.center()) or QGuiApplication.primaryScreen()).geometry()
        x = g.right() + 6
        if x + self.width() > scr.right():
            x = g.left() - self.width() - 6
        self.move(x, max(scr.top() + 4, g.top() + 30))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(16, 22, 38, 240))
        p.setPen(QPen(QColor("#ff2bd6"), 2))
        p.drawRoundedRect(QRectF(2, 2, self.width() - 4, self.height() - 4), 12, 12)