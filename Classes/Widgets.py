from PySide2.QtWidgets import (QListWidget, QVBoxLayout, QPushButton, QWidget, QTextEdit, QHBoxLayout, QComboBox,
                               QLineEdit, QLabel, QListWidgetItem, QStyle, QSlider, QApplication, QSizePolicy)
from PySide2.QtCore import QUrl, QFile, QTextStream, Qt, QTimer, QSize
from PySide2.QtGui import QImage, QPixmap, QIcon, QPalette, QColor, QPainter
from Classes.Utilities import (search_youtube, search_spotify, dominant_color_from_url, get_youtube_video_id, format_time)
from Classes.SignalsAndWorkers import Worker, SongFoundSignal
import os
import yt_dlp
import pythoncom
import vlc

class CustomSlider(QSlider):
    def __init__(self, player, *args, **kwargs):
        super(CustomSlider, self).__init__(*args, **kwargs)
        self._player = player

    def mousePressEvent(self, event):
        """Jump to click position."""
        fraction = event.x() / self.width()
        new_value = fraction * self.maximum()
        self.setValue(int(new_value))
        self._player.set_time(self.value())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Make the slider follow the cursor while dragging."""
        fraction = event.x() / self.width()
        new_value = fraction * self.maximum()
        self.setValue(int(new_value))
        self._player.set_time(self.value())
        super().mouseMoveEvent(event)

class AnimatedLabel(QWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.text = text
        self.offset = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.speed = 1
        self.delay = 50
        self.interval = 50
        self.timer.start(self.interval)

    def animate(self):
        if self.fontMetrics().width(self.text) > self.width():
            self.offset -= self.speed
            if abs(self.offset) > self.fontMetrics().width(self.text):
                self.offset = 0
                self.timer.setInterval(self.delay)
            else:
                self.timer.setInterval(self.interval)
        self.update()

    def setText(self, text):
        self.text = text
        self.offset = 0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawText(self.offset, 0, self.fontMetrics().width(self.text), self.height(), Qt.AlignVCenter | Qt.AlignLeft, self.text)
        if self.fontMetrics().width(self.text) > self.width():
            painter.drawText(self.offset + self.fontMetrics().width(self.text), 0, self.fontMetrics().width(self.text), self.height(), Qt.AlignVCenter | Qt.AlignLeft, self.text)

    def sizeHint(self):
        return QSize(self.fontMetrics().width(self.text), super().sizeHint().height())

