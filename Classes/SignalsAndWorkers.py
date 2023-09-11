from PySide2.QtCore import QRunnable, QObject, Signal
from PySide2.QtGui import QImage, QPixmap, QIcon, QPalette, QColor, QPainter

class Worker(QRunnable):
    def __init__(self, func, signal_instance, *args):
        super().__init__()
        self.func = func
        self.signal_instance = signal_instance
        self.args = args

    def run(self):
        self.func(self.signal_instance, *self.args)

class SongFoundSignal(QObject):
    song_found = Signal(str, QIcon)
