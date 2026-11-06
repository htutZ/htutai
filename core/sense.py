import time

from PySide2.QtCore import QObject, QTimer, Signal

try:
    import psutil
except ImportError:
    psutil = None


class Sense(QObject):
    event = Signal(str)   # hot, battery_low, plugged, unplugged

    def __init__(self):
        super().__init__()
        self.hot_since = self.last_hot = self.last_low = 0.0
        self.plugged = None
        if psutil:
            psutil.cpu_percent(interval=None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check)
        self.timer.start(5000)

    def check(self):
        if not psutil:
            return
        now = time.time()
        cpu = psutil.cpu_percent(interval=None)
        if cpu > 90:
            if not self.hot_since:
                self.hot_since = now
            elif now - self.hot_since > 30 and now - self.last_hot > 900:
                self.last_hot = now
                self.event.emit("hot")
        else:
            self.hot_since = 0
        try:
            b = psutil.sensors_battery()
        except Exception:
            b = None
        if b:
            if self.plugged is not None and b.power_plugged != self.plugged:
                self.event.emit("plugged" if b.power_plugged else "unplugged")
            self.plugged = b.power_plugged
            if not b.power_plugged and b.percent <= 20 and now - self.last_low > 1800:
                self.last_low = now
                self.event.emit("battery_low")