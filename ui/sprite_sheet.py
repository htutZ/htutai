import os

from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import Qt

try:
    import numpy as np
except Exception:
    np = None

SCALE = 0.45
DEBUG = True

# Positive pushes her down into the window edge, negative lifts her up (screen pixels).
FOOT_ADJUST = {"perch": 50, "laydown": 47, "dangle": 0, "idle": 0, "walk": 0}
ALIAS = {"blink": "idle"}

FRAMES = {
    "idle": 16, "walk": 12, "arrive": 14, "dance": 16, "grin": 16, "sleep": 16,
    "wake": 13, "glitch": 10, "blink": 4, "dangle": 24, "leave": 12,
    "perch": 60, "perch_talk": 60, "perch_blink": 8,
    "laydown": 60, "laydown_sleep": 60, "laydown_talk": 60, "laydown_blink": 8,
     "perch_music": 60,
}


def _bottom_pad(img):
    if np is None:
        return 0
    try:
        im = img.convertToFormat(QImage.Format_RGBA8888)
        w, h, bpl = im.width(), im.height(), im.bytesPerLine()
        buf = np.frombuffer(im.constBits(), dtype=np.uint8, count=bpl * h)
        alpha = buf.reshape(h, bpl // 4, 4)[:, :w, 3]
        rows = np.where(alpha.max(axis=1) > 80)[0]
        return int(h - 1 - rows.max()) if len(rows) else 0
    except Exception:
        return 0


class Sheet:
    def __init__(self, path, count):
        self.frames, self.pad = [], 0
        img = QImage(path)
        if img.isNull() or count < 1:
            return
        fw, h = img.width() // count, img.height()
        self.pad = int(_bottom_pad(img) * SCALE)
        for i in range(count):
            fr = QPixmap.fromImage(img.copy(i * fw, 0, fw, h))
            self.frames.append(fr.scaled(max(1, int(fw * SCALE)), max(1, int(h * SCALE)),
                                         Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

    def __bool__(self):
        return bool(self.frames)


class Sheets:
    def __init__(self, folder):
        self.folder, self.cache, self.names = folder, {}, set()
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(".png") and f[:-4].replace("_flipped", "") in FRAMES:
                    self.names.add(f[:-4])

    def __contains__(self, name):
        return name in self.names

    def get(self, name, default=None):
        if name not in self.names:
            return default
        if name not in self.cache:
            sheet = Sheet(os.path.join(self.folder, name + ".png"), FRAMES[name.replace("_flipped", "")])
            self.cache[name] = sheet
            if DEBUG:
                print(f"[sprite] {name}: bottom pad {sheet.pad}px (scaled)")
        return self.cache[name] or default

    def preload(self, names):
        for n in names:
            self.get(n)

    def pad_for(self, name):
        fam = name.replace("_flipped", "").split("_")[0]
        s = self.get(ALIAS.get(fam, fam))
        return s.pad if s else 0


def load_sheets(folder):
    return Sheets(folder)