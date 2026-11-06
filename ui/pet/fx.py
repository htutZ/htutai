import random

from PySide2.QtCore import Qt
from PySide2.QtGui import QColor, QPainter, QPixmap

_tint_cache = {}


def tinted(pm, color):
    key = (pm.cacheKey(), color)
    t = _tint_cache.get(key)
    if t is None:
        t = QPixmap(pm.size())
        t.fill(Qt.transparent)
        p = QPainter(t)
        p.drawPixmap(0, 0, pm)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(t.rect(), QColor(color))
        p.end()
        if len(_tint_cache) > 400:
            _tint_cache.clear()
        _tint_cache[key] = t
    return t


def draw_fx(p, pm, x, y, fx):
    w, h = pm.width(), pm.height()
    c = fx.get("collapse", 0)
    if c > 0:
        p.save()
        p.translate(x + w / 2.0, y + h)
        p.scale(max(0.04, 1 - c), 1.0 + 0.15 * c)
        p.setOpacity(max(0.0, 1 - c * 0.6))
        p.drawPixmap(int(-w / 2), -h, pm)
        p.restore()
        return
    split = fx.get("split", 0)
    if fx.get("rim"):
        p.drawPixmap(x - 2, y, tinted(pm, "#7a3dff"))
    if split > 0:
        p.setOpacity(0.8)
        p.drawPixmap(int(x - split), y, tinted(pm, "#ff2060"))
        p.drawPixmap(int(x + split), y, tinted(pm, "#20e0ff"))
        p.setOpacity(1.0)
    g = fx.get("glitch", 0)
    if g > 0:
        rng = random.Random(fx.get("seed", 0))
        y0 = 0
        while y0 < h:
            bh = rng.randint(4, 14)
            off = int((rng.random() - 0.5) * 40 * g) if rng.random() < 0.6 * g else 0
            p.drawPixmap(x + off, y + y0, pm, 0, y0, w, min(bh, h - y0))
            y0 += bh
    else:
        p.drawPixmap(x, y, pm)