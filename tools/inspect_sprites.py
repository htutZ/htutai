# tools/inspect_sprites.py
import os, sys
from PySide2.QtGui import QImage
d = sys.argv[1] if len(sys.argv) > 1 else "assets/capri/sprites"
for f in sorted(os.listdir(d)):
    if f.endswith(".png"):
        im = QImage(os.path.join(d, f))
        print(f"{f:34s} {im.width():5d}x{im.height():<5d} frames~{max(1, round(im.width() / max(1, im.height())))}")