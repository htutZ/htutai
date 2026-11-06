import sys

from PySide2.QtCore import QRect
from PySide2.QtGui import QGuiApplication


class Ledge:
    def __init__(self, left, right, y, hwnd=None, floor=False):
        self.left, self.right, self.y, self.hwnd, self.floor = left, right, y, hwnd, floor


# Windows of this app she can sit on (like the mini player): key -> function returning a QRect or None
OWN_SEATS = {}
OWN_HWNDS = {}


def register_seat(key, getter, hwnd=None):
    OWN_SEATS[key] = getter
    if hwnd:
        OWN_HWNDS[key] = int(hwnd)


def unregister_seat(key):
    OWN_SEATS.pop(key, None)
    OWN_HWNDS.pop(key, None)


def own_hwnds():
    return set(OWN_HWNDS.values())


def open_spans(left, right, y, above):
    spans = [(left, right)]
    for c in above:
        if y < c.top() or y >= c.top() + c.height():
            continue
        cl, cr = c.left(), c.left() + c.width()
        nxt = []
        for a, b in spans:
            if cr <= a or cl >= b:
                nxt.append((a, b))
                continue
            if cl > a:
                nxt.append((a, cl))
            if cr < b:
                nxt.append((cr, b))
        spans = nxt
    return spans


SHELL = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "#32768",
         "tooltips_class32", "Windows.UI.Core.CoreWindow"}

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
    PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [PROC, wintypes.LPARAM]
    for fn in (user32.IsWindowVisible, user32.IsIconic, user32.IsZoomed):
        fn.argtypes = [wintypes.HWND]
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

    def _screen_of(r):
        s = QGuiApplication.screenAt(r.center()) or QGuiApplication.primaryScreen()
        return s.geometry()

    def visible_windows(skip, scale):
        out = []

        def cb(hwnd, _):
            if not hwnd or int(hwnd) in skip:
                return True
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            cloaked = wintypes.DWORD(0)
            dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), 4)
            if cloaked.value:
                return True
            ex = user32.GetWindowLongW(hwnd, -20)
            if ex & 0x20 or (ex & 0x08080000) == 0x08080000:
                return True
            buf = ctypes.create_unicode_buffer(128)
            user32.GetClassNameW(hwnd, buf, 128)
            if buf.value in SHELL:
                return True
            rc = wintypes.RECT()
            if dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rc), ctypes.sizeof(rc)) != 0:
                return True
            r = QRect(int(rc.left / scale), int(rc.top / scale),
                      int((rc.right - rc.left) / scale), int((rc.bottom - rc.top) / scale))
            if r.width() <= 0 or r.height() <= 0:
                return True
            seat = not (ex & 0x80 and not ex & 0x40000)
            seat = seat and not user32.IsZoomed(hwnd) and r.width() >= 120 and r.height() >= 60
            seat = seat and not r.contains(_screen_of(r))
            out.append((int(hwnd), r, seat))
            return True

        cbp = PROC(cb)
        user32.EnumWindows(cbp, 0)
        return out
else:
    def visible_windows(skip, scale):
        return []


def find_ledges(skip, scale):
    skip = set(skip) | own_hwnds()
    ledges, above = [], []

    for key, getter in list(OWN_SEATS.items()):
        try:
            r = getter()
        except Exception:
            r = None
        if r is not None and r.width() > 40 and r.height() > 20:
            ledges.append(Ledge(r.left(), r.left() + r.width(), r.top(), key))
            above.append(r)

    for hwnd, r, seat in visible_windows(skip, scale):
        if seat:
            for a, b in open_spans(r.left(), r.left() + r.width(), r.top(), above):
                if b - a >= 24:
                    ledges.append(Ledge(a, b, r.top(), hwnd))
        above.append(r)

    for s in QGuiApplication.screens():
        g = s.availableGeometry()
        ledges.append(Ledge(g.left(), g.left() + g.width(), g.top() + g.height(), None, True))
    return ledges