import random
import time

from PySide2.QtWidgets import QWidget, QLabel, QMenu, QApplication
from PySide2.QtGui import QPainter, QTransform, QColor, QPen, QCursor
from PySide2.QtCore import Qt, QTimer, QRectF, Signal

from ui.sprite_sheet import load_sheets, FOOT_ADJUST
from ui.pet.ledges import find_ledges, own_hwnds
from ui.pet.fx import draw_fx
from core import personality as P

SIZE = 190
EXTRA = 100               # extra window height below her so lowered poses aren't clipped
BASE_FACES = "right"      # set to "left" if she walks backwards
GRAVITY = 1800
WALK_SPEED = 70
MARGIN = 20
SLEEP_AFTER = 180
NEAR_COOLDOWN = 300       # seconds between cursor teases
RIM = False               # True = violet outline behind her (Capri's misregistration look)
ONCE = {"arrive", "wake", "leave"}


def fps_for(name):
    n = name.replace("_flipped", "")
    if "blink" in n:
        return 24
    if n.startswith(("perch", "laydown")):
        return 30
    if n == "dangle":
        return 15
    if n in ("idle", "sleep"):
        return 8
    return 12


class Bubble(QWidget):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color:#e6faff; background:transparent; font-size:15px;")
        self.label.move(14, 10)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_text(self, text, ms):
        w = min(280, self.label.fontMetrics().horizontalAdvance(text) + 6)
        self.label.setFixedWidth(w)
        self.label.setText(text)
        self.label.setFixedHeight(self.label.heightForWidth(w))
        self.resize(w + 28, self.label.height() + 34)
        self.show()
        self.timer.start(ms)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(16, 22, 38, 238))
        p.setPen(QPen(QColor("#ff2bd6"), 2))
        p.drawRoundedRect(QRectF(2, 2, self.width() - 4, self.height() - 16), 12, 12)


class PetWindow(QWidget):
    poked = Signal()
    woke = Signal()
    petted = Signal()
    dragged = Signal()
    near_cursor = Signal()

    def __init__(self, sprite_dir):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.resize(SIZE, SIZE + EXTRA)

        self.sheets = load_sheets(sprite_dir)
        self.sheets.preload(["idle", "walk", "arrive", "blink", "wake", "grin"])
        self.bubble = Bubble()
        self.scale = QApplication.primaryScreen().devicePixelRatio()
        self.is_speaking = lambda: False
        self.music_on = False
        self.stay_on = None          # key of an own-window seat (the mini player) she should sit on
        self.stay_since = 0.0
        self.last_drop = 0.0

        self.ledges, self.ledge = [], None
        self.scan_t = 0
        self.facing, self.dir = "right", 1
        self.speed = WALK_SPEED
        self.target = 0
        self.vx = self.vy = 0.0
        self.dragging = self.moved = False
        self.gx = self.gy = 0
        self.press = None
        self.talk_until = 0
        self.force_sleep_until = 0
        self.rest_pose = "perch"
        self.hold = False
        self.pet_dir, self.pet_flips, self.last_mx = 0, [], None
        self.last_pet = 0
        self.near_since, self.last_near = 0, time.time()
        self.fx_kind, self.fx_t0, self.fx_dur = None, 0, 1
        self.blink_until, self.next_blink = 0, time.time() + 3
        self.last_cursor, self.last_active = QCursor.pos(), time.time()
        self.anim_name, self.anim_t0 = "", time.time()
        self.t_last = time.time()

        self._scan()
        floor = [l for l in self.ledges if l.floor][0]
        self.x = random.uniform(floor.left + 200, max(floor.left + 201, floor.right - 200))
        self.y = floor.y
        self.ledge = floor
        self._set("arrive", 1.4)
        self.fx("teleport_in", 0.7)
        self._place()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    # ---------- public ----------
    def fx(self, kind, dur=0.8):
        self.fx_kind, self.fx_t0, self.fx_dur = kind, time.time(), dur

    def say(self, text, mood="talking"):
        now = time.time()
        self.talk_until = now + min(8, 1 + len(text) * 0.05)
        self.bubble.show_text(text, int(min(9000, 2500 + len(text) * 70)))
        if mood == "tease" and self.mode in ("idle", "rest"):
            self._set("grin", 1.6)
        elif mood == "happy" and self.mode in ("idle", "rest"):
            self._set("dance", 2.5)
        self._place_bubble()

    def hint(self, text):
        self.bubble.show_text(text, 3500)
        self._place_bubble()

    def think(self):
        self.fx("glitch", 0.7)
        if self.mode in ("idle", "rest", "walk"):
            self._set("glitch", 0.9)

    def sit_on(self, key):
        self.stay_on = key
        self.stay_since = time.time()
        self.last_drop = 0

    def command(self, name):
        now = time.time()
        if name == "show":
            self.show()
            self.fx("teleport_in", 0.6)
            return
        if name == "hide":
            self._set("leave", 1.0)
            self.fx("teleport_out", 0.5)
            QTimer.singleShot(1050, self._do_hide)
            return
        if self.mode == "fall" or self.ledge is None:
            return
        if name == "dance":
            self._set("dance", 4)
        elif name == "sleep":
            self.force_sleep_until = now + 30
            self._set("sleep")
        elif name == "rest":
            self._start_rest(None, 15)
        elif name in ("sit", "lie", "dangle"):
            self._start_rest({"sit": "perch", "lie": "laydown", "dangle": "dangle"}[name], 90)
        elif name == "glitch":
            self._set("glitch", 1.0)
            self.fx("glitch", 1.0)
        elif name == "walk":
            self._decide_walk()
        elif name == "come":
            self._walk_to(QCursor.pos().x(), fast=True)

    def _do_hide(self):
        self.bubble.hide()
        self.hide()

    def _nudge(self, d):
        fam = self.sprite_for(time.time()).replace("_flipped", "").split("_")[0]
        FOOT_ADJUST[fam] = FOOT_ADJUST.get(fam, 0) + d
        print(f'[nudge] FOOT_ADJUST["{fam}"] = {FOOT_ADJUST[fam]}')

    def _toggle_hold(self, on):
        self.hold = on

    # ---------- state helpers ----------
    def _set(self, mode, duration=None):
        self.mode = mode
        self.mode_end = time.time() + (duration if duration else 1e9)
        self.speed = WALK_SPEED

    def _start_rest(self, pose, duration):
        if pose is None:
            win = self.ledge is not None and not self.ledge.floor
            if win:
                near_edge = min(self.x - self.ledge.left, self.ledge.right - self.x) < 70
                pose = "dangle" if near_edge else random.choice(["laydown", "perch"])
            else:
                pose = "perch"
        self.rest_pose = pose
        self._set("rest", duration)

    def _walk_to(self, x, fast=False):
        l = self.ledge
        self.target = min(max(x, l.left + MARGIN), l.right - MARGIN)
        if abs(self.target - self.x) < 25:
            self._set("grin", 1.5)
            return
        self.dir = 1 if self.target > self.x else -1
        self._set("walk")
        if fast:
            self.speed = 140

    def _decide_walk(self):
        l = self.ledge
        lo, hi = l.left + MARGIN + SIZE // 4, l.right - MARGIN - SIZE // 4
        if hi - lo < 80:
            self._set("idle", 3)
            return
        if not l.floor and random.random() < 0.25:
            self.target = (l.right + SIZE) if random.random() < 0.5 else (l.left - SIZE)
        else:
            self.target = random.uniform(lo, hi)
        self.dir = 1 if self.target > self.x else -1
        self._set("walk")

    # ---------- world / ledges ----------
    def _scan(self):
        skip = {int(self.winId()), int(self.bubble.winId())} | own_hwnds()
        self.ledges = find_ledges(skip, self.scale)
        self.scan_t = time.time()

    def _floor_at(self, x):
        floors = [l for l in self.ledges if l.floor]
        for l in floors:
            if l.left <= x <= l.right:
                return l
        return floors[0]

    def _stick(self):
        if self.ledge is None:
            return False
        if self.ledge.floor:
            f = self._floor_at(self.x)
            self.ledge, self.y = f, f.y
            return True
        cand = [l for l in self.ledges
                if l.hwnd == self.ledge.hwnd and l.left - 10 <= self.x <= l.right + 10]
        if cand:
            self.ledge, self.y = cand[0], cand[0].y
            return True
        self._start_fall(self.dir * 40 if self.mode == "walk" else 0)
        return False

    def _start_fall(self, vx=0):
        self.mode, self.mode_end = "fall", 1e12
        self.vx, self.vy, self.ledge = vx, 0.0, None

    def _watch_cursor(self, now):
        pos = QCursor.pos()
        if pos != self.last_cursor:
            self.last_cursor, self.last_active = pos, now

    # ---------- mini player seat ----------
    def _seat_logic(self, now):
        key = self.stay_on
        if key is None or self.dragging or not self.isVisible() or self.mode in ("fall", "leave", "arrive"):
            return
        if self.mode == "sleep":
            self._set("wake", 1.0)
            return
        seat = next((l for l in self.ledges if l.hwnd == key), None)
        if seat is None:
            if now - self.stay_since > 4:
                self.stay_on = None
            return
        on = self.ledge is not None and self.ledge.hwnd == key
        if not on:
            if now - self.last_drop < 3:
                return
            self.last_drop = now
            self.x = seat.left + (seat.right - seat.left) * random.uniform(0.3, 0.8)
            self.y = min(max(SIZE, seat.y - 260), seat.y - 5)
            self._start_fall(0)
            self.fx("teleport_in", 0.5)
        elif self.mode in ("idle", "grin", "glitch") and now > self.mode_end:
            self._start_rest("perch", 3600)

    # ---------- teasing ----------
    def _cursor_tease(self, now):
        if self.mode not in ("idle", "rest", "walk"):
            self.near_since = 0
            return
        c = QCursor.pos()
        near = abs(c.x() - self.x) < 150 and abs(c.y() - (self.y - 60)) < 150
        if not near:
            self.near_since = 0
            return
        if not self.near_since:
            self.near_since = now
            return
        if now - self.near_since >= 2.5 and now - self.last_near > NEAR_COOLDOWN:
            self.last_near, self.near_since = now, 0
            self._set("grin", 1.8)
            self.near_cursor.emit()

    def _pet_check(self, mx):
        now = time.time()
        if self.last_mx is None:
            self.last_mx = mx
            return
        if abs(mx - self.last_mx) < 6:
            return
        d = 1 if mx > self.last_mx else -1
        if self.pet_dir and d != self.pet_dir:
            self.pet_flips = [t for t in self.pet_flips if now - t < 2.0] + [now]
            if len(self.pet_flips) >= 6 and now - self.last_pet > 8:
                self.last_pet, self.pet_flips = now, []
                self._set("grin", 2.5)
                self.petted.emit()
        self.pet_dir, self.last_mx = d, mx

    # ---------- behaviour ----------
    def _decide(self, now):
        if now - self.last_active > SLEEP_AFTER and self.stay_on is None:
            self._set("sleep")
            return
        r = random.random()
        if r < 0.40:
            self._decide_walk()
        elif r < 0.65:
            self._start_rest(None, random.uniform(8, 20))
        elif r < 0.75:
            self._set("dance", 3)
        elif r < 0.85:
            self._set("grin", 2)
        elif r < 0.90:
            self._set("glitch", 0.7)
            self.fx("glitch", 0.7)
        else:
            self._set("idle", random.uniform(3, 6))

    def _tick(self):
        now = time.time()
        dt = min(0.05, now - self.t_last)
        self.t_last = now
        self._watch_cursor(now)
        self._cursor_tease(now)
        self._seat_logic(now)
        if now - self.scan_t > 0.4:
            self._scan()
        if now > self.next_blink:
            self.blink_until, self.next_blink = now + 0.18, now + random.uniform(2.5, 6)
        if not self.dragging:
            getattr(self, "_do_" + self.mode)(now, dt)
        self._place()
        self.update()
        if self.bubble.isVisible():
            self._place_bubble()

    def _timed(self, now):
        if self._stick() and now > self.mode_end:
            self._set("idle", random.uniform(2, 5))

    def _do_arrive(self, now, dt): self._timed(now)
    def _do_wake(self, now, dt):   self._timed(now)
    def _do_dance(self, now, dt):  self._timed(now)
    def _do_grin(self, now, dt):   self._timed(now)
    def _do_glitch(self, now, dt): self._timed(now)
    def _do_leave(self, now, dt):  self._timed(now)

    def _do_idle(self, now, dt):
        if self._stick() and now > self.mode_end and not self.hold:
            self._decide(now)

    def _do_rest(self, now, dt):
        if self._stick() and now > self.mode_end and not self.hold:
            self._decide(now)

    def _do_sleep(self, now, dt):
        if not self._stick():
            return
        if now < self.force_sleep_until:
            return
        if now - self.last_active < 1.0:
            self._set("wake", 1.2)
            self.woke.emit()

    def _do_walk(self, now, dt):
        if not self._stick():
            return
        self.x += self.dir * self.speed * dt
        self.facing = "right" if self.dir > 0 else "left"
        if (self.dir > 0 and self.x >= self.target) or (self.dir < 0 and self.x <= self.target):
            self._set("idle", random.uniform(2, 5))

    def _do_fall(self, now, dt):
        self.vy += GRAVITY * dt
        ny = self.y + self.vy * dt
        self.x += self.vx * dt
        land = None
        for l in self.ledges:
            if l.left <= self.x <= l.right and self.y - 1 <= l.y <= ny:
                if land is None or l.y < land.y:
                    land = l
        if land is None:
            f = self._floor_at(self.x)
            if ny >= f.y:
                land = f
        if land:
            self.ledge, self.y, self.vx, self.vy = land, land.y, 0, 0
            self._set("idle", 1.5)
        else:
            self.y = ny

    # ---------- sprites ----------
    def _speaking(self, now):
        try:
            return now < self.talk_until or bool(self.is_speaking())
        except Exception:
            return now < self.talk_until

    def sprite_for(self, now):
        win = self.ledge is not None and not self.ledge.floor
        m = self.mode
        if m == "rest":
            name = self.rest_pose
        elif m == "idle":
            name = "perch" if win else "idle"
        elif m == "sleep":
            name = "laydown_sleep" if win else "sleep"
        elif m == "fall":
            name = "dangle"
        else:
            name = m
        if name == "perch" and self.music_on and win and "perch_music" in self.sheets:
            name = "perch_music"
        if self._speaking(now) and name + "_talk" in self.sheets:
            name += "_talk"
        elif now < self.blink_until:
            b = "blink" if name == "idle" else name + "_blink"
            if b in self.sheets:
                name = b
        return name

    def _fx_state(self, now):
        if not self.fx_kind:
            return None
        t = (now - self.fx_t0) / self.fx_dur
        if t >= 1:
            self.fx_kind = None
            return None
        seed = int(now * 18)
        k = self.fx_kind
        if k == "glitch":
            return {"glitch": 1 - t, "seed": seed, "rim": RIM}
        if k == "split":
            return {"split": 6 * (1 - t), "glitch": 0.5 * (1 - t), "seed": seed, "rim": RIM}
        if k == "teleport_in":
            return {"collapse": 1 - t}
        if k == "teleport_out":
            return {"collapse": t}
        return None

    def paintEvent(self, _):
        now = time.time()
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        name = self.sprite_for(now)
        mirror = False
        if self.facing != BASE_FACES:
            if name + "_flipped" in self.sheets:
                name += "_flipped"
            else:
                mirror = True
        sheet = self.sheets.get(name) or self.sheets.get("idle")
        if not sheet:
            p.setBrush(QColor("#00f0ff"))
            p.drawEllipse(40, 40, 90, 120)
            return
        if name != self.anim_name:
            self.anim_name, self.anim_t0 = name, now
        n = len(sheet.frames)
        i = int((now - self.anim_t0) * fps_for(name))
        i = min(i, n - 1) if name.split("_")[0] in ONCE else i % n
        pm = sheet.frames[i]
        if mirror:
            pm = pm.transformed(QTransform().scale(-1, 1))
        fam = name.replace("_flipped", "").split("_")[0]
        drop = self.sheets.pad_for(name) + FOOT_ADJUST.get(fam, 0)
        x, y = (SIZE - pm.width()) // 2, SIZE - pm.height() + drop
        state = self._fx_state(now)
        if state:
            draw_fx(p, pm, x, y, state)
        elif RIM:
            draw_fx(p, pm, x, y, {"rim": True})
        else:
            p.drawPixmap(x, y, pm)

    # ---------- placement ----------
    def _place(self):
        self.move(int(self.x - SIZE / 2), int(self.y - SIZE))

    def _place_bubble(self):
        g = self.geometry()
        self.bubble.move(g.center().x() - self.bubble.width() // 2,
                         g.top() - self.bubble.height() + 20)

    # ---------- mouse ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging, self.moved = True, False
            self.gx = e.globalPos().x() - self.x
            self.gy = e.globalPos().y() - self.y
            self.press = e.globalPos()

    def mouseMoveEvent(self, e):
        if not self.dragging:
            self._pet_check(e.pos().x())
            return
        if (e.globalPos() - self.press).manhattanLength() > 5:
            self.moved = True
        if self.moved:
            self.x = e.globalPos().x() - self.gx
            self.y = e.globalPos().y() - self.gy
            self.last_active = time.time()

    def mouseReleaseEvent(self, e):
        if not self.dragging:
            return
        self.dragging = False
        self.last_active = time.time()
        if self.moved:
            self._start_fall()
            self.dragged.emit()
        else:
            self._set("grin", 1.5)
            self.poked.emit()

    def contextMenuEvent(self, e):
        m = QMenu(self)
        m.addAction("Dance", lambda: self.command("dance"))
        m.addAction("Sleep", lambda: self.command("sleep"))
        m.addAction("Say something", lambda: self.say(P.idle_tease(), "tease"))
        m.addSeparator()
        m.addAction("Pose: sit", lambda: self.command("sit"))
        m.addAction("Pose: lie down", lambda: self.command("lie"))
        m.addAction("Pose: dangle legs", lambda: self.command("dangle"))
        hold = m.addAction("Hold pose (for tuning)")
        hold.setCheckable(True)
        hold.setChecked(self.hold)
        hold.toggled.connect(self._toggle_hold)
        m.addAction("Lower her (+2)", lambda: self._nudge(2))
        m.addAction("Raise her (-2)", lambda: self._nudge(-2))
        m.addSeparator()
        m.addAction("Glitch", lambda: self.command("glitch"))
        m.addAction("Hide", lambda: self.command("hide"))
        m.exec_(e.globalPos())

    def closeEvent(self, e):
        self.bubble.close()
        super().closeEvent(e)