import sys, os, time, random
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from PySide2.QtWidgets import QApplication, QMainWindow, QTabWidget, QLabel, QSystemTrayIcon, QMenu
from PySide2.QtGui import QFontDatabase, QFont, QIcon
from PySide2.QtCore import QTimer

from language_detector import detect_language
from core.brain import Brain
from core.bond import Bond
from core.llm import load_config
from core.profile import get_name
from core.sense import Sense
from core.teases import TeasePicker
from core.watch import foreground_ai
from core import personality as P
from ui import clock_tab, calendar_tab, menu_tab, music_tab
from ui.chat_tab import ChatTab
from ui.intro import Intro
from ui.miniplayer import MiniPlayer
from ui.pet.card import Card
from ui.pet.pet import PetWindow
from ui.theme import QSS

try:
    from functions.load_responses import load_responses_from_database
    queries, responses, _ = load_responses_from_database()
except Exception as e:
    print("Database load failed, starting empty:", e)
    queries, responses = [], []

try:
    from functions.improved_transliterate import improved_transliterate
    from functions.myanglish_to_burmese import myanglish_to_burmese

    def convert(text):
        return myanglish_to_burmese(improved_transliterate(text))
except Exception:
    convert = None

MUSIC_TAB = 3
SENSE_LINES = {
    "hot": "ပရိုဆက်ဆာ ပူလာပြီ... ဘာတွေ run နေတာလဲ။",
    "battery_low": "ဘက်ထရီ နည်းနေပြီ၊ အားသွင်းပါဦး။",
    "plugged": "အားသွင်းနေပြီ၊ ကောင်းတယ်။",
    "unplugged": "အားမသွင်းတော့ဘူးနော်။",
}


class Toast(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("toast")
        self.setWordWrap(True)
        self.hide()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_text(self, text, ms=5000):
        self.setText(text)
        self.setMaximumWidth(360)
        self.adjustSize()
        p = self.parent()
        self.move(p.width() - self.width() - 24, p.height() - self.height() - 24)
        self.show()
        self.raise_()
        self.timer.start(ms)


class AssistantApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("htut A.I")
        self.setWindowIcon(QIcon("icon.png"))
        self.resize(1000, 680)

        cfg = load_config()
        self.teases_cfg = cfg.get("teases", True)
        self.jealous_on = cfg.get("jealous", True)
        self.intro_on = cfg.get("intro", True)
        self.quiet = False
        self.picker = TeasePicker()

        self.brain = Brain(queries, responses)
        self.bond = Bond()
        self.chat = ChatTab(self.brain, self.bond, detect_fn=detect_language, convert_fn=convert)
        self.music_widget = music_tab.create_music_tab()

        self.tabs = QTabWidget()
        for widget, icon, label in [
            (self.chat, "chat", "Chat"),
            (clock_tab.create_clock_tab(), "clock", "Clock"),
            (calendar_tab.create_calendar_tab(), "calendar", "Calendar"),
            (self.music_widget, "music", "Music"),
            (menu_tab.create_menu_tab(), "menu", "Menu"),
        ]:
            self.tabs.addTab(widget, QIcon(f"./icons/{icon}.png"), label)
        self.setCentralWidget(self.tabs)

        self.toast = Toast(self)
        self.pet = PetWindow(os.path.join(BASE, "assets", "capri", "sprites"))
        self.pet.hide()                      # she appears with the intro
        self.pet.is_speaking = self.chat.speaker.busy
        self.card = Card()
        self.card.pet = self.pet
        self.card.clicked.connect(self.on_card)
        self.mini = MiniPlayer(self.music_widget)
        self.mini.closed.connect(self.on_mini_closed)
        self.reminder = ""

        self.session_start = time.time()
        self.last_typed = 0.0
        self.last_wake_line = 0.0
        self.last_jeal_say = 0.0
        self.last_late = 0.0
        self.next_tease = time.time() + random.randint(120, 200)

        self.chat.tab_requested.connect(self.tabs.setCurrentIndex)
        self.chat.toast_requested.connect(self.on_toast)
        self.chat.reminder_due.connect(self.on_reminder)
        self.chat.spoke.connect(self.pet.say)
        self.chat.thinking.connect(self.pet.think)
        self.chat.hint.connect(self.pet.hint)
        self.chat.pet_command.connect(self.pet.command)
        self.chat.user_active.connect(self.on_typed)
        self.pet.poked.connect(self.chat.on_poke)
        self.pet.woke.connect(self.on_woke)
        self.pet.petted.connect(self.on_petted)
        self.pet.dragged.connect(self.on_dragged)
        self.pet.near_cursor.connect(lambda: self.chat.say(self.bond.line("near"), "tease"))

        events = getattr(self.music_widget, "events", None)
        if events:
            events.started.connect(self.on_song)
            events.paused.connect(self.on_song_pause)
            events.resumed.connect(lambda: setattr(self.pet, "music_on", True))
        else:
            self.tabs.currentChanged.connect(lambda i: setattr(self.pet, "music_on", i == MUSIC_TAB))

        self.sense = Sense()
        self.sense.event.connect(self.on_sense)

        self.tease_timer = QTimer(self)
        self.tease_timer.timeout.connect(self.check_tease)
        self.tease_timer.start(2000)
        self.jeal_timer = QTimer(self)
        self.jeal_timer.timeout.connect(self.check_jealous)
        self.jeal_timer.start(5000)

        self._make_tray()
        self._start_intro()

        if self.bond.gone_days >= 2:
            QTimer.singleShot(5000, lambda: self.chat.say(self.bond.line("back"), "tease"))

    # ---------- startup ----------
    def _start_intro(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.intro_on and self.bond.d.get("intro_day") != today:
            self.bond.d["intro_day"] = today
            self.bond.save()
            self.intro = Intro(get_name())
            self.intro.cue.connect(lambda: self.pet.command("show"))
            self.intro.start()
        else:
            self.pet.command("show")

    def _make_tray(self):
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(QIcon("icon.png"), self)
        self.tray_menu = QMenu()
        self.tray_menu.addAction("Show / hide her", self.toggle_pet)
        self.tray_menu.addAction("Dance", lambda: self.pet.command("dance"))
        self.tray_menu.addAction("Sleep", lambda: self.pet.command("sleep"))
        quiet = self.tray_menu.addAction("Quiet mode")
        quiet.setCheckable(True)
        quiet.toggled.connect(self.set_quiet)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction("Open chat window", self.show_window)
        self.tray_menu.addAction("Quit", self.close)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(lambda reason: self.show_window() if reason == QSystemTrayIcon.Trigger else None)
        self.tray.show()

    def show_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def toggle_pet(self):
        self.pet.command("hide" if self.pet.isVisible() else "show")

    def set_quiet(self, on):
        self.quiet = on

    # ---------- activity ----------
    def on_typed(self):
        self.last_typed = time.time()
        self.pet.last_active = self.last_typed

    def idle_for(self):
        return time.time() - max(self.pet.last_active, self.last_typed)

    def busy(self):
        return (self.chat.pending or self.chat.speaker.busy()
                or (self.chat.listener and self.chat.listener.isRunning()))

    # ---------- music ----------
    def on_song(self, title, pixmap):
        self.pet.music_on = True
        self.mini.show_song(title, pixmap)
        self.pet.sit_on(MiniPlayer.KEY)
        if not self.quiet and random.random() < 0.35:
            self.chat.say("ဒီသီချင်း ကြိုက်တယ်နော်။", "happy")

    def on_song_pause(self):
        self.pet.music_on = False

    def on_mini_closed(self):
        self.pet.stay_on = None
        self.pet.music_on = False
        if hasattr(self.music_widget, "stop"):
            self.music_widget.stop()

    # ---------- events ----------
    def on_toast(self, text):
        if not self.pet.isVisible():
            self.toast.show_text(text)

    def on_reminder(self, what):
        self.reminder = what
        self.pet.command("come")
        self.chat.say(f"သတိပေးချက်: {what}", "happy")
        self.card.show_card("Reminder", what, None, [("done", "Done"), ("later", "10 more min")], 0)
        if not self.pet.isVisible():
            self.toast.show_text("Reminder: " + what)

    def on_card(self, cid):
        if cid == "later" and self.reminder:
            what = self.reminder
            QTimer.singleShot(600000, lambda: self.chat._fire_reminder(what))
            self.chat.say("ကောင်းပြီ၊ ၁၀ မိနစ်နေရင် ပြန်သတိပေးမယ်။", "happy")

    def on_sense(self, kind):
        if self.quiet or kind not in SENSE_LINES:
            return
        if kind in ("hot", "battery_low"):
            self.pet.fx("glitch", 1.0)
        self.chat.say(SENSE_LINES[kind], "tease")

    def on_woke(self):
        now = time.time()
        if self.quiet or now - self.last_wake_line < 600 or random.random() < 0.5:
            return
        self.last_wake_line = now
        self.chat.say(*P.wake_clip())

    def on_petted(self):
        if self.bond.on_pet():
            self.chat.say(self.bond.line("levelup", stage=self.bond.stage_name), "happy")
        else:
            self.chat.say(self.bond.line("pet"), "happy")

    def on_dragged(self):
        if self.bond.on_thrown():
            self.pet.fx("split", 1.2)
            self.chat.say(self.bond.line("angry"), "tease")

    def check_jealous(self):
        if not self.jealous_on or self.quiet:
            return
        now = time.time()
        if now - self.last_jeal_say < 1200 or self.bond.stage < 1:
            return
        ai = foreground_ai()
        if ai and self.bond.can_be_jealous():
            self.last_jeal_say = now
            self.bond.on_jealous()
            key = "jealous_high" if self.bond.stage >= 3 else "jealous_low"
            self.chat.say(self.bond.line(key, ai=ai), "tease")

    def check_tease(self):
        if not self.teases_cfg or self.quiet or self.busy():
            return
        now = time.time()
        hour = datetime.now().hour
        if hour in (1, 2, 3, 4) and now - self.last_late > 1800 and self.idle_for() < 30:
            self.last_late = now
            self.next_tease = now + 300
            self.chat.say("Isn't it late, choom?", "tease", "hello-late")
            return
        if now < self.next_tease or self.pet.mode == "sleep":
            return
        idle = self.idle_for()
        if idle < 60 or idle > 170:
            return
        self.next_tease = now + random.randint(240, 480)
        if random.random() < 0.15:
            self.chat.say("Miss me, choom?", "tease", "hello-miss")
            return
        kind = "long" if (now - self.session_start > 7200 and random.random() < 0.4) else "idle"
        line = self.picker.pick(kind, self.bond.stage)
        self.chat.say(line, "tease")

    def closeEvent(self, e):
        self.chat.close()
        if hasattr(self.music_widget, "shutdown"):
            self.music_widget.shutdown()
        self.mini.close()
        self.pet.close()
        self.card.close()
        if self.tray:
            self.tray.hide()
        self.bond.save()
        super().closeEvent(e)
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont("./icons/Pyidaungsu.ttf")
    app.setFont(QFont("Pyidaungsu", 11))
    app.setStyleSheet(QSS)
    win = AssistantApp()
    win.show()
    sys.exit(app.exec_())