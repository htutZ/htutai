import html
import os
import random
import re
import subprocess
import sys
import unicodedata
import webbrowser
from collections import deque
from datetime import datetime

from PySide2.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                               QTextBrowser, QCheckBox, QComboBox, QLabel)
from PySide2.QtCore import QTimer, Signal

from core import personality as P
from core.brain import TABS
from core.intents import SITES, APPS
from core.llm import load_config, AIWorker
from core.myanglish import to_burmese
from core.persona import (build_system, parse_tags, load_memory, save_memory, forget,
                          strip_emoji, acceptable, DEFLECT)
from core.voice import Speaker, Listener

RE_MY = re.compile(r"[\u1000-\u109F]")
USE_LEGACY_CONVERTER = False   # True = also use your old improved_transliterate / myanglish_to_burmese


class ChatTab(QWidget):
    tab_requested = Signal(int)
    toast_requested = Signal(str)
    reminder_due = Signal(str)
    user_active = Signal()
    spoke = Signal(str, str)
    thinking = Signal()
    hint = Signal(str)
    pet_command = Signal(str)

    def __init__(self, brain, bond, detect_fn=None, convert_fn=None):
        super().__init__()
        self.brain, self.bond = brain, bond
        self.detect_fn, self.convert_fn = detect_fn, convert_fn
        self.messages, self.queue = [], deque()
        self.typing, self.pos = None, 0
        self.pending = False
        self.listener = None
        self.worker = None
        self.conv_fails = 0
        self._expect_my = False
        self.ai_cfg = load_config()
        self.history = load_memory()
        self._fallback_reply = None
        self.speaker = Speaker()
        self.speaker.failed.connect(self._status)

        self.log = QTextBrowser()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Talk to me... (Burmese, Myanglish or English)")
        self.mic_btn = QPushButton("Mic")
        self.send_btn = QPushButton("Send")
        self.voice_box = QCheckBox("Voice replies")
        self.voice_box.setChecked(True)
        self.ai_box = QCheckBox("Local AI")
        self.ai_box.setChecked(bool(self.ai_cfg.get("enabled", True)))
        self.conv_box = QCheckBox("Conversation mode")
        self.lang_box = QComboBox()
        self.lang_box.addItem("Listen: Burmese", "my-MM")
        self.lang_box.addItem("Listen: English", "en-US")
        self.status = QLabel("")

        row = QHBoxLayout()
        row.addWidget(self.input, 1)
        row.addWidget(self.mic_btn)
        row.addWidget(self.send_btn)
        opts = QHBoxLayout()
        opts.addWidget(self.voice_box)
        opts.addWidget(self.ai_box)
        opts.addWidget(self.conv_box)
        opts.addWidget(self.lang_box)
        opts.addWidget(self.status, 1)
        root = QVBoxLayout(self)
        root.addWidget(self.log, 1)
        root.addLayout(opts)
        root.addLayout(row)

        self.send_btn.clicked.connect(lambda: self._send_text())
        self.input.returnPressed.connect(lambda: self._send_text())
        self.input.textEdited.connect(lambda _: self.user_active.emit())
        self.mic_btn.clicked.connect(self.toggle_listen)
        self.voice_box.toggled.connect(self.speaker.set_enabled)
        self.ai_box.toggled.connect(lambda on: self.ai_cfg.update(enabled=on))

        self.type_timer = QTimer(self)
        self.type_timer.timeout.connect(self._type_step)
        self.poll = QTimer(self)
        self.poll.timeout.connect(self._poll)
        self.poll.start(500)

        h = datetime.now().hour
        if 5 <= h < 12:
            self.say("Good morning, choom! ငါ့ကို စကားပြောလို့ရပြီနော်။", "happy", "hello-morning")
        elif h < 5:
            self.say("Isn't it late, choom? ဒီလောက်ညနက်နေပြီ...", "tease", "hello-late")
        else:
            self.say("Miss me, choom? Mic ကိုနှိပ်ပြီး ပြောကြည့်ပါ။", "tease", "hello-miss")

    # ---------- input ----------
    def _ai_running(self):
        return bool(self.worker and self.worker.isRunning())

    def _send_text(self):
        if self._ai_running():
            self._status("ခဏစောင့်ပါ... စဉ်းစားနေတယ်")
            return
        t = self.input.text()
        self.input.clear()
        self.submit(t)

    def submit(self, text, voice=False):
        text = text.strip()
        if not text:
            return
        self.user_active.emit()
        self.messages.append(("you", ("[mic] " if voice else "") + text))
        self.render()
        self.pending = True
        self.thinking.emit()
        QTimer.singleShot(650, lambda: self._answer(text))

    def on_poke(self):
        self.user_active.emit()
        self.say(P.pick(P.POKE), "tease")

    # ---------- voice input ----------
    def toggle_listen(self):
        if self.listener and self.listener.isRunning():
            self.listener.cancel()
        else:
            self.start_listen()

    def start_listen(self):
        if self.listener and self.listener.isRunning():
            return
        self.speaker.stop()
        self.listener = Listener(self.lang_box.currentData())
        self.listener.status.connect(self._status)
        self.listener.status.connect(lambda s: self.hint.emit(strip_emoji(s)))
        self.listener.heard.connect(self._heard)
        self.listener.failed.connect(self._listen_failed)
        self.listener.finished.connect(lambda: self.mic_btn.setText("Mic"))
        self.mic_btn.setText("Stop")
        self.listener.start()

    def _heard(self, text):
        self.conv_fails = 0
        self.submit(text, voice=True)

    def _listen_failed(self, msg):
        self._status(msg)
        self.hint.emit(strip_emoji(msg))
        if self.conv_box.isChecked():
            self.conv_fails += 1
            if self.conv_fails >= 2:
                self.conv_box.setChecked(False)
                self.conv_fails = 0

    def _status(self, msg):
        self.status.setText(strip_emoji(msg))

    def _poll(self):
        if (self.conv_box.isChecked() and not self.pending and not self.typing and not self.queue
                and not self.speaker.busy() and not (self.listener and self.listener.isRunning())):
            self.start_listen()

    # ---------- answering ----------
    def _answer(self, raw):
        lang, converted = "en", raw
        if RE_MY.search(raw):
            lang = "my"
        quick = to_burmese(raw)
        if quick:
            lang, converted = "my", quick
        else:
            try:
                if self.detect_fn and self.detect_fn(raw) == "Myanglish":
                    lang = "my"
                    if USE_LEGACY_CONVERTER and self.convert_fn:
                        converted = self.convert_fn(raw)
            except Exception:
                pass
        self._expect_my = lang == "my"
        try:
            if self.bond.on_chat(raw):
                self.say(self.bond.line("levelup", stage=self.bond.stage_name), "happy")
            reply = self.brain.respond(raw, converted, lang)
        except Exception as e:
            self.pending = False
            self.say(f"Oops, something broke: {e}", "tease")
            return
        if reply.fallback and self.ai_box.isChecked():
            self._ask_ai(raw, converted, reply)
            return
        self.say(reply.text, reply.mood, reply.clip)
        self.pending = False
        self._do_action(reply.action)

    def _ask_ai(self, raw, converted, fallback_reply):
        self.history.append({"role": "user", "content": raw})
        msgs = [{"role": "system", "content": build_system(self.bond, style=self.ai_cfg.get("style"))}] \
            + self.history[-6:]
        if converted and converted != raw:
            msgs[-1] = {"role": "user",
                        "content": raw + f"\n(Myanglish converter's guess, may be wrong: {converted})"}
        self._fallback_reply = fallback_reply
        self._status("စဉ်းစားနေတယ်...")
        self.worker = AIWorker(self.ai_cfg, msgs)
        self.worker.done.connect(self._ai_done)
        self.worker.failed.connect(self._ai_failed)
        self.worker.start()

    def _ai_failed(self, msg):
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        self._status("AI: " + msg)
        self.hint.emit("AI: " + msg)
        r = self._fallback_reply
        self.say(r.text, r.mood)
        self.pending = False

    def _ai_done(self, text):
        self._status("")
        clean, acts, mood = parse_tags(text)
        if not acceptable(clean, self.history, self._expect_my):
            print("[ai] rejected reply:", clean[:80])
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            self.say(random.choice(DEFLECT), "tease")
            self.pending = False
            return
        self.history.append({"role": "assistant", "content": clean})
        save_memory(self.history)
        action = None
        for kind, val in acts:
            if kind == "pet":
                action = ("pet", val)
            elif kind == "open_name" and val in SITES:
                action = ("open", SITES[val])
            elif kind == "open_name" and val in APPS and sys.platform == "win32":
                action = ("open", APPS[val])
            elif kind == "tab_name" and val in TABS:
                action = ("tab", TABS[val])
            if action:
                break
        self.say(clean, mood or "talking")
        self.pending = False
        self._do_action(action)

    def _do_action(self, a):
        if not a:
            return
        kind = a[0]
        if kind == "tab":
            self.tab_requested.emit(a[1])
        elif kind == "remind":
            secs, what = a[1], a[2]
            QTimer.singleShot(secs * 1000, lambda: self._fire_reminder(what))
        elif kind == "pet":
            self.pet_command.emit(a[1])
        elif kind == "voice":
            self.voice_box.setChecked(a[1])
        elif kind == "open":
            self._open(a[1])
        elif kind == "forget":
            self.history = []
            forget()

    def _open(self, target):
        try:
            if target.startswith("http"):
                webbrowser.open(target)
            elif sys.platform == "win32":
                os.startfile(target)
            else:
                subprocess.Popen([target])
        except Exception as e:
            self.say(f"Couldn't open it: {e}", "tease")

    def _fire_reminder(self, what):
        self.reminder_due.emit(what)

    # ---------- speaking / typing ----------
    def say(self, text, mood="talking", clip=None):
        text = strip_emoji(text) or "..."
        self.spoke.emit(text, mood)
        if self.voice_box.isChecked():
            if not (clip and self.speaker.clip(clip)):
                self.speaker.speak(text)
        self.queue.append((text, mood))
        if self.typing is None:
            self._next()

    def _next(self):
        if not self.queue:
            self.typing = None
            return
        self.typing = self.queue.popleft()
        self.pos = 0
        self.type_timer.start(28)

    def _type_step(self):
        text, _mood = self.typing
        self.pos += 1
        while self.pos < len(text) and unicodedata.category(text[self.pos]).startswith("M"):
            self.pos += 1
        if self.pos >= len(text):
            self.type_timer.stop()
            self.messages.append(("bot", text))
            self.typing = None
            self.render()
            self._next()
            return
        self.render(partial=text[:self.pos])

    def render(self, partial=None):
        out = []
        for who, t in self.messages + ([("bot", partial)] if partial else []):
            t = html.escape(t).replace("\n", "<br>")
            if who == "you":
                out.append(f'<p align="right" style="color:#00f0ff; font-size:16px;"><b>You:</b> {t}</p>')
            else:
                out.append(f'<p style="color:#ff9bf0; font-size:18px;"><b>htut A.I:</b> {t}</p>')
        self.log.setHtml("".join(out))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, e):
        self.speaker.stop()
        if self.listener and self.listener.isRunning():
            self.listener.cancel()
            self.listener.wait(1500)
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(e)