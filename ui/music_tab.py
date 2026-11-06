import json
import os
import shutil
import subprocess
import urllib.request

from PySide2.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                               QListWidget, QListWidgetItem, QLabel, QSlider, QComboBox)
from PySide2.QtCore import Qt, QThread, QTimer, QObject, Signal, QSize
from PySide2.QtGui import QPixmap, QIcon

try:
    import vlc
    VLC_ERROR = ""
except Exception as e:
    vlc = None
    VLC_ERROR = str(e)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SONG_DIR = os.path.join(HERE, "songs")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
EXE_NAME = "yt-dlp.exe" if os.name == "nt" else "yt-dlp"


def find_ytdlp():
    local = os.path.join(PROJECT, "tools", EXE_NAME)
    if os.path.exists(local):
        return local
    return shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")


YTDLP = find_ytdlp()
HINT = "Put yt-dlp.exe in the tools folder, run:  tools\\yt-dlp.exe --update-to nightly   and install Deno (winget install DenoLand.Deno)"


def run_ytdlp(args, timeout=60):
    if not YTDLP:
        return 1, "", "ERROR: yt-dlp.exe not found"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        p = subprocess.run([YTDLP] + args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=env, creationflags=flags)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "ERROR: yt-dlp timed out"
    except Exception as e:
        return 1, "", f"ERROR: {e}"


def short_error(err):
    lines = [l for l in (err or "").splitlines() if "ERROR" in l]
    return (lines[-1] if lines else (err or "unknown error").strip()[-200:])[:200]


def fmt_ms(ms):
    s = max(0, int(ms)) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def fetch(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


class MusicEvents(QObject):
    started = Signal(str, QPixmap)
    paused = Signal()
    resumed = Signal()


class SearchWorker(QThread):
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        code, out, err = run_ytdlp(["--no-warnings", "--flat-playlist", "-J", f"ytsearch10:{self.query}"], 45)
        if code != 0 or not out.strip():
            print("[music] search error:", err[-300:])
            self.failed.emit(short_error(err))
            return
        try:
            data = json.loads(out)
            items = []
            for e in data.get("entries") or []:
                if not e or not e.get("id"):
                    continue
                items.append({"id": e["id"], "title": e.get("title") or "Unknown",
                              "channel": e.get("channel") or e.get("uploader") or "",
                              "duration": int(e.get("duration") or 0)})
            self.done.emit(items)
        except Exception as ex:
            self.failed.emit(f"bad search result: {ex}")


class ThumbWorker(QThread):
    got = Signal(int, bytes)

    def __init__(self, items):
        super().__init__()
        self.items = items
        self.cancelled = False

    def run(self):
        for row, vid in self.items:
            if self.cancelled:
                return
            data = fetch(f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg")
            if data:
                self.got.emit(row, data)


class StreamWorker(QThread):
    ready = Signal(int, dict)
    failed = Signal(int, str)

    def __init__(self, token, vid):
        super().__init__()
        self.token, self.vid = token, vid

    def run(self):
        url = f"https://www.youtube.com/watch?v={self.vid}"
        base = ["--no-warnings", "--no-playlist", "-f", "bestaudio/best", "-j"]
        attempts = [
            [],
            ["--extractor-args", "youtube:player_client=default,-tv_downgraded,web_embedded"],
            ["--extractor-args", "youtube:player_client=web_safari,web_embedded,-tv_downgraded"],
        ]
        last = ""
        for extra in attempts:
            code, out, err = run_ytdlp(base + extra + [url], 60)
            if code == 0 and out.strip():
                try:
                    info = json.loads(out.strip().splitlines()[0])
                    stream = info.get("url")
                    if not stream:
                        for f in reversed(info.get("formats") or []):
                            if f.get("url") and f.get("acodec") not in (None, "none"):
                                stream = f["url"]
                                break
                    if stream:
                        data = fetch(f"https://i.ytimg.com/vi/{self.vid}/mqdefault.jpg") or b""
                        self.ready.emit(self.token, {
                            "url": stream, "title": info.get("title") or "Unknown",
                            "duration": int(info.get("duration") or 0),
                            "headers": dict(info.get("http_headers") or {}), "thumb": data})
                        return
                except Exception as ex:
                    err = f"ERROR: bad stream info: {ex}"
            last = err or last
            print("[music] yt-dlp attempt failed:", (err or "")[-300:])
        print("[music]", HINT)
        self.failed.emit(self.token, short_error(last))


class MusicTab(QWidget):
    def __init__(self):
        super().__init__()
        self.events = MusicEvents()
        self.workers = []
        self.thumb_worker = None
        self.token = 0
        self.seeking = False
        self.current_row = -1
        self.total_ms = 0
        self.err_shown = False
        self.cur_title = ""
        self.cur_pm = None
        self.instance = self.player = None

        if vlc:
            try:
                self.instance = vlc.Instance("--no-video", "--quiet")
                self.player = self.instance.media_player_new()
                self.player.audio_set_volume(60)
            except Exception as e:
                print("[music] VLC init failed:", e)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search for songs...")
        self.search_btn = QPushButton("Search")
        self.local_btn = QPushButton("Local songs")
        top = QHBoxLayout()
        top.addWidget(self.search_bar, 1)
        top.addWidget(self.search_btn)
        top.addWidget(self.local_btn)

        self.listw = QListWidget()
        self.listw.setIconSize(QSize(64, 36))

        self.cover = QLabel()
        self.cover.setFixedSize(56, 56)
        self.cover.setScaledContents(True)
        self.now = QLabel("Nothing playing")
        self.now.setWordWrap(True)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        info = QHBoxLayout()
        info.addWidget(self.cover)
        col = QVBoxLayout()
        col.addWidget(self.now)
        col.addWidget(self.status)
        info.addLayout(col, 1)

        self.prev_btn = QPushButton("Prev")
        self.play_btn = QPushButton("Play")
        self.next_btn = QPushButton("Next")
        self.loop = QComboBox()
        self.loop.addItems(["No loop", "Repeat one", "Repeat all"])
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(60)
        self.vol.setFixedWidth(110)
        controls = QHBoxLayout()
        for w in (self.prev_btn, self.play_btn, self.next_btn, self.loop):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(QLabel("Volume"))
        controls.addWidget(self.vol)

        self.cur_lbl = QLabel("00:00")
        self.total_lbl = QLabel("00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1)
        self.slider.setPageStep(10000)
        prog = QHBoxLayout()
        prog.addWidget(self.cur_lbl)
        prog.addWidget(self.slider, 1)
        prog.addWidget(self.total_lbl)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.listw, 1)
        root.addLayout(info)
        root.addLayout(controls)
        root.addLayout(prog)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.search_btn.clicked.connect(self.search)
        self.search_bar.returnPressed.connect(self.search)
        self.local_btn.clicked.connect(self.load_local)
        self.listw.itemDoubleClicked.connect(lambda it: self.play_row(self.listw.row(it)))
        self.play_btn.clicked.connect(self.toggle)
        self.next_btn.clicked.connect(lambda: self.next_song(False))
        self.prev_btn.clicked.connect(self.prev_song)
        self.vol.valueChanged.connect(lambda v: self.player and self.player.audio_set_volume(v))
        self.slider.sliderPressed.connect(lambda: setattr(self, "seeking", True))
        self.slider.sliderReleased.connect(self._seek_release)
        self.slider.actionTriggered.connect(lambda _a: QTimer.singleShot(0, self._seek_now))

        if not self.player:
            self.status.setText(f"VLC not available: {VLC_ERROR or 'install VLC (same 32/64-bit as Python)'}")
            for b in (self.play_btn, self.next_btn, self.prev_btn):
                b.setEnabled(False)
        elif not YTDLP:
            self.status.setText("yt-dlp.exe not found. " + HINT)
        else:
            self.status.setText("Type a song name and press Search.")
        if os.path.isdir(SONG_DIR) and any(f.lower().endswith(AUDIO_EXT) for f in os.listdir(SONG_DIR)):
            self.load_local()

    # ---------- helpers ----------
    def _keep(self, w):
        self.workers.append(w)
        w.finished.connect(lambda w=w: self._drop(w))

    def _drop(self, w):
        if w in self.workers:
            self.workers.remove(w)
        w.deleteLater()

    # ---------- list ----------
    def load_local(self):
        self.listw.clear()
        if not os.path.isdir(SONG_DIR):
            os.makedirs(SONG_DIR, exist_ok=True)
        for f in sorted(os.listdir(SONG_DIR)):
            if f.lower().endswith(AUDIO_EXT):
                item = QListWidgetItem(f"{os.path.splitext(f)[0]}\nLocal file")
                item.setData(Qt.UserRole, {"kind": "file", "path": os.path.join(SONG_DIR, f),
                                           "title": os.path.splitext(f)[0]})
                self.listw.addItem(item)
        self.status.setText(f"{self.listw.count()} local songs. Double-click to play.")

    def search(self):
        q = self.search_bar.text().strip()
        if not q:
            return
        if not YTDLP:
            self.status.setText("yt-dlp.exe not found. " + HINT)
            return
        if self.thumb_worker:
            self.thumb_worker.cancelled = True
        self.listw.clear()
        self.status.setText("Searching...")
        w = SearchWorker(q)
        w.done.connect(self._on_results)
        w.failed.connect(lambda m: self.status.setText("Search failed: " + m))
        self._keep(w)
        w.start()

    def _on_results(self, items):
        self.listw.clear()
        for it in items:
            dur = fmt_ms(it["duration"] * 1000) if it["duration"] else ""
            item = QListWidgetItem(f"{it['title']}\n{it['channel']}   {dur}")
            item.setData(Qt.UserRole, dict(it, kind="yt"))
            self.listw.addItem(item)
        self.status.setText(f"{len(items)} results. Double-click to play." if items else "No results.")
        tw = ThumbWorker([(i, it["id"]) for i, it in enumerate(items)])
        tw.got.connect(self._set_thumb)
        self.thumb_worker = tw
        self._keep(tw)
        tw.start()

    def _set_thumb(self, row, data):
        pm = QPixmap()
        pm.loadFromData(data)
        item = self.listw.item(row)
        if item and not pm.isNull():
            item.setIcon(QIcon(pm))

    # ---------- playback ----------
    def play_row(self, row):
        if not self.player or row < 0 or row >= self.listw.count():
            return
        self.current_row = row
        self.listw.setCurrentRow(row)
        data = self.listw.item(row).data(Qt.UserRole)
        self.token += 1
        if data["kind"] == "file":
            self._start(data["path"], data["title"], None, {}, 0, True)
            return
        self.status.setText("Loading...")
        w = StreamWorker(self.token, data["id"])
        w.ready.connect(self._on_stream)
        w.failed.connect(self._on_stream_fail)
        self._keep(w)
        w.start()

    def _on_stream(self, token, info):
        if token != self.token:
            return
        pm = QPixmap()
        if info["thumb"]:
            pm.loadFromData(info["thumb"])
        self._start(info["url"], info["title"], pm, info["headers"], info["duration"], False)

    def _on_stream_fail(self, token, msg):
        if token != self.token:
            return
        self.status.setText("Couldn't play: " + msg)

    def _start(self, src, title, pm, headers, duration, local):
        try:
            media = self.instance.media_new_path(src) if local else self.instance.media_new(src)
            ua = (headers or {}).get("User-Agent")
            if ua:
                media.add_option(f":http-user-agent={ua}")
            self.player.set_media(media)
            self.player.play()
        except Exception as e:
            self.status.setText(f"VLC error: {e}")
            return
        self.total_ms = duration * 1000
        self.cur_title = title
        self.cur_pm = pm if (pm is not None and not pm.isNull()) else None
        self.now.setText(title)
        if self.cur_pm:
            self.cover.setPixmap(self.cur_pm)
        else:
            self.cover.clear()
        self.play_btn.setText("Pause")
        self.status.setText("Playing")
        self.err_shown = False
        self.slider.setValue(0)
        self.timer.start(400)
        self.events.started.emit(title, self.cur_pm if self.cur_pm else QPixmap())

    def is_playing(self):
        if not self.player:
            return False
        try:
            return self.player.get_state() in (vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening)
        except Exception:
            return False

    def toggle(self):
        if not self.player:
            return
        st = self.player.get_state()
        if st in (vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening):
            self.player.set_pause(1)
            self.play_btn.setText("Play")
            self.events.paused.emit()
        elif st == vlc.State.Paused:
            self.player.set_pause(0)
            self.play_btn.setText("Pause")
            self.events.resumed.emit()
        else:
            row = self.current_row if self.current_row >= 0 else max(0, self.listw.currentRow())
            self.play_row(row)

    def stop(self):
        if self.player:
            self.player.stop()
        self.timer.stop()
        self.play_btn.setText("Play")
        self.events.paused.emit()

    def next_song(self, auto):
        n = self.listw.count()
        if n == 0:
            return
        row = self.current_row + 1
        if row >= n:
            if self.loop.currentIndex() == 2 or not auto:
                row = 0
            else:
                self.timer.stop()
                self.play_btn.setText("Play")
                self.events.paused.emit()
                return
        self.play_row(row)

    def prev_song(self):
        self.play_row(max(0, self.current_row - 1))

    def _ended(self):
        self.timer.stop()
        if self.loop.currentIndex() == 1:
            self.play_row(self.current_row)
        else:
            self.next_song(True)

    def _tick(self):
        if not self.player:
            return
        st = self.player.get_state()
        if st == vlc.State.Ended:
            self._ended()
            return
        if st == vlc.State.Error and not self.err_shown:
            self.err_shown = True
            self.status.setText("Playback error. Check that VLC is installed and matches your Python (64-bit).")
            self.play_btn.setText("Play")
            self.events.paused.emit()
            return
        length = self.player.get_length()
        if length <= 0:
            length = self.total_ms
        t = self.player.get_time()
        if length > 0:
            self.slider.setMaximum(length)
            self.total_lbl.setText(fmt_ms(length))
        if not self.seeking and t >= 0:
            self.slider.setValue(t)
        self.cur_lbl.setText(fmt_ms(t))

    # ---------- seeking / state for the mini player ----------
    def seek(self, ms):
        if self.player and self.player.get_media():
            self.player.set_time(int(ms))

    def _seek_now(self):
        self.seek(self.slider.value())

    def _seek_release(self):
        self._seek_now()
        self.seeking = False

    def snapshot(self):
        t, length = 0, self.total_ms
        if self.player:
            try:
                t = max(0, self.player.get_time())
                l = self.player.get_length()
                length = l if l > 0 else self.total_ms
            except Exception:
                pass
        return {"title": self.cur_title, "pixmap": self.cur_pm, "playing": self.is_playing(),
                "t": t, "length": length}

    def shutdown(self):
        self.timer.stop()
        if self.thumb_worker:
            self.thumb_worker.cancelled = True
        for w in list(self.workers):
            w.wait(1500)
        if self.player:
            self.player.stop()


def create_music_tab():
    return MusicTab()