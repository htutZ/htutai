import asyncio
import hashlib
import os
import re
import threading
from collections import deque

from PySide2.QtCore import QObject, QThread, Signal

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, "cache", "tts")
CLIP_DIR = os.path.join(BASE, "assets", "capri", "voice")
RE_MY = re.compile(r"[\u1000-\u109F]")
RE_JUNK = re.compile(r"[^\w\s\u1000-\u109F.,!?;:'\-]")


class Speaker(QObject):
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.voice_my = "my-MM-NilarNeural"      # female; use my-MM-ThihaNeural for male
        self.voice_en = "en-US-AriaNeural"
        self.rate, self.pitch = "+0%", "+0Hz"
        self._gen = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._mixer = None
        os.makedirs(CACHE, exist_ok=True)

    def _get_mixer(self):
        if self._mixer is None:
            import pygame
            pygame.mixer.init()
            self._mixer = pygame.mixer
        return self._mixer

    def set_enabled(self, on):
        self.enabled = on
        if not on:
            self.stop()

    def busy(self):
        if self._pending > 0:
            return True
        try:
            return bool(self._mixer and self._mixer.music.get_busy())
        except Exception:
            return False

    def stop(self):
        self._gen += 1
        try:
            if self._mixer:
                self._mixer.music.stop()
        except Exception:
            pass

    def clip(self, name):
        """Play a recorded clip from assets/capri/voice/<name>.wav. Returns True if it started."""
        if not self.enabled:
            return False
        path = os.path.join(CLIP_DIR, name + ".wav")
        if not os.path.exists(path):
            return False
        self._gen += 1
        try:
            mixer = self._get_mixer()
            mixer.music.load(path)
            mixer.music.play()
            return True
        except Exception as e:
            self.failed.emit(f"Voice error: {e}")
            return False

    def pick_voice(self, text):
        mm = len(RE_MY.findall(text))
        return self.voice_my if mm >= max(2, len(text) * 0.25) else self.voice_en

    def speak(self, text):
        if not self.enabled:
            return
        clean = re.sub(r"\s+", " ", RE_JUNK.sub(" ", text)).strip()
        if len(clean) < 2:
            return
        self._gen += 1
        gen = self._gen
        with self._lock:
            self._pending += 1
        threading.Thread(target=self._work, args=(clean, gen), daemon=True).start()

    def _work(self, text, gen):
        try:
            import edge_tts
            voice = self.pick_voice(text)
            key = hashlib.md5(f"{voice}|{self.rate}|{self.pitch}|{text}".encode("utf-8")).hexdigest()
            path = os.path.join(CACHE, key + ".mp3")
            if not os.path.exists(path):
                part = path + ".part"
                asyncio.run(edge_tts.Communicate(text, voice, rate=self.rate, pitch=self.pitch).save(part))
                os.replace(part, path)
            if gen != self._gen:
                return
            mixer = self._get_mixer()
            mixer.music.load(path)
            mixer.music.play()
        except ImportError:
            self.failed.emit("Voice needs: pip install edge-tts pygame")
        except Exception as e:
            self.failed.emit(f"Voice error: {e}")
        finally:
            with self._lock:
                self._pending -= 1


class Listener(QThread):
    status = Signal(str)
    heard = Signal(str)
    failed = Signal(str)

    def __init__(self, lang="my-MM"):
        super().__init__()
        self.lang = lang
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import numpy as np
            import sounddevice as sd
            import speech_recognition as sr
        except ImportError:
            self.failed.emit("Mic needs: pip install sounddevice SpeechRecognition numpy")
            return
        rate, block = 16000, 1600
        frames, pre, noise = [], deque(maxlen=3), []
        started, silence, thr = False, 0, 350.0
        try:
            self.status.emit("Listening… 🎤")
            with sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=block) as stream:
                for i in range(220):
                    if self._cancel:
                        self.status.emit("Cancelled")
                        return
                    data, _ = stream.read(block)
                    rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                    if i < 4:
                        noise.append(rms)
                        thr = max(350.0, 2.5 * sum(noise) / len(noise))
                        continue
                    if rms > thr:
                        if not started:
                            frames.extend(pre)
                        started, silence = True, 0
                    elif started:
                        silence += 1
                    if started:
                        frames.append(data.copy())
                        if silence >= 12:
                            break
                    else:
                        pre.append(data.copy())
                        if i > 70:
                            break
            if not frames:
                self.failed.emit("I didn't hear anything")
                return
            self.status.emit("Understanding…")
            audio = sr.AudioData(np.concatenate(frames).tobytes(), rate, 2)
            text = sr.Recognizer().recognize_google(audio, language=self.lang)
            self.heard.emit(text)
        except sr.UnknownValueError:
            self.failed.emit("Couldn't understand, try again")
        except sr.RequestError:
            self.failed.emit("Speech service unreachable (needs internet)")
        except Exception as e:
            self.failed.emit(f"Mic error: {e}")