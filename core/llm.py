import json
import os
import urllib.error
import urllib.request

from PySide2.QtCore import QThread, Signal

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(BASE, "databases", "ai_config.json")

# Local only. provider "ollama" = Ollama on this PC.
# provider "openai" = any OpenAI-compatible server (LM Studio, llama.cpp) running on this PC.
# style: "sassy", "soft" or "cool"
DEFAULT = {
    "enabled": True,
    "provider": "ollama",
    "model": "gemma3",
    "style": "sassy",
    "url": "http://127.0.0.1:11434",
    "api_key": "",
    "timeout": 180,
    "keep_alive": "30m",
}


def load_config():
    cfg = dict(DEFAULT)
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT, f, indent=2)
    return cfg


def _post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _installed(base):
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=5) as r:
            return [m["name"] for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        return []


def _resolve_model(cfg, base):
    if cfg.get("_resolved"):
        return cfg["_resolved"]
    want = cfg["model"]
    names = _installed(base)
    pick = want
    if names and want not in names and want + ":latest" not in names:
        family = want.split(":")[0]
        same = [n for n in names if n.split(":")[0] == family]
        pick = same[0] if same else names[0]
        print(f"[ai] '{want}' not installed, using '{pick}'")
    cfg["_resolved"] = pick
    return pick


def chat(cfg, messages):
    h = {"Content-Type": "application/json"}
    t = cfg.get("timeout", 180)
    base = cfg["url"].rstrip("/")
    try:
        if cfg["provider"] == "ollama":
            model = _resolve_model(cfg, base)
            data = _post(base + "/api/chat", {
                "model": model, "messages": messages, "stream": False,
                "keep_alive": cfg.get("keep_alive", "30m"),
                "options": {"temperature": 0.5, "top_p": 0.9, "repeat_penalty": 1.25,
                            "num_predict": 90, "num_ctx": 4096},
            }, h, t)
            return data["message"]["content"].strip()
        if cfg.get("api_key"):
            h["Authorization"] = "Bearer " + cfg["api_key"]
        data = _post(base + "/chat/completions", {"model": cfg["model"], "messages": messages,
                                                  "temperature": 0.5, "max_tokens": 90}, h, t)
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"model not installed. Run: ollama pull {cfg['model']}")
        raise RuntimeError(f"AI server said {e.code}")
    except urllib.error.URLError:
        raise RuntimeError("local AI not running. Start Ollama, then try again")
    except Exception as e:
        raise RuntimeError(str(e)[:120])


class AIWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg, messages):
        super().__init__()
        self.cfg, self.messages = cfg, messages

    def run(self):
        try:
            self.done.emit(chat(self.cfg, self.messages))
        except Exception as e:
            self.failed.emit(str(e))