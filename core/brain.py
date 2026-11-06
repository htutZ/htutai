import ast
import difflib
import json
import operator as op
import os
import random
import re
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import datetime

from core import personality as P
from core.intents import INTENTS, SITES, APPS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNED = os.path.join(BASE, "databases", "learned.json")
NOTES = os.path.join(BASE, "databases", "notes.json")

B2A = str.maketrans("၀၁၂၃၄၅၆၇၈၉", "0123456789")
A2B = str.maketrans("0123456789", "၀၁၂၃၄၅၆၇၈၉")
TABS = {"chat": 0, "clock": 1, "calendar": 2, "music": 3, "menu": 4}
WEEK = ["တနင်္လာ", "အင်္ဂါ", "ဗုဒ္ဓဟူး", "ကြာသပတေး", "သောကြာ", "စနေ", "တနင်္ဂနွေ"]
MONTH = ["ဇန်နဝါရီ", "ဖေဖော်ဝါရီ", "မတ်", "ဧပြီ", "မေ", "ဇွန်", "ဇူလိုင်", "ဩဂုတ်",
         "စက်တင်ဘာ", "အောက်တိုဘာ", "နိုဝင်ဘာ", "ဒီဇင်ဘာ"]
UNIT_MY = {"s": "စက္ကန့်", "m": "မိနစ်", "h": "နာရီ"}

PET_CMDS = [
    ("dance", r"\bdance\b|ကပြ|ကပေး|ကလိုက်|ka\s?pya"),
    ("sleep", r"\b(go to sleep|sleep now|nap)\b|အိပ်တော့|အိပ်လိုက်"),
    ("rest", r"\b(sit down|sit|rest|relax|lie down)\b|ထိုင်|လဲလျောင်း"),
    ("come", r"\b(come here|come to me|come)\b|လာခဲ့|ဒီလာ|လာပါ"),
    ("walk", r"\b(walk|roam|wander)\b|လမ်းလျှောက်"),
    ("glitch", r"\bglitch\b"),
    ("hide", r"\bhide\b|ပျောက်သွား"),
    ("show", r"\b(show yourself|appear|unhide)\b|ပြန်ပေါ်"),
]
PET_REPLY = {
    "dance": (["Showtime! 💃", "Watch this~"], ["ကပြမယ်နော်! 💃", "ကြည့်ထား~"]),
    "sleep": (["Fine, nap time. 😴"], ["အိပ်တော့မယ်... 😴"]),
    "rest": (["Chilling. 😌"], ["နားလိုက်ပြီနော် 😌"]),
    "come": (["Coming!", "On my way~"], ["လာပြီ!", "လာနေပြီ~"]),
    "walk": (["Off I go."], ["လမ်းလျှောက်လိုက်ဦးမယ်။"]),
    "glitch": (["bZZzt... ✨"], ["bZZzt... ✨"]),
    "hide": (["Hiding. Say 'show yourself' or use the tray icon."], ["ပုန်းနေမယ်။ 'show yourself' လို့ ပြောပါ၊ ဒါမှမဟုတ် tray icon ကိုသုံးပါ။"]),
    "show": (["Ta-da! I'm back."], ["ပြန်ပေါ်ပြီ!"]),
}

OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
       ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg}


def safe_eval(expr):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 8:
                raise ValueError("power too big")
            return OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError("bad expression")
    return ev(ast.parse(expr, mode="eval"))


def my_num(n):
    return str(n).translate(A2B)


def my_time(dt):
    h = dt.hour
    period = "နံနက်" if 4 <= h < 12 else "နေ့လည်" if 12 <= h < 16 else "ညနေ" if 16 <= h < 19 else "ည"
    h12 = h % 12 or 12
    return f"{period} {my_num(h12)}:{my_num(f'{dt.minute:02d}')} နာရီ"


def my_date(dt):
    return f"ဒီနေ့ {WEEK[dt.weekday()]}နေ့၊ {my_num(dt.year)} ခုနှစ် {MONTH[dt.month - 1]} {my_num(dt.day)} ရက် ဖြစ်ပါတယ်။"


@dataclass
class Reply:
    text: str
    mood: str = "talking"
    action: tuple = None   # ("tab", i) ("remind", secs, what) ("pet", cmd) ("open", target) ("voice", bool) ("forget",)
    clip: str = None       # voice clip name in assets/capri/voice (without .wav)
    fallback: bool = False  # True when no rule matched (the local AI gets a turn)


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Brain:
    def __init__(self, queries=None, responses=None):
        self.qa = {}
        for q, r in zip(queries or [], responses or []):
            self.qa[str(q).lower().strip()] = r
        self.qa.update(_load(LEARNED, {}))

    def respond(self, raw, converted=None, lang="en"):
        my = lang == "my"
        text = raw.strip().translate(B2A)
        low = text.lower()
        conv = (converted or raw).strip().translate(B2A).lower()
        hay = (low, conv) if conv != low else (low,)

        def has(*pats):
            return any(re.search(p, h) for p in pats for h in hay)

        def pick(en, mm=None):
            return random.choice(mm if (my and mm) else en)

        # learn
        m = re.match(r"learn:\s*(.+?)\s*=>\s*(.+)", text, re.I | re.S)
        if m:
            q, a = m.group(1).lower().strip(), m.group(2).strip()
            self.qa[q] = a
            data = _load(LEARNED, {})
            data[q] = a
            _save(LEARNED, data)
            return Reply(f"မှတ်ထားလိုက်ပြီ! '{q}' → '{a}' 😎" if my else f"Got it! Now I know: '{q}' → '{a}' 😎", "happy")

        # what she calls you
        m = re.search(r"(?:call me|my name is)\s+([^\s,.!?]{1,24})", text, re.I) \
            or re.search(r"ငါ့ကို\s*(.{1,24}?)\s*လို့\s*ခေါ်", text)
        if m:
            from core.profile import set_name
            set_name(m.group(1))
            return Reply(f"ကောင်းပြီ၊ {m.group(1)} လို့ ခေါ်မယ်။" if my else f"Okay, I'll call you {m.group(1)}.", "happy")

        # remember a fact about the user / forget the chat
        m = re.match(r"(?:remember(?: that)?|မှတ်ထား(?:ပါ)?)\s*[:,]?\s*(.+)", text, re.I | re.S)
        if m:
            from core.persona import add_fact
            add_fact(m.group(1))
            return Reply("မှတ်ထားလိုက်ပြီ၊ မမေ့ဘူး 😌" if my else "I'll remember that. 😌", "happy")
        if has(r"forget (our )?(chat|chats|conversation|everything)", r"စကားပြောတာတွေ မေ့လိုက်|အားလုံးမေ့လိုက်"):
            return Reply("ကောင်းပြီ၊ အားလုံး မေ့လိုက်ပြီ။" if my else "Okay, wiped our chat memory.", "talking", ("forget",))

        # notes
        m = re.match(r"note\s*:\s*(.+)", text, re.I | re.S)
        if m:
            notes = _load(NOTES, [])
            notes.append({"t": datetime.now().isoformat(timespec="minutes"), "text": m.group(1).strip()})
            _save(NOTES, notes)
            return Reply("မှတ်စုထဲ ထည့်လိုက်ပြီ 📝" if my else "Saved to your notes 📝", "happy")
        if has(r"^(show )?notes$|\bmy notes\b", r"မှတ်စု"):
            notes = _load(NOTES, [])[-5:]
            if not notes:
                return Reply("မှတ်စု မရှိသေးဘူး။" if my else "No notes yet. Use: note: your text")
            return Reply("\n".join(f"• {n['text']}" for n in notes), "happy")

        # calculator
        expr = low.replace("x", "*").replace("×", "*").replace("÷", "/").replace("^", "**")
        m = re.fullmatch(r"(?:calc(?:ulate)?\s*|=\s*)?([\d\s.+\-*/()%]+)", expr)
        if m and re.search(r"\d", m.group(1)) and re.search(r"[+\-*/%]", m.group(1).strip()[1:]):
            try:
                val = safe_eval(m.group(1).strip())
                return Reply(f"{m.group(1).strip()} = {val:g}", "happy")
            except Exception:
                return Reply("တွက်လို့မရဘူး 😅" if my else "Can't calculate that one 😅", "tease")

        # reminders
        m = re.search(r"remind me(?: in)?\s+(\d+)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hours?)\b\s*(?:to|about)?\s*(.*)", low)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            what = m.group(3).strip() or "your thing"
            k = "h" if unit.startswith("h") else "m" if unit.startswith("m") else "s"
            secs = n * {"h": 3600, "m": 60, "s": 1}[k]
            txt = (f"ကောင်းပြီ! {my_num(n)} {UNIT_MY[k]} နေရင် '{what}' ကို သတိပေးမယ်။ မမေ့နဲ့နော် 😏" if my
                   else f"Okay! I'll remind you to {what} in {n}{k}. Don't blame me if you forget. 😏")
            return Reply(txt, "happy", ("remind", secs, what))
        m = re.search(r"(\d+)\s*(စက္ကန့်|မိနစ်|နာရီ)\s*(?:နေရင်|ကြာရင်|အကြာ|အတွင်း|ကြာလျှင်)?\s*(.*?)\s*(?:ကို\s*)?(?:သတိပေး|သတိရ)", text)
        if m:
            n = int(m.group(1))
            k = {"စက္ကန့်": "s", "မိနစ်": "m", "နာရီ": "h"}[m.group(2)]
            what = m.group(3).strip() or "အချိန်စေ့ပြီ"
            secs = n * {"h": 3600, "m": 60, "s": 1}[k]
            return Reply(f"ကောင်းပြီ! {my_num(n)} {UNIT_MY[k]} နေရင် သတိပေးမယ် - {what}", "happy", ("remind", secs, what))

        # tabs
        m = re.search(r"\b(open|go to|show|switch to)\s+(chat|clock|calendar|music|menu)\b", low)
        if m:
            name = m.group(2)
            return Reply(f"{name} ကို ဖွင့်ပေးမယ် ✨" if my else f"Opening {name}! ✨", "happy", ("tab", TABS[name]))
        if has(r"\bplay\b.*\b(music|song)\b|\bmusic\b|သီချင်း|thi\s?chin"):
            return Reply("သီချင်းတဲ့ tab ကို ဖွင့်လိုက်ပြီ 🎵" if my else "Music time! Switching to the music tab 🎵",
                         "happy", ("tab", TABS["music"]))
        if has(r"\bcalendar\b|ပြက္ခဒိန်"):
            return Reply("ပြက္ခဒိန်ကို ဖွင့်ပေးမယ် 📅" if my else "Here's your calendar 📅", "happy", ("tab", TABS["calendar"]))

        # pet commands
        for cmd, pat in PET_CMDS:
            if has(pat):
                en, mm = PET_REPLY[cmd]
                return Reply(pick(en, mm), "happy", ("pet", cmd))

        # voice toggles
        if has(r"\bvoice off\b|\bmute\b|\bshut up\b|အသံပိတ်"):
            return Reply("အသံပိတ်လိုက်ပြီ။" if my else "Okay, going quiet. 🤐", "talking", ("voice", False))
        if has(r"\bvoice on\b|\bunmute\b|အသံဖွင့်"):
            return Reply("အသံပြန်ဖွင့်ပြီ!" if my else "Voice is back on! 🔊", "happy", ("voice", True))

        # open sites / apps, search
        if has(r"\b(open|launch|start)\b|ဖွင့်|pwint"):
            for name, target in SITES.items():
                if re.search(r"\b" + re.escape(name) + r"\b", low):
                    return Reply(f"{name} ကို ဖွင့်လိုက်ပြီ 🌐" if my else f"Opening {name} 🌐", "happy", ("open", target))
            if sys.platform == "win32":
                for name, target in APPS.items():
                    if re.search(r"\b" + re.escape(name) + r"\b", low):
                        return Reply(f"{name} ကို ဖွင့်လိုက်ပြီ" if my else f"Launching {name}", "happy", ("open", target))
        m = re.match(r"(?:search|google|find)\s+(.+)", text, re.I) or re.match(r"(.+?)\s*(?:ကို\s*)?ရှာပေး", text)
        if m:
            q = m.group(1).strip()
            return Reply(f"'{q}' ကို ရှာပေးမယ် 🔎" if my else f"Searching for '{q}' 🔎", "happy",
                         ("open", "https://www.google.com/search?q=" + urllib.parse.quote(q)))

        # time & date
        now = datetime.now()
        if has(r"\b(time|clock)\b|အချိန်|နာရီ|a\s?chein"):
            return Reply(my_time(now) + " ဖြစ်ပါပြီ 😌" if my
                         else "It's " + now.strftime("%I:%M %p") + ". Time flies when you're talking to me. 😌", "happy")
        if has(r"\b(date|today|what day)\b|ဒီနေ့|ရက်စွဲ|ဘာနေ့"):
            return Reply(my_date(now) if my else "Today is " + now.strftime("%A, %d %B %Y") + ".", "happy")

        # fun
        if has(r"\bcoin\b|ဒင်္ဂါး"):
            side = random.choice(["Heads", "Tails"])
            return Reply(("ခေါင်း" if side == "Heads" else "ပန်း") + " 🪙" if my else f"{side}! 🪙", "happy")
        if has(r"\b(roll|dice|die)\b|အန်စာ"):
            n = random.randint(1, 6)
            return Reply(f"အန်စာ - {my_num(n)} 🎲" if my else f"You rolled a {n} 🎲", "happy")

        # chit-chat intents
        for it in INTENTS:
            if has(*it["patterns"]):
                lst = it["my"] if my and it.get("my") else it["en"]
                return Reply(random.choice(lst), it.get("mood", "talking"), None, None if my else it.get("clip"))

        # database + learned answers
        for key in (low, conv):
            if key in self.qa:
                return Reply(str(self.qa[key]))
        match = (difflib.get_close_matches(conv, list(self.qa), n=1, cutoff=0.62)
                 or difflib.get_close_matches(low, list(self.qa), n=1, cutoff=0.62))
        if match:
            return Reply(str(self.qa[match[0]]))

        if my:
            return Reply("ဒါကို မသိသေးဘူး။ သင်ပေးချင်ရင် learn: မေးခွန်း => အဖြေ လို့ ရိုက်ပါ။", "tease", None, None, True)
        return Reply(P.pick(P.UNKNOWN), "tease", None, None, True)