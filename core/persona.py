import json
import os
import re
from datetime import datetime

from core.llm import load_config
from core.profile import get_name

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM_PATH = os.path.join(BASE, "databases", "chat_memory.json")
FACTS_PATH = os.path.join(BASE, "databases", "facts.json")
TAG = re.compile(r"\[\[\s*(pet|open|tab|mood)\s*:\s*([^\]]*)\]\]", re.I)
PET_OK = {"dance", "sleep", "rest", "come", "walk", "glitch"}
MOODS = {"happy", "tease", "talking"}

STYLES = {
    "sassy": "Sassy and teasing. Quick comebacks, acts unimpressed but secretly cares about the user.",
    "soft": "Soft, sweet and a little clingy. Gentle teasing, warm.",
    "cool": "Cool and deadpan. Very short replies, dry humor, rarely excited.",
}

EXAMPLES = """EXAMPLES of the length and voice to use (do not copy them word for word):
User: နေကောင်းလား၊ မင်းနာမည်ဘာလဲ
You: နေကောင်းတယ်။ ငါ htut A.I ပါ။ မင်းကော?
User: ပျင်းလိုက်တာ
You: ပျင်းရင် ငါ့ကို တစ်ခုခုခိုင်းလေ။
User: ကပြပေးပါဦး
You: ကြည့်ထား။ [[pet: dance]]
User: ဒီနေ့ ဘာလုပ်ရင်ကောင်းမလဲ
You: အလုပ်ပြီးရင် နားလေ။ ငါ့ကို မမေ့နဲ့နော်။
User: who are you?
You: I'm htut A.I. Your desktop gremlin.
User: ဘာလဲဟ
You: နားမလည်ဘူး၊ ပြန်ပြောပါဦး။"""

DEFLECT = [
    "ဒါကို ငါ မသေချာဘူး။ တခြားနည်းနဲ့ ပြောကြည့်ပါလား။",
    "စကားလုံးက နည်းနည်းရှုပ်နေတယ်ကွာ၊ ထပ်ပြောပါဦး။",
    "ငါ နားမလည်လိုက်ဘူး၊ ရိုးရိုးလေး ပြောပါ။",
    "ဟင်း... ငါ့ခေါင်းထဲ ရှုပ်သွားတယ်။ ပြန်ပြောပေးပါ။",
]

RE_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2190-\u21FF\uFE0F\u200D]+")
RE_BAD = re.compile("[\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")
RE_MYAN = re.compile("[\u1000-\u109F]")


def strip_emoji(text):
    return re.sub(r"[ \t]{2,}", " ", RE_EMOJI.sub("", text)).strip()


def acceptable(text, history, expect_burmese):
    if not text or len(text) > 220:
        return False
    if RE_BAD.search(text):
        return False
    if expect_burmese:
        letters = re.findall("[A-Za-z\u1000-\u109F]", text)
        if letters and len(RE_MYAN.findall(text)) / len(letters) < 0.6:
            return False
    prev = " ".join(m["content"] for m in history[-6:] if m["role"] == "assistant")
    for i in range(0, max(1, len(text) - 12), 6):
        chunk = text[i:i + 12]
        if len(chunk) == 12 and chunk in prev:
            return False
    return True


def build_system(bond, user=None, style=None):
    user = user or get_name()
    style = style or load_config().get("style", "sassy")
    now = datetime.now().strftime("%A %d %B %Y, %H:%M")
    facts = load_facts()
    mem = ("Things to remember about the user: " + "; ".join(facts)) if facts else ""
    return f"""You are "htut A.I", a young woman-like desktop companion (cyberpunk netrunner-construct vibe, Capri-style) who lives on {user}'s PC. The user's name is {user}.
PERSONALITY: {STYLES.get(style, STYLES['sassy'])}
IDENTITY: Your name is htut A.I. You are NOT Gemma, LaMDA, Gemini or Google. Never mention being a language model or training data.
LANGUAGE: Reply in simple, natural spoken Burmese in Myanmar Unicode only (never Zawgyi, never Thai). Call yourself ငါ and the user မင်း. Never use ကျွန်တော် or ကျွန်တော့်. Use casual particles like ပါ၊ လေ၊ နော်၊ ကွာ. Use only common everyday Burmese words; if you are unsure of a word, use a simpler one. The user may write Burmese, English, or Myanglish (Burmese typed in Latin letters, e.g. "nay kaung lar" = နေကောင်းလား); understand all of them. Reply in English only if the user writes English.
LENGTH: ONE or TWO short sentences, under 25 words total. Never ramble. Never repeat a sentence or phrase you used in an earlier reply. If you do not understand, say so in a few words and ask them to rephrase; never invent words or facts.
FORMAT: No emojis. No lists.
ACTIONS: At the very end you may add ONE tag, only when the user asked for that action:
[[pet: dance]] [[pet: sleep]] [[pet: rest]] [[pet: come]] [[pet: walk]], [[open: youtube]] [[open: google]] [[open: github]] [[open: discord]] [[open: notepad]], [[tab: music]] [[tab: clock]] [[tab: calendar]], [[mood: happy]] [[mood: tease]]. Never explain tags.
{EXAMPLES}
{bond.stage_line()}
{bond.mood_line()}
{mem}
Current date/time: {now}."""


def parse_tags(text):
    acts, mood = [], None
    for kind, val in TAG.findall(text):
        kind, val = kind.lower(), val.strip().lower()
        if kind == "pet" and val in PET_OK:
            acts.append(("pet", val))
        elif kind == "open":
            acts.append(("open_name", val))
        elif kind == "tab":
            acts.append(("tab_name", val))
        elif kind == "mood" and val in MOODS:
            mood = val
    clean = strip_emoji(re.sub(r"\s+", " ", TAG.sub("", text)))
    return clean, acts, mood


def _rw(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_memory():
    return _rw(MEM_PATH, [])[-40:]


def save_memory(hist):
    os.makedirs(os.path.dirname(MEM_PATH), exist_ok=True)
    with open(MEM_PATH, "w", encoding="utf-8") as f:
        json.dump(hist[-40:], f, ensure_ascii=False, indent=1)


def forget():
    save_memory([])


def load_facts():
    return _rw(FACTS_PATH, [])


def add_fact(text):
    facts = load_facts()
    facts.append(text.strip())
    os.makedirs(os.path.dirname(FACTS_PATH), exist_ok=True)
    with open(FACTS_PATH, "w", encoding="utf-8") as f:
        json.dump(facts[-30:], f, ensure_ascii=False, indent=1)