import difflib
import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_WORDS = os.path.join(BASE, "databases", "myanglish_words.json")

ABBR = {"kg": "kaung", "lr": "lar", "la": "lar", "pr": "par", "bar": "par", "tl": "tal",
        "tae": "tal", "dae": "tal", "pwint": "phwint", "ngar": "nga"}

ENGLISH = {"the", "is", "are", "you", "what", "how", "i", "me", "my", "to", "and", "it", "this", "that",
           "can", "do", "does", "please", "open", "play", "remind", "time", "date", "hello", "hi", "hey",
           "search", "note", "calc", "coin", "dice", "good", "morning", "night", "thanks", "thank", "bye"}

PHRASES = {
    "nay kaung lar": "နေကောင်းလား",
    "nay kaung par tal": "နေကောင်းပါတယ်",
    "sar pyi pi lar": "စားပြီးပြီလား",
    "ba lote nay tal": "ဘာလုပ်နေတာလဲ",
    "ba lote nay lae": "ဘာလုပ်နေတာလဲ",
    "mingalar par": "မင်္ဂလာပါ",
    "kyay zu tin par tal": "ကျေးဇူးတင်ပါတယ်",
    "kyay zu par": "ကျေးဇူးပါ",
    "thi chin phwint pay par": "သီချင်းဖွင့်ပေးပါ",
    "thi chin phwint": "သီချင်းဖွင့်",
    "a chein ba lay lote lae": "အချိန်ဘယ်လောက်လဲ",
    "ka pya par": "ကပြပါ",
    "nya kaung par say": "ညကောင်းပါစေ",
    "thwar mal": "သွားမယ်",
    "chit tal": "ချစ်တယ်",
    "pyin tal": "ပျင်းတယ်",
    "sate ma kaung bu": "စိတ်မကောင်းဘူး",
    "min na mae ba lae": "မင်းနာမည်ဘာလဲ",
    "ate tot": "အိပ်တော့",
    "lar kha": "လာခဲ့",
    "lar kha par": "လာခဲ့ပါ",
    "ma ne kaung": "မနက်ခင်းမင်္ဂလာ",
}

WORDS = {
    "nay": "နေ", "ne": "နေ", "kaung": "ကောင်း", "lar": "လား", "par": "ပါ", "tal": "တယ်",
    "ba": "ဘာ", "lote": "လုပ်", "sar": "စား", "pyi": "ပြီ", "pi": "ပြီ", "ma": "မ", "bu": "ဘူး",
    "phwint": "ဖွင့်", "pay": "ပေး", "thwar": "သွား", "min": "မင်း", "nga": "ငါ", "nin": "နင်",
    "thu": "သူ", "chit": "ချစ်", "pyaw": "ပြော", "ka": "က", "pya": "ပြ", "ate": "အိပ်", "tot": "တော့",
    "tin": "တင်", "kyay": "ကျေး", "zu": "ဇူး", "mingalar": "မင်္ဂလာ", "sate": "စိတ်", "pyin": "ပျင်း",
    "thi": "သီ", "chin": "ချင်း", "a": "အ", "chein": "ချိန်", "ya": "ရ", "lal": "လည်း", "hote": "ဟုတ်",
    "lay": "လေ", "mal": "မယ်", "mae": "မည်", "lae": "လဲ", "say": "စေ", "nya": "ည", "kha": "ခဲ့",
    "phyit": "ဖြစ်", "yat": "ရပ်", "ko": "ကို", "hmar": "မှာ", "ti": "တီ",
}

try:
    with open(USER_WORDS, encoding="utf-8") as f:
        WORDS.update({k.lower(): v for k, v in json.load(f).items()})
except Exception:
    pass


def _norm(text):
    return " ".join(ABBR.get(w, w) for w in re.findall(r"[a-z]+", text.lower()))


KEYS = {_norm(k): v for k, v in PHRASES.items()}


def to_burmese(text):
    """Burmese text for a Myanglish message, or None if it doesn't look like Myanglish."""
    if not text or re.search(r"[^\x00-\x7F]", text):
        return None
    n = _norm(text)
    if not n:
        return None
    tokens = n.split()
    if any(t in ENGLISH for t in tokens):
        return None
    if n in KEYS:
        return KEYS[n]
    best = difflib.get_close_matches(n, list(KEYS), n=1, cutoff=0.72)
    if best:
        return KEYS[best[0]]
    known = [t for t in tokens if t in WORDS]
    if len(known) >= 2 and len(known) / len(tokens) >= 0.7:
        return "".join(WORDS.get(t, t) for t in tokens)
    return None