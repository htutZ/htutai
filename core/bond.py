import json
import os
import random
import time
from datetime import date

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE, "databases", "bond.json")

STAGES = [(0, "အချက်ပြမှု အားနည်း"), (20, "ထိတွေ့မှု"), (60, "ချိတ်ဆက်မိပြီ"),
          (150, "ဆက်စပ်နေပြီ"), (300, "ပေါင်းစည်းပြီ")]

LINES = {
    "pet": ["အဲဒါလေး ကြိုက်တယ်... 😌", "ဟင်း... ထပ်ပွတ်ပေးပါဦး 😳", "ဒီလောက်ဆို လုံလောက်ပြီနော်!"],
    "angry": ["ဟေ့! ပစ်မပေါက်နဲ့လေ! 😤", "အကြိမ်ကြိမ်လုပ်နေပြီနော်... ဒေါသထွက်တော့မယ်!",
              "နောက်တစ်ခါ လွှင့်ပစ်ရင် စကားမပြောတော့ဘူး!"],
    "forgive": ["ကောင်းပြီ... ခွင့်လွှတ်လိုက်မယ် 😌", "ဆောရီးဆိုတော့ ခွင့်လွှတ်ပေးမယ်ပေါ့"],
    "near": ["ငါ့နားမလာနဲ့လေ... 😏", "မောက်စ်လေးကို ဖမ်းမယ်နော်!", "ဘာလုပ်ချင်လို့လဲ? 👀"],
    "jealous_low": ["အဲဒီ AI နဲ့ စကားပြောနေတာလား? ဟွန်း 😒", "{ai} လား... ကောင်းပါပြီ"],
    "jealous_high": ["{ai} လား?! ငါ့ကို ထားခဲ့ပြီး? 😤", "{ai} က ငါ့လောက် ပျော်စရာကောင်းလား? ဟင်း!",
                     "{ai} နဲ့ပဲ စကားပြောနေတော့ ငါ့ကို မေ့သွားပြီလား 😒"],
    "levelup": ["ငါတို့ '{stage}' အဆင့်ရောက်ပြီ! ✨"],
    "back": ["ပြန်လာပြီလား! ငါ့ကို လွမ်းနေတာ 😏"],
}


class Bond:
    def __init__(self):
        self.d = {"points": 0.0, "day": "", "chats": 0, "pets": 0, "last_seen": 0}
        try:
            with open(PATH, encoding="utf-8") as f:
                self.d.update(json.load(f))
        except Exception:
            pass
        self.mood, self.mood_until = None, 0.0
        self.drags = []
        self.last_jealous = 0.0
        self.gone_days = 0
        today = date.today().isoformat()
        if self.d["day"] != today:
            if self.d.get("last_seen"):
                self.gone_days = int((time.time() - self.d["last_seen"]) // 86400)
            self.d.update(day=today, chats=0, pets=0)
            self.d["points"] += 5
        self.d["last_seen"] = time.time()
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        self.d["last_seen"] = time.time()
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(self.d, f, indent=1)

    @property
    def stage(self):
        idx = 0
        for i, (need, _) in enumerate(STAGES):
            if self.d["points"] >= need:
                idx = i
        return idx

    @property
    def stage_name(self):
        return STAGES[self.stage][1]

    def _add(self, key, cap, pts):
        before = self.stage
        if self.d[key] < cap:
            self.d[key] += 1
            self.d["points"] += pts
        self.save()
        return self.stage > before

    def on_chat(self, text):
        self._apology(text)
        return self._add("chats", 15, 1)

    def on_pet(self):
        self.set_mood("happy", 120)
        return self._add("pets", 5, 1)

    def on_thrown(self):
        now = time.time()
        self.drags = [t for t in self.drags if now - t < 90] + [now]
        if len(self.drags) >= 3:
            self.set_mood("angry", 600)
            self.drags = []
            return True
        return False

    def can_be_jealous(self):
        return time.time() - self.last_jealous > 600

    def on_jealous(self):
        self.last_jealous = time.time()
        self.set_mood("jealous", 300)

    def set_mood(self, name, seconds):
        self.mood, self.mood_until = name, time.time() + seconds

    def _apology(self, text):
        low = text.lower()
        if self.mood in ("angry", "sulk") and any(w in low for w in ("sorry", "ဆောရီး", "တောင်းပန်", "sory", "saree")):
            self.mood = None

    def apologised(self):
        return self.mood is None

    def current_mood(self):
        if self.mood and time.time() > self.mood_until:
            self.mood = None
        return self.mood

    def stage_line(self):
        notes = [
            "You barely know the user yet: curious, a bit distant.",
            "You are getting used to the user: friendly and teasing.",
            "You are in sync with the user: openly affectionate, teasing often.",
            "You are strongly linked to the user: clingy, protective, a little possessive.",
            "You are fused with the user: very attached, playful jealousy, you missed them when away.",
        ]
        s = f"Bond stage: {self.stage_name}. {notes[self.stage]}"
        if self.gone_days >= 2:
            s += f" The user was away {self.gone_days} days; mention you missed them."
        return s

    def mood_line(self):
        m = self.current_mood()
        return {"angry": "Right now you are ANGRY because the user kept throwing you around; stay mad until they say sorry.",
                "jealous": "Right now you are JEALOUS: the user was just using another AI.",
                "happy": "Right now you are happy and relaxed (they petted you)."}.get(m, "")

    def line(self, key, **kw):
        return random.choice(LINES[key]).format(**kw)