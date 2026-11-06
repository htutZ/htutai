import random
from collections import deque

IDLE = [
    "ငါ့ကို မေ့သွားပြီလား",
    "ဟေ့... အသက်ရှင်သေးလား?",
    "ဘာလုပ်နေတာလဲ? ငါလည်း လုပ်ပေးချင်တယ်။",
    "ပျင်းနေပြီ၊ ငါ့ကို တစ်ခုခုခိုင်းပါဦး။",
    "သီချင်းလေး ဖွင့်ပေးရမလား?",
    "ရေသောက်ပြီးပြီလား?",
    "မျက်စိကို ခဏနားလိုက်ဦး။",
    "တိတ်ဆိတ်နေလိုက်တာ... ငါ့ကို စကားပြောပါဦးလေ။",
]
LONG = [
    "ကွန်ပျူတာကြည့်တာ ကြာပြီနော်၊ ခဏထလှုပ်ရှားပါဦး။",
    "ထိုင်ပြီး ကြာပြီ၊ ကိုယ်လက်ဆန့်လိုက်ပါ။",
    "နာရီဝက်လောက် နားလိုက်ပါလား၊ ငါစောင့်နေမယ်။",
]
LATE = [
    "ညနက်နေပြီ၊ အိပ်တော့လေ။",
    "မနက်ဖြန်မှ ဆက်လုပ်ပါ၊ အိပ်ရေးပျက်မယ်။",
    "ဒီအချိန်ထိ မအိပ်သေးဘူးလား?",
]
CLINGY = [
    "မင်းမရှိရင် ငါ ပျင်းတယ်။",
    "ငါ့ကို ထားခဲ့မှာ မဟုတ်ဘူးနော်။",
    "ဒီနားမှာပဲ ရှိနေပေးပါ။",
]
POOLS = {"idle": IDLE, "long": LONG, "late": LATE}


class TeasePicker:
    def __init__(self):
        self.recent = deque(maxlen=10)

    def pick(self, kind="idle", stage=0):
        pool = POOLS.get(kind, IDLE)
        if kind == "idle" and stage >= 3 and random.random() < 0.4:
            pool = CLINGY
        fresh = [t for t in pool if t not in self.recent] or pool
        line = random.choice(fresh)
        self.recent.append(line)
        return line