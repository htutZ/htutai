import random
from datetime import datetime

IDLE_TEASES = [
    "Hellooo~ are you still alive over there? 😏",
    "You've been staring at the screen for a while. Need me to do something?",
    "I'm bored. Ask me the time, a joke, or tell me to dance.",
    "Psst. Want me to open the music tab? I know you do. 🎵",
    "Typing nothing is also a choice, I guess. 👀",
    "Drink some water. I'm serious. 💧",
    "Rest your eyes for a second, you've been on the screen forever.",
    "ငါ့ကို မေ့သွားပြီလား 😏",
    "ဟေ့... အသက်ရှင်သေးလား? 😏",
    "ဘာလုပ်နေတာလဲ? ငါလည်း လုပ်ပေးချင်တယ်။",
    "ငါ့ကို ဘာခိုင်းမလဲ? ပျင်းနေပြီ။",
    "သီချင်းလေး ဖွင့်ပေးရမလား? 🎵",
    "ရေသောက်ပြီးပြီလား? 💧",
    "မျက်စိကို ခဏနားလိုက်ဦး၊ ကွန်ပျူတာကြည့်တာ ကြာပြီ 👀",
]
LATE_NIGHT = [
    "It's way past midnight. Sleep, htut. I'll still be here. 😴",
    "Another late night? Tell me to remind you to sleep.",
    "သန်းခေါင်ကျော်ပြီနော်၊ အိပ်တော့လေ 😴",
    "ညနက်နေပြီ htut၊ မနက်ဖြန်မှ ဆက်လုပ်ပါ။",
]
POKE = [
    "Hey! Hands off. ...okay, do it again. 😳",
    "Ow! That tickles.",
    "Stop poking me and give me a task!",
    "ဟေ့! မထိနဲ့လေ 😤",
    "ဟိတ်! ကလိနေတာလား?",
    "ထပ်ထိရင် ငါ ပြေးမှာနော်!",
]
SLEEPY = ["...zzz... (I dozed off waiting for you)", "...zzz... ငါ အိပ်ပျော်သွားတယ်..."]
WAKE = [
    "Oh, you're back. Missed me? 😏",
    "Finally! I was getting lonely.",
    "ပြန်လာပြီလား? ငါ့ကို လွမ်းလား? 😏",
    "နောက်ဆုံးတော့ ပြန်လာပြီ!",
]
THINKING = ["Hmm...", "Let me think...", "One sec..."]
UNKNOWN = [
    "I didn't get that. Teach me: learn: your question => my answer",
    "Hmm, no idea. Teach me with: learn: question => answer",
]


def idle_tease():
    if datetime.now().hour in (0, 1, 2, 3, 4) and random.random() < 0.6:
        return random.choice(LATE_NIGHT)
    return random.choice(IDLE_TEASES)


def pick(lst):
    return random.choice(lst)


def idle_tease_clip():
    if datetime.now().hour in (0, 1, 2, 3, 4) and random.random() < 0.6:
        return "Isn't it late, choom? 😴 Go sleep. I'll watch the windows.", "hello-late"
    if random.random() < 0.2:
        return "Miss me, choom? 😏", "hello-miss"
    return idle_tease(), None


def wake_clip():
    if random.random() < 0.5:
        return "Miss me, choom? 😏 I was getting lonely.", "tease", "hello-miss"
    return pick(WAKE), "tease", None