import re
import sys

AI_RE = re.compile(r"chatgpt|openai|gemini|copilot|perplexity|deepseek|grok|claude|character\.ai|meta ai|le chat|mistral", re.I)
NAMES = {"chatgpt": "ChatGPT", "openai": "ChatGPT", "gemini": "Gemini", "copilot": "Copilot",
         "perplexity": "Perplexity", "deepseek": "DeepSeek", "grok": "Grok", "claude": "Claude",
         "character.ai": "Character.AI", "meta ai": "Meta AI", "le chat": "Le Chat", "mistral": "Mistral"}


def foreground_title():
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        u = ctypes.windll.user32
        h = u.GetForegroundWindow()
        n = u.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, buf, n + 1)
        return buf.value
    except Exception:
        return ""


def foreground_ai():
    title = foreground_title()
    if not title or "htut" in title.lower():
        return None
    m = AI_RE.search(title)
    return NAMES.get(m.group(0).lower()) if m else None