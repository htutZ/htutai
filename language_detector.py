import re
from databases.symbols_db import my_symbols
from databases.transliteration_db import my_transliteration
from databases.context_pairs_db import context_pairs

def context_aware_translation(text):
    for word, contexts in context_pairs.items():
        if word in text:
            for context in contexts:
                if context['context'] in text:
                    text = text.replace(word, context['translation'])
    return text

def detect_language(text):
    # Check for Myanmar characters in the text
    myanmar_chars = "က-႟"
    if re.search("[{}]+".format(myanmar_chars), text):
        return "Myanmar (Burmese)"

    # Use context-aware translation
    text = context_aware_translation(text)

    # Check for Myanglish (combination of Latin and Myanmar)
    for key in my_transliteration:
        if key in text and my_transliteration[key] in text:
            return "Myanglish"

    # If neither condition is met, it's likely Latin
    return "Latin"
