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

def convert_to_myanmar(text):
    """Convert potential Myanglish words in the text to Myanmar."""
    for key, value in my_transliteration.items():
        text = text.replace(key, value)
    return text

def detect_language(text):

    combined_myanglish_words = [sym + trans for sym in my_symbols for trans in my_transliteration]

    myanmar_chars = "က-႟"
    if re.search("[{}]+".format(myanmar_chars), text):
        return "Myanmar (Burmese)"

    for word in combined_myanglish_words:
        if word in text:
            return "Myanglish"

    return "English"



