from databases.transliteration_db import my_transliteration
from databases.myanglish_context_db import myanglish_context
from databases.priority_transliterations import priority_transliterations
from databases.symbols_db import my_symbols

def improved_transliterate(myanglish_text):
    words = myanglish_text.split()

    burmese_sentence = ""

    for i, word in enumerate(words):
        converted_word = None

        for pattern, replacement in priority_transliterations.items():
            word = word.replace(pattern, replacement)

        if word not in priority_transliterations:
            if word in myanglish_context:
                for entry in myanglish_context[word]:
                    surrounding_words = words[max(0, i-1):i+2]  
                    if any(context_word in surrounding_words for context_word in entry["context"]):
                        converted_word = entry["translation"]
                        break

        if not converted_word:
            for pattern in sorted(my_transliteration.keys(), key=len, reverse=True):
                if pattern in word:
                    word = word.replace(pattern, my_transliteration[pattern])

            for char in word:
                if char in my_symbols:
                    word = word.replace(char, my_symbols[char])
                elif char in my_transliteration:
                    word = word.replace(char, my_transliteration[char])

            converted_word = word

        burmese_sentence += converted_word 

    return burmese_sentence