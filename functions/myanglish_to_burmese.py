from databases.transliteration_db import my_transliteration
from databases.symbols_db import my_symbols
from databases.priority_transliterations import priority_transliterations
from databases.myanglish_context_db import myanglish_context

def myanglish_to_burmese(myanglish_text):
    words = myanglish_text.split()

    for i, word in enumerate(words):
        for key, translations in myanglish_context.items():
            if word == key:
                for translation in translations:
                    context_match = any(context_word in words for context_word in translation['context'])
                    if context_match:
                        words[i] = translation['translation']

    burmese_text = ''
    for word in words:
        if word in priority_transliterations:
            burmese_text += priority_transliterations[word]
        else:

            first_char = word[0]
            if first_char in my_symbols:
                burmese_text += my_symbols[first_char]
                word = word[1:]  

            for char in word:
                if char in my_transliteration:
                    burmese_text += my_transliteration[char]
                else:
                    burmese_text += char
            burmese_text += ' '

    return burmese_text.strip()