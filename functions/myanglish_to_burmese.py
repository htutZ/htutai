from databases.transliteration_db import my_transliteration
from databases.symbols_db import my_symbols

def myanglish_to_burmese(text):
    # Convert using the symbols database
    for key, value in my_symbols.items():
        text = text.replace(key, value)

    # Convert using the transliteration database
    for key, value in my_transliteration.items():
        text = text.replace(key, value)

    return text
