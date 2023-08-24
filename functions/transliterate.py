from databases.transliteration_db import my_transliteration
from databases.symbols_db import my_symbols

def transliterate(text):
    # Order the dictionaries by key length in descending order
    sorted_transliteration = dict(sorted(my_transliteration.items(), key=lambda item: len(item[0]), reverse=True))
    sorted_symbols = dict(sorted(my_symbols.items(), key=lambda item: len(item[0]), reverse=True))
    
    # Try replacing with radicals first
    for key, value in sorted_transliteration.items():
        text = text.replace(key, value)
    
    # Then replace with symbols
    for key, value in sorted_symbols.items():
        text = text.replace(key, value)
        
    return text
