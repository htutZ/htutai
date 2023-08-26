from databases.transliteration_db import my_transliteration
from databases.symbols_db import my_symbols
from databases.myanglish_context_db import myanglish_context

def transliterate(text):
    words = text.split()
    translated_words = []
    
    for idx, word in enumerate(words):
        if word in myanglish_context:
            best_translation = myanglish_context[word][0]["translation"]
            for option in myanglish_context[word]:
                if any(context_word in words[max(0, idx-1):min(len(words), idx+2)] for context_word in option["context"]):
                    best_translation = option["translation"]
                    break
            translated_words.append(best_translation)
        elif word in my_transliteration:
            translated_words.append(my_transliteration[word])
        else:
            translated_words.append(word)
    
    return ' '.join(translated_words)

