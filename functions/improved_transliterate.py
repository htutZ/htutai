from databases.transliteration_db import my_transliteration
from databases.myanglish_context_db import myanglish_context
from databases.priority_transliterations import priority_transliterations
from databases.symbols_db import my_symbols

def improved_transliterate(myanglish_text):
    # Split the Myanglish sentence into words
    words = myanglish_text.split()

    burmese_sentence = ""

    # Iterate through the words
    for i, word in enumerate(words):
        converted_word = None

        # Apply priority_transliterations first
        for pattern, replacement in priority_transliterations.items():
            word = word.replace(pattern, replacement)

        # Check if the word is in the myanglish_context if not converted by priority
        if word not in priority_transliterations:
            if word in myanglish_context:
                for entry in myanglish_context[word]:
                    # Check if any of the surrounding words match the context
                    surrounding_words = words[max(0, i-1):i+2]  # get the previous, current, and next word
                    if any(context_word in surrounding_words for context_word in entry["context"]):
                        converted_word = entry["translation"]
                        break

        # If the word wasn't converted using context or priority, transliterate using the regular databases
        if not converted_word:
            # Using the transliteration database, prioritizing longer patterns
            for pattern in sorted(my_transliteration.keys(), key=len, reverse=True):
                if pattern in word:
                    word = word.replace(pattern, my_transliteration[pattern])

            # Further refinement: For any remaining Latin characters, attempt a direct transliteration
            for char in word:
                if char in my_symbols:
                    word = word.replace(char, my_symbols[char])
                elif char in my_transliteration:
                    word = word.replace(char, my_transliteration[char])

            converted_word = word

        burmese_sentence += converted_word  # Concatenate without adding spaces

    return burmese_sentence