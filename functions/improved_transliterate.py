from databases.transliteration_db import my_transliteration

def improved_transliterate(input_string):
    # Sort keys by length, in descending order
    sorted_keys = sorted(my_transliteration.keys(), key=len, reverse=True)
    
    output_string = ""
    
    while input_string:
        matched = False
        for key in sorted_keys:
            if input_string.startswith(key):
                output_string += my_transliteration[key]
                input_string = input_string[len(key):]
                matched = True
                break
                
        if not matched:
            output_string += input_string[0]
            input_string = input_string[1:]
                
    return output_string
