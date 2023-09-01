from PySide2.QtWidgets import QVBoxLayout, QLineEdit, QPushButton, QLabel, QWidget
from PySide2.QtGui import QFont
from language_detector import detect_language
from functions.transliterate import transliterate
from functions.myanglish_to_burmese import myanglish_to_burmese
from functions.improved_transliterate import improved_transliterate

from app import model, VirtualAssistant

def create_chat_tab():
    widget = QWidget()
    layout = QVBoxLayout()
    
    # User input
    user_input = QLineEdit(widget)
    user_input.setPlaceholderText('Type your query here')
    user_input.setFont(QFont("Arial", 14))
    layout.addWidget(user_input)

    # Button
    query_button = QPushButton('Ask Assistant', widget)
    query_button.setFont(QFont("Arial", 14))
    layout.addWidget(query_button)

    # Log for user input
    user_input_log = QLabel("User Input: ", widget)
    layout.addWidget(user_input_log)

    # Log for detected language
    detected_language_log = QLabel("Detected Language: ", widget)
    layout.addWidget(detected_language_log)

    # Log for converted words
    converted_words_log = QLabel("Converted Words: ", widget)
    layout.addWidget(converted_words_log)

    # Label for response
    response_label = QLabel(widget)
    response_label.setFont(QFont("Pyidaungsu", 24))
    layout.addWidget(response_label)

    widget.setLayout(layout)

    # Connect the button to the query handler
    def handle_query():
        original_user_query = user_input.text()

        # Update user input log
        user_input_log.setText(f"User Input: {original_user_query}")
        
        language_used = detect_language(original_user_query)
        
        # Update detected language log
        detected_language_log.setText(f"Detected Language: {language_used}")

        user_query = original_user_query
        if language_used == "Myanglish":
            user_query = improved_transliterate(original_user_query)
            burmese_translation = myanglish_to_burmese(user_query)
            user_query = burmese_translation

            # Update converted words log
            converted_words_log.setText(f"Converted Words: {user_query}")
        else:
            # Clear the converted words log if not Myanglish
            converted_words_log.setText("Converted Words: ")

        assistant = VirtualAssistant(model)
        response = assistant.respond_to_query(user_query)
        response_label.setText(response)

    query_button.clicked.connect(handle_query)

    return widget
