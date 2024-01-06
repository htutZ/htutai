from PySide2.QtWidgets import QVBoxLayout, QLineEdit, QPushButton, QLabel, QWidget
from PySide2.QtGui import QFont
from PySide2.QtCore import QThread, Signal

from PySide2.QtGui import QFont
from PySide2.QtCore import QThread, Signal

from language_detector import detect_language
from functions.transliterate import transliterate
from functions.myanglish_to_burmese import myanglish_to_burmese
from functions.improved_transliterate import improved_transliterate
from app import model, VirtualAssistant


class QueryProcessor(QThread):
    response_signal = Signal(str)

    def __init__(self, user_query, language_used, assistant_model):
        super().__init__()
        self.user_query = user_query
        self.language_used = language_used
        self.assistant_model = assistant_model

    def run(self):
        try:
            response = self.assistant_model.respond_to_query(self.user_query)
            self.response_signal.emit(response)
        except Exception as e:
            self.response_signal.emit(f"Error: {str(e)}")

def create_chat_tab():
    widget = QWidget()
    layout = QVBoxLayout()

    widget.user_input = QLineEdit(widget)
    widget.user_input.setPlaceholderText('Type your query here')
    widget.user_input.setFont(QFont("Arial", 14))
    layout.addWidget(widget.user_input)

    widget.query_button = QPushButton('Ask Assistant', widget)
    widget.query_button.setFont(QFont("Arial", 14))
    layout.addWidget(widget.query_button)

    widget.user_input_log = QLabel("User Input: ", widget)
    layout.addWidget(widget.user_input_log)

    widget.detected_language_log = QLabel("Detected Language: ", widget)
    layout.addWidget(widget.detected_language_log)

    widget.converted_words_log = QLabel("Converted Words: ", widget)
    layout.addWidget(widget.converted_words_log)

    widget.response_label = QLabel(widget)
    widget.response_label.setFont(QFont("Pyidaungsu", 24))
    layout.addWidget(widget.response_label)

    widget.setLayout(layout)

    assistant = VirtualAssistant(model)

    widget.query_processor = None

    def handle_query():
        try:
            original_user_query = widget.user_input.text()
            widget.user_input_log.setText(f"User Input: {original_user_query}")
            
            language_used = detect_language(original_user_query)
            widget.detected_language_log.setText(f"Detected Language: {language_used}")

            user_query = original_user_query
            if language_used == "Myanglish":
                user_query = improved_transliterate(original_user_query)
                burmese_translation = myanglish_to_burmese(user_query)
                user_query = burmese_translation
                widget.converted_words_log.setText(f"Converted Words: {user_query}")
            else:
                widget.converted_words_log.setText("Converted Words: ")

            widget.query_processor = QueryProcessor(user_query, language_used, assistant)
            widget.query_processor.response_signal.connect(widget.response_label.setText)
            widget.query_processor.start()

        except Exception as e:
            widget.response_label.setText(f"Error: {str(e)}")

    widget.query_button.clicked.connect(handle_query)

    return widget