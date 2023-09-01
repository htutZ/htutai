from PySide2.QtWidgets import (QApplication, QMainWindow, QTabWidget, QTabBar, 
                               QCalendarWidget, QVBoxLayout, QWidget)
from PySide2.QtGui import QFontDatabase, QFont, QIcon, QColor, QPainter, QBrush
from PySide2.QtCore import Qt, QTimer, Slot
from language_detector import detect_language
from functions.load_responses import load_responses_from_database
from functions.transliterate import transliterate
from functions.myanglish_to_burmese import myanglish_to_burmese
from functions.improved_transliterate import improved_transliterate
from ui import chat_tab, clock_tab, calendar_tab, menu_tab, music_tab
import sys
import torch
import torch.nn as nn

queries, responses, response_category_mapping = load_responses_from_database()

vocab = set(' '.join(queries))
vocab_size = 21
print(f"Vocabulary Size: {vocab_size}")
index_to_char = dict(enumerate(vocab))
char_to_index = {char: idx for idx, char in index_to_char.items()}

def string_to_onehot(string, max_length=50):
    tensor = torch.zeros(max_length, vocab_size)
    for li, letter in enumerate(string):
        if letter in char_to_index:
            tensor[li][char_to_index[letter]] = 1
    return tensor

input_size = 50 * vocab_size  
print(f"Input Size: {input_size}")
hidden_size = 128
output_size = len(responses)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

model = SimpleNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

class VirtualAssistant:
    def __init__(self, model):
        self.model = model

    def respond_to_query(self, user_query):
        with torch.no_grad():
            user_query = user_query.lower()
            input_tensor = string_to_onehot(user_query)
            prediction = self.model(input_tensor.view(1, -1))
            response_idx = torch.argmax(prediction).item()
            if response_idx < len(responses):
                response = responses[response_idx]
                category = response_category_mapping[response_idx]
                print(f"Response taken from category: {category}")
                return response
            else:
                return "I'm not sure what you mean."


class AssistantApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Assistant')
        self.setGeometry(100, 100, 500, 400)
        self.setMinimumSize(400, 300)
        self.setWindowIcon(QIcon('icon.png'))

        self.tab_widget = QTabWidget(self)

        self.tab_widget.addTab(chat_tab.create_chat_tab(), "Chat")
        self.tab_widget.addTab(clock_tab.create_clock_tab(), "Clock")
        self.tab_widget.addTab(calendar_tab.create_calendar_tab(), "Calendar")
        self.tab_widget.addTab(music_tab.create_music_tab(), "Music")
        self.tab_widget.addTab(menu_tab.create_menu_tab(), "Menu")


        self.setCentralWidget(self.tab_widget)

        QApplication.setStyle('Fusion')
        
    @Slot()
    def handle_query(self):
        original_user_query = self.user_input.text()
        language_used = detect_language(original_user_query)

        user_query = original_user_query
        if language_used == "Myanglish":
             user_query = improved_transliterate(original_user_query)
             burmese_translation = myanglish_to_burmese(user_query)
             user_query = burmese_translation

        assistant = VirtualAssistant(model)
        response = assistant.respond_to_query(user_query)
        self.response_label.setText(response)

if __name__ == "__main__":
    app = QApplication([])
    window = AssistantApp()
    window.show()
    app.exec_()