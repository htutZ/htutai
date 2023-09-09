from PySide2.QtWidgets import (QApplication, QMainWindow, QTabWidget, QTabBar, 
                               QCalendarWidget, QVBoxLayout, QWidget, QSizePolicy)
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

        # Load Icons
        chat_icon = QIcon('./icons/chat.png')
        clock_icon = QIcon('./icons/clock.png')
        calendar_icon = QIcon('./icons/calendar.png')
        music_icon = QIcon('./icons/music.png')
        menu_icon = QIcon('./icons/menu.png')


        # Load the screen dimensions
        screen = app.primaryScreen()
        screen_size = screen.size()

        # App window properties
        self.setWindowTitle('Assistant')
        self.resize(screen_size)
        self.setWindowIcon(QIcon('icon.png'))

        # Styling the app
        self.init_ui()

        # Set up Tabs
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Tabs
        self.tab_widget.addTab(chat_tab.create_chat_tab(), QIcon(chat_icon), "")
        self.tab_widget.addTab(clock_tab.create_clock_tab(), QIcon(clock_icon), "")
        self.tab_widget.addTab(calendar_tab.create_calendar_tab(), QIcon(calendar_icon), "")
        self.tab_widget.addTab(music_tab.create_music_tab(), QIcon(music_icon), "")
        self.tab_widget.addTab(menu_tab.create_menu_tab(), QIcon(menu_icon), "")
        
        # Connect signal for tab change to update text
        self.tab_widget.currentChanged.connect(self.update_tab_text)

        self.setCentralWidget(self.tab_widget)

    def update_tab_text(self, index):
        # Reset all tabs to have no text
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabText(i, "")

        # Set text for the selected tab
        if index == 0:
            self.tab_widget.setTabText(index, "  Chat")
        elif index == 1:
            self.tab_widget.setTabText(index, "  Clock")
        elif index == 2:
            self.tab_widget.setTabText(index, "  Calendar")
        elif index == 3:
            self.tab_widget.setTabText(index, "  Music")  

        
    def init_ui(self):

        num_tabs = 5  # Adjust this if you change the number of tabs
        tab_width = 100 / num_tabs  # Calculate the width for each tab
        # Style
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #2E2E2E;
            }}
            QTabWidget::pane {{
                border: 1px solid #444;
            }}
            QTabBar::tab {{
                background: #555;
                color: white;
                padding: 10px;
                width: {tab_width}%;
            }}
            QTabBar::tab:selected {{
                background: #888;
            }}
            QPushButton {{
                background-color: #333;
                color: white;
                padding: 5px 15px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QLabel, QLineEdit {{
                color: white;
            }}
        """)

        # Font
        font_database = QFontDatabase()
        font_database.addApplicationFont("./icons/Pyidaungsu.ttf")  # Replace with the path to your font
        app_font = QFont("Pyidaungsu", 10)  # Replace with the name of your font
        self.setFont(app_font)
        
        
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