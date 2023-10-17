from PySide2.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                               QStackedWidget, QGridLayout, QGraphicsDropShadowEffect, QHBoxLayout)
from PySide2.QtCore import QTimer, Qt, QTime, QRect, QPropertyAnimation, QEasingCurve
from PySide2.QtGui import QFont, QFontDatabase, QColor

class CustomWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.resizeEvent(None)  # Initial positioning

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Loading the provided font
        QFontDatabase.addApplicationFont("fonts\DS-Digital-Bold.TTF")

        # Digital Clock Display
        self.digital_clock = QLabel()
        self.digital_clock.setAlignment(Qt.AlignCenter)
        self.digital_clock.setFont(QFont("DS-Digital", 60))
    
        # Style for the digital clock
        digital_clock_style = """
        QLabel {
            color: cyan;
            background-color: transparent;
            border: none;
        }
        """
        self.digital_clock.setStyleSheet(digital_clock_style)

        # Date Display
        self.date_label = QLabel()
        self.date_label.setAlignment(Qt.AlignCenter)
        self.date_label.setFont(QFont("DS-Digital", 20))
    
        # Style for the date label
        date_label_style = """
        QLabel {
            color: cyan;
            background-color: transparent;
            border: none;
        }
        """
        self.date_label.setStyleSheet(date_label_style)

        # Container for the clock and date
        self.clock_container = QWidget()
        clock_layout = QVBoxLayout()
        clock_layout.addWidget(self.digital_clock)
        clock_layout.addWidget(self.date_label)
        self.clock_container.setLayout(clock_layout)
    
        # Style for the container box that holds both clock and date
        container_style = """
        QWidget {
            background-color: black;
            border: 3px solid cyan;
            border-radius: 15px;
            padding: 20px;
            margin: 20px;
        }
        """
        self.clock_container.setStyleSheet(container_style)

        # Updating the time and date every second
        timer = QTimer(self)
        timer.timeout.connect(self.update_time_and_date)
        timer.start(1000)
        self.update_time_and_date()

        self.menu_button = QPushButton("☰", self)
        self.menu_button.setFixedSize(50, 50)
        self.menu_button.setStyleSheet("""
            QPushButton {
                border-radius: 25px;
                background-color: cyan;
            }
        """)

        self.bubble1 = QPushButton(self)
        self.bubble2 = QPushButton(self)
        self.bubble3 = QPushButton(self)

        for bubble in [self.bubble1, self.bubble2, self.bubble3]:
            bubble.setFixedSize(40, 40)
            bubble.setStyleSheet("""
                QPushButton {
                    border-radius: 20px;
                    background-color: cyan;
                }
            """)
            bubble.hide()

        # Connect the toggle_menu function for the menu_button
        self.menu_button.clicked.connect(self.toggle_menu)

        layout.addWidget(self.clock_container)
        layout.addStretch(1)

    def resizeEvent(self, event):
        # Repositioning the menu_button and bubbles every time the widget is resized
        offset = 60
        self.menu_button.move(self.width() - self.menu_button.width() - 10, self.height() - self.menu_button.height() - 10)
        self.bubble1.move(self.menu_button.x() - offset, self.menu_button.y())
        self.bubble2.move(self.menu_button.x() - offset/1.5, self.menu_button.y() - offset/1.5)
        self.bubble3.move(self.menu_button.x(), self.menu_button.y() - offset)

    def toggle_menu(self):
        offset = 60

        for bubble in [self.bubble1, self.bubble2, self.bubble3]:
                anim = QPropertyAnimation(bubble, b"geometry")
                anim.setDuration(500)
                anim.setStartValue(bubble.geometry())
                anim.setEasingCurve(QEasingCurve.OutElastic)
                if bubble == self.bubble1:
                    anim.setEndValue(QRect(self.menu_button.x() - offset, self.menu_button.y(), 40, 40))
                elif bubble == self.bubble2:
                    anim.setEndValue(QRect(self.menu_button.x() - offset/1.5, self.menu_button.y() - offset/1.5, 40, 40))
                else:
                    anim.setEndValue(QRect(self.menu_button.x(), self.menu_button.y() - offset, 40, 40))
                anim.start()
                bubble.show()

    def update_time_and_date(self):
        current_time = QTime.currentTime()
        self.digital_clock.setText(current_time.toString("hh:mm:ss"))
        self.date_label.setText(current_time.toString("dddd, MMMM dd, yyyy"))

def create_clock_tab():
    return CustomWidget()
