from PySide2.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QGraphicsOpacityEffect)
from PySide2.QtCore import QTimer, Qt, QTime, QRect, QPropertyAnimation, QEasingCurve, QDateTime
from PySide2.QtGui import QFont, QFontDatabase

class CustomWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.resizeEvent(None)  # Initial positioning
        self.menu_shown = False
        self.animations = []  # List to hold animations
        self.effects = []  # List to hold opacity effects

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Loading the provided font
        QFontDatabase.addApplicationFont("fonts\DS-Digital-Bold.TTF")

        # Digital Clock Display
        self.digital_clock = QLabel()
        self.digital_clock.setAlignment(Qt.AlignCenter)
        self.digital_clock.setFont(QFont("DS-Digital", 60))
        self.digital_clock.setContentsMargins(0, 20, 0, 0)
    
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
            margin-bottom: -25px;
        }
        """
        self.date_label.setStyleSheet(date_label_style)

        # Timezone Display
        self.timezone_label = QLabel("(Asia/Rangoon)")
        self.timezone_label.setAlignment(Qt.AlignCenter)
        self.timezone_label.setFont(QFont("DS-Digital", 14))
        self.timezone_label.setContentsMargins(0, 0, 0, 20)
        self.timezone_label.setStyleSheet(date_label_style)

        # Container for the clock and date
        self.clock_container = QWidget()
        clock_layout = QVBoxLayout()
        clock_layout.addStretch(3)
        clock_layout.addWidget(self.digital_clock)
        clock_layout.addStretch(1)
        clock_layout.addWidget(self.date_label)
        clock_layout.addWidget(self.timezone_label)
        clock_layout.addStretch(3)
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
        layout.addWidget(self.menu_button, alignment=Qt.AlignRight)

    def resizeEvent(self, event):
        # Repositioning the menu_button and bubbles every time the widget is resized
        offset = 60
        self.menu_button.move(self.width() - self.menu_button.width() - 10, self.height() - self.menu_button.height() - 10)
        self.bubble1.move(self.menu_button.x() - offset, self.menu_button.y())
        self.bubble2.move(self.menu_button.x() - offset/1.5, self.menu_button.y() - offset/1.5)
        self.bubble3.move(self.menu_button.x(), self.menu_button.y() - offset)

    def toggle_menu(self):
        offset = 60

        if not self.menu_shown:  # If the menu is not shown, show the bubbles
            for bubble in [self.bubble1, self.bubble2, self.bubble3]:
                opacity_effect = QGraphicsOpacityEffect()
                self.effects.append(opacity_effect)
                bubble.setGraphicsEffect(opacity_effect)
                opacity_effect.setOpacity(0.0)
                bubble.show()

                opacity_anim = QPropertyAnimation(opacity_effect, b"opacity")
                self.animations.append(opacity_anim)  # Keep a reference to prevent garbage collection
                opacity_anim.setDuration(200)
                opacity_anim.setStartValue(0.0)
                opacity_anim.setEndValue(1.0)
                opacity_anim.start()

                anim = QPropertyAnimation(bubble, b"geometry")
                self.animations.append(anim)
                anim.setDuration(200)
                anim.setStartValue(bubble.geometry())
                anim.setEasingCurve(QEasingCurve.OutElastic)

                if bubble == self.bubble1:
                    anim.setEndValue(QRect(self.menu_button.x() - offset, self.menu_button.y(), 40, 40))
                elif bubble == self.bubble2:
                    anim.setEndValue(QRect(self.menu_button.x() - offset/1.5, self.menu_button.y() - offset/1.5, 40, 40))
                else:
                    anim.setEndValue(QRect(self.menu_button.x(), self.menu_button.y() - offset, 40, 40))
                anim.start()

            self.menu_button.setText("✕")
            self.menu_shown = True  # Update the state to show that the menu is now shown
        else:  # If the menu is shown, hide the bubbles
            for bubble in [self.bubble1, self.bubble2, self.bubble3]:
                opacity_effect = bubble.graphicsEffect()

                opacity_anim = QPropertyAnimation(opacity_effect, b"opacity")
                self.animations.append(opacity_anim)  # Keep a reference to prevent garbage collection
                opacity_anim.setDuration(200)
                opacity_anim.setStartValue(1.0)
                opacity_anim.setEndValue(0.0)
                opacity_anim.finished.connect(bubble.hide)
                opacity_anim.start()

                anim = QPropertyAnimation(bubble, b"geometry")
                self.animations.append(anim)
                anim.setDuration(200)
                anim.setStartValue(bubble.geometry())
                anim.setEasingCurve(QEasingCurve.OutElastic)

                if bubble == self.bubble1:
                    anim.setEndValue(QRect(self.menu_button.x() - offset, self.menu_button.y(), 40, 40))
                elif bubble == self.bubble2:
                    anim.setEndValue(QRect(self.menu_button.x() - offset/1.5, self.menu_button.y() - offset/1.5, 40, 40))
                else:
                    anim.setEndValue(QRect(self.menu_button.x(), self.menu_button.y() - offset, 40, 40))
                anim.start()

            self.menu_button.setText("☰")
            self.menu_shown = False 

    def update_time_and_date(self):
        current_time = QTime.currentTime()
        self.digital_clock.setText(current_time.toString("hh:mm:ss"))
        current_date = QDateTime.currentDateTime()
        self.date_label.setText(current_date.toString("dddd, MMMM d, yyyy"))

def create_clock_tab():
    return CustomWidget()
