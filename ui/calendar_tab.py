from PySide2.QtWidgets import QWidget, QVBoxLayout, QCalendarWidget, QLabel
from PySide2.QtCore import Qt
from PySide2.QtGui import QFont

def create_calendar_tab():
    widget = QWidget()
    layout = QVBoxLayout()

    # Title
    titleLabel = QLabel("Calendar")
    titleLabel.setAlignment(Qt.AlignCenter)
    titleLabel.setFont(QFont("Arial", 24, QFont.Bold))
    
    # Calendar
    calendar = QCalendarWidget()
    calendar.setStyleSheet("""
        QCalendarWidget {
            border: none;
            background: #f0f0f0;
        }
        QCalendarWidget QWidget {
            alternate-background-color: #f5f5f5;
        }
        QCalendarWidget QAbstractItemView:enabled {
            font-size: 14px;
            color: #484848;
            background: #ffffff;
            border: none;
            border-radius: 5px;
        }
        QCalendarWidget QAbstractItemView:disabled {
            color: #b0b0b0;
        }
        QCalendarWidget QToolButton {
            border: none;
            background: transparent;
            icon-size: 20px, 20px;
        }
        QCalendarWidget QMenu {
            background: #ffffff;
        }
        QCalendarWidget QHeaderView {
            background-color: #f5f5f5;
            border: 1px solid #d0d0d0;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
    """)

    layout.addWidget(titleLabel)
    layout.addWidget(calendar)
    layout.setSpacing(20)
    layout.setContentsMargins(10, 10, 10, 10)

    widget.setLayout(layout)
    return widget
