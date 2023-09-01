from PySide2.QtWidgets import QWidget, QVBoxLayout, QCalendarWidget

def create_calendar_tab():
    widget = QWidget()
    layout = QVBoxLayout()
    calendar = QCalendarWidget(widget)
    layout.addWidget(calendar)
    widget.setLayout(layout)
    return widget
