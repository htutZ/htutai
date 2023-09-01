from PySide2.QtWidgets import QWidget, QVBoxLayout, QTimeEdit

def create_clock_tab():
    widget = QWidget()
    layout = QVBoxLayout()
    time_display = QTimeEdit(widget)
    layout.addWidget(time_display)
    widget.setLayout(layout)
    return widget
