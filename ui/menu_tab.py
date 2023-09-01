from PySide2.QtWidgets import QWidget, QVBoxLayout, QLabel

def create_menu_tab():
    widget = QWidget()
    layout = QVBoxLayout()
    label = QLabel("Menu content goes here", widget)
    layout.addWidget(label)
    widget.setLayout(layout)
    return widget
