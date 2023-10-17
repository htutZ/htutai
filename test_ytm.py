from PySide2.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from PySide2.QtCore import QTimer, QTime, Qt

class ClockWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Digital Clock Display
        self.digital_clock = QLabel()
        self.digital_clock.setAlignment(Qt.AlignCenter)
        
        # Start timer to update the clock every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()  # Initial update

        layout.addWidget(self.digital_clock)
        self.setLayout(layout)

    def update_time(self):
        current_time = QTime.currentTime()
        
        # Use HTML and CSS to style the clock
        text = f"""
        <div style="color: #555; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
            <h1 style="margin: 0; font-family: 'Arial'; font-size: 48px;">{current_time.toString("hh:mm:ss")}</h1>
        </div>
        """
        
        self.digital_clock.setText(text)

if __name__ == '__main__':
    app = QApplication([])
    window = ClockWidget()
    window.resize(300, 150)
    window.show()
    app.exec_()
