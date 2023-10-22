from PySide2.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGraphicsOpacityEffect)
from PySide2.QtCore import QTimer, Qt, QTime, QRect, QPropertyAnimation, QEasingCurve, QDateTime, QTimeZone
from PySide2.QtGui import QFont, QFontDatabase
import datetime

<<<<<<< HEAD
def get_spotify_token(client_id, client_secret):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    })
    auth_response_data = auth_response.json()
    return auth_response_data['access_token']

def search_for_track_id(token, track_name, artist_name):
    search_url = 'https://api.spotify.com/v1/search'
    query = f"{track_name} artist:{artist_name}"
    params = {
        'q': query,
        'type': 'track',
        'limit': 1
    }
    headers = {
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(search_url, headers=headers, params=params)
    results = response.json()
    if results['tracks']['items']:
        return results['tracks']['items'][0]['id']
    return None

def get_recommendations_by_track_id(token, track_id):
    recommendations_url = 'https://api.spotify.com/v1/recommendations'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    params = {
        'seed_tracks': track_id
    }
    response = requests.get(recommendations_url, headers=headers, params=params)
    return response.json()

# Example Usage
client_id = '27b9d621b0344860b22480a9a0042240'
client_secret = '8ca1c330100b4224b2b774383dffdfa7'

token = get_spotify_token(client_id, client_secret)

# For demonstration purposes, let's use a well-known song:
track_name = "Shape of You"
artist_name = "Ed Sheeran"

track_id = search_for_track_id(token, track_name, artist_name)
if track_id:
    recommendations = get_recommendations_by_track_id(token, track_id)
    for track in recommendations['tracks']:
        print(track['name'], "-", track['artists'][0]['name'])
else:
    print(f"No track found for {track_name} by {artist_name}.")

# Count on Me - Bruno Mars
# Mama - Jonas Blue
# Treat You Better - Shawn Mendes
# New Man - Ed Sheeran
# Watermelon Sugar - Harry Styles
# Hymn for the Weekend - Coldplay
# Rockabye (feat. Sean Paul & Anne-Marie) - Clean Bandit
# Love Me Now - John Legend
# Don't Start Now - Dua Lipa
# September Song - JP Cooper
# The Middle - Zedd
# Meant to Be (feat. Florida Georgia Line) - Bebe Rexha
# Never Be the Same - Camila Cabello
# Run (feat. Ed Sheeran) (Taylor’s Version) (From The Vault) - Taylor Swift
# Thinking out Loud - Alex Adair Remix - Ed Sheeran
# Say Something (feat. Chris Stapleton) - Justin Timberlake
# Don't Call Me Up - Mabel
# Rooftop - Nico Santos
# All About That Bass - Meghan Trainor
# Let Me Hold You (Turn Me On) -  Cheat Codes
=======
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
        self.clock_container = QWidget(self)
        self.clock_container.setFixedSize(1345, 250)    
        self.digital_clock = QLabel(self.clock_container)
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
       
        # Style for the date label
        date_label_style = """
        QLabel {
            color: cyan;
            background-color: transparent;
            border: none;
        }
        """

        # Date Display
        self.date_label = QLabel(self.clock_container)
        self.date_label.setFont(QFont("DS-Digital", 20))  # Made this larger than the timezone label
        self.date_label.setStyleSheet(date_label_style)

        # Timezone Display
        self.timezone_label = QLabel(self.clock_container)
        self.timezone_label.setFont(QFont("DS-Digital", 15))  # Made this smaller
        self.timezone_label.setStyleSheet(date_label_style)

        # Adjusting the size and position of the labels
        container_width = 1345
        self.digital_clock.setGeometry((container_width - 400) // 2, 67, 400, 80)
        self.date_label.setGeometry((container_width - 290) // 2, 147, 400, 50)
        self.timezone_label.setGeometry((container_width - 215) // 2, 187, 400, 30)

        # Style for the container box that holds both clock and date
        container_style = """
        QWidget {
            background-color: black;
            border: 3px solid cyan;
            border-radius: 15px;
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
        
        # Getting the timezone information and setting it to timezone_label
        tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        self.timezone_label.setText(tz.tzname(None))

def create_clock_tab():
    return CustomWidget()

if __name__ == "__main__":
    app = QApplication([])
    window = CustomWidget()
    window.show()
    app.exec_()
>>>>>>> 6291af202be8f6b65c29043db0c1d394200629f0
