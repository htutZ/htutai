from PySide2.QtWidgets import QListWidget, QVBoxLayout, QPushButton, QWidget
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtCore import QUrl
import os

def create_music_tab():
    widget = QWidget()
    layout = QVBoxLayout()

    player = QMediaPlayer()

    song_list_widget = QListWidget()
    layout.addWidget(song_list_widget)

    play_song_button = QPushButton("Play Selected Song", widget)
    layout.addWidget(play_song_button)

    song_database = {}

    app_dir = os.path.dirname(os.path.abspath(__file__))
    song_folder = os.path.join(app_dir, 'songs')
    if not os.path.exists(song_folder):
        os.makedirs(song_folder)

    for song in os.listdir(song_folder):
        if song.endswith('.mp3'):
            song_database[song] = os.path.join(song_folder, song)
            song_list_widget.addItem(song)

    def play_selected_song():
        selected_song = song_list_widget.currentItem().text()
        song_path = song_database[selected_song]
        player.setMedia(QMediaContent(QUrl.fromLocalFile(song_path)))
        player.play()

    play_song_button.clicked.connect(play_selected_song)

    widget.setLayout(layout)
    return widget
