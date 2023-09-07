from pytube import Search
from PySide2.QtWidgets import (QListWidget, QVBoxLayout, QPushButton, QWidget, QTextEdit, QHBoxLayout, QComboBox, 
                               QLineEdit, QLabel)
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtGui import QPixmap
from PySide2.QtCore import QUrl, QFile, QTextStream
import os
import pytube

def search_youtube(query, max_results=10):
    search_results = Search(query).results[:max_results]
    return [(video.title, video.video_id) for video in search_results]

def get_stream_url(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = pytube.YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    return stream.url

def create_music_tab():
    widget = QWidget()
    layout = QVBoxLayout()
    controls_layout = QHBoxLayout()

    player = QMediaPlayer()
    loop_modes = ["No Loop", "Repeat One", "Repeat All"]

    # Search bar
    search_bar = QLineEdit(widget)
    search_bar.setPlaceholderText('Search for songs...')
    layout.addWidget(search_bar)

    # Search button
    search_button = QPushButton("Search", widget)
    layout.addWidget(search_button)

    # Music list
    song_list_widget = QListWidget()
    layout.addWidget(song_list_widget)

    # Lyrics display
    lyrics_display = QTextEdit()
    lyrics_display.setReadOnly(True)
    layout.addWidget(lyrics_display)

    # Play song button
    play_pause_button = QPushButton("Play", widget)
    controls_layout.addWidget(play_pause_button)

    # Next and Previous buttons
    next_song_button = QPushButton("Next", widget)
    controls_layout.addWidget(next_song_button)

    prev_song_button = QPushButton("Previous", widget)
    controls_layout.addWidget(prev_song_button)

    # Loop control
    loop_control = QComboBox()
    loop_control.addItems(loop_modes)
    controls_layout.addWidget(loop_control)

    layout.addLayout(controls_layout)

    song_database = {}

    app_dir = os.path.dirname(os.path.abspath(__file__))
    song_folder = os.path.join(app_dir, 'songs')
    if not os.path.exists(song_folder):
        os.makedirs(song_folder)

    for song in os.listdir(song_folder):
        if song.endswith('.mp3'):
            song_database[song] = os.path.join(song_folder, song)
            song_list_widget.addItem(song)

    # Adding to the UI:
    title_label = QLabel()
    layout.addWidget(title_label)

    artist_label = QLabel()
    layout.addWidget(artist_label)

    thumbnail_label = QLabel()
    layout.addWidget(thumbnail_label)

    duration_label = QLabel("Duration:")
    layout.addWidget(duration_label)

    def search_song():
        query = search_bar.text()
        song_list_widget.clear()
        youtube_results = search_youtube(query)
        for title, video_id in youtube_results:
            song_list_widget.addItem(title)
            song_database[title] = video_id
            print(f"Video ID for '{title}': {video_id}")

    def play_selected_song():
        selected_song = song_list_widget.currentItem().text()
        stream_url = get_stream_url(selected_song)
        
        if stream_url:
            player.setMedia(QMediaContent(QUrl(stream_url)))
            player.play()

            title_label.setText("Title: " + selected_song)
            artist_label.setText("Artist: ")        

    song_list_widget.itemDoubleClicked.connect(play_selected_song)
    search_button.clicked.connect(search_song)

    widget.setLayout(layout)
    return widget
