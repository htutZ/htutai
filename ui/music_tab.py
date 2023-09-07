from PySide2.QtWidgets import (QListWidget, QVBoxLayout, QPushButton, QWidget, QTextEdit, QHBoxLayout, QComboBox,
                               QLineEdit, QLabel, QListWidgetItem, QStyle, QSlider, QApplication)
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtCore import QUrl, QFile, QTextStream, Qt, QTimer
from PySide2.QtGui import QImage, QPixmap, QIcon, QPalette, QColor
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from ytmusicapi import YTMusic
import pytube
from pytube import Search, YouTube

import vlc
import pythoncom
import yt_dlp
import requests
import logging
logging.getLogger('pytube').setLevel(logging.CRITICAL)

os.environ["PAFY_BACKEND"] = "yt_dlp"
ytmusic = YTMusic('headers_auth.json')
# Set up Spotify credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="92b308edd46a4134894df230bf543670",
                                                           client_secret="9ef2001ccf21447f937679801d935b3d"))



def search_youtube(query, max_results=10):
    search_results = Search(query).results[:max_results]
    return [(video.title, video.video_id) for video in search_results]

def search_spotify(query):
    results = sp.search(q=query, limit=5)
    return [track['name'] for track in results['tracks']['items']]

def search_ytm(query):
    search_results = ytmusic.search(query, filter="songs")
    if search_results:
        return search_results[0]['videoId']
    else:
        return None
    
def get_stream_url(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = pytube.YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    return stream.url

def create_music_tab():

        # Dark theme settings
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    QApplication.setPalette(dark_palette)
    QApplication.setStyle("Fusion")

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

    def get_thumbnail_as_pixmap(url):
        image = QImage()
        image.loadFromData(requests.get(url).content)
        pixmap = QPixmap(image)
        return pixmap
    
    def search_song():
        query = search_bar.text()
        song_list_widget.clear()
        youtube_results = search_youtube(query)

        for title, video_id in youtube_results:
            song_database[title] = video_id

            item = QListWidgetItem(title)
            song_list_widget.addItem(item)
            print(f"Video ID for '{title}': {video_id}")

            thumbnail_url = f"http://i4.ytimg.com/vi/{video_id}/default.jpg"
            thumbnail = get_thumbnail_as_pixmap(thumbnail_url)
            item.setIcon(QIcon(thumbnail))

    instance = vlc.Instance()
    player = instance.media_player_new() 
    is_playing = False

    progress_bar = QSlider(Qt.Horizontal)
    layout.addWidget(progress_bar)

    progress_timer = QTimer()
    progress_timer.timeout.connect(lambda: progress_bar.setValue(player.get_position() * 1000))

    def play_selected_song():
        nonlocal is_playing
        try:
            selected_song = song_list_widget.currentItem().text()
            video_id = song_database.get(selected_song)

            if not video_id:
                print(f"No video ID found for song: {selected_song}")
                return

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
              }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                audio_url = info_dict['url']

                title_label.setText(info_dict.get('title', 'Unknown Title'))
                duration_label.setText(f"Duration: {info_dict.get('duration', 'N/A')} seconds")
                thumbnail_label.setPixmap(get_thumbnail_as_pixmap(info_dict['thumbnail']))

                pythoncom.CoInitialize()

                media = instance.media_new(audio_url)
                player.set_media(media)
                player.play()

                play_pause_button.setText("▶️")
                is_playing = True
                progress_timer.start(1000)  # Update every second
                
        except Exception as e:
            print(f"Error playing song: {e}")

    def toggle_play_pause():
        nonlocal is_playing
        if is_playing:
            player.pause()
            play_pause_button.setText("▶️")
            progress_timer.stop()
        else:
            if player.get_media():
                player.play()
                play_pause_button.setText("⏸︎")
                progress_timer.start(1000)
            else:
                play_selected_song()
        is_playing = not is_playing    

    def play_next_song():
        current_row = song_list_widget.currentRow()
        if current_row < song_list_widget.count() - 1:
            song_list_widget.setCurrentRow(current_row + 1)
        elif loop_control.currentText() == "Repeat All":
            song_list_widget.setCurrentRow(0)
        play_selected_song()

    def play_prev_song():
        current_row = song_list_widget.currentRow()
        if current_row > 0:
            song_list_widget.setCurrentRow(current_row - 1)
        play_selected_song()

    play_pause_button.clicked.connect(toggle_play_pause)
    next_song_button.clicked.connect(play_next_song)
    prev_song_button.clicked.connect(play_prev_song)
    search_button.clicked.connect(search_song)
    play_pause_button.setText("▶️")  # Initialize with play icon
    next_song_button.setText("⏭️")
    prev_song_button.setText("⏮️")

    widget.setLayout(layout)
    return widget
