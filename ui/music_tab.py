from PySide2.QtWidgets import (QListWidget, QVBoxLayout, QPushButton, QWidget, QTextEdit, QHBoxLayout, QComboBox,
                               QLineEdit, QLabel, QListWidgetItem, QStyle, QSlider, QApplication, QSizePolicy)
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtCore import QUrl, QFile, QTextStream, Qt, QTimer, QSize, QThreadPool, QRunnable, Signal, QObject
from PySide2.QtGui import QImage, QPixmap, QIcon, QPalette, QColor, QPainter

import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from ytmusicapi import YTMusic
import pytube
from pytube import Search, YouTube
from colorthief import ColorThief
from io import BytesIO
import vlc
import pythoncom
import yt_dlp
import requests
import logging
import threading

from Classes.Utilities import (dominant_color_from_url, search_youtube, search_spotify, search_ytm, 
                          get_stream_url, get_youtube_video_id, format_time)
from Classes.Widgets import CustomSlider, AnimatedLabel
from Classes.SignalsAndWorkers import Worker, SongFoundSignal

logging.getLogger('pytube').setLevel(logging.CRITICAL)

os.environ["PAFY_BACKEND"] = "yt_dlp"
ytmusic = YTMusic('headers_auth.json')

def create_music_tab():

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(0, 0, 0))  # Black
    dark_palette.setColor(QPalette.WindowText, QColor(0, 255, 255))  # Cyan
    dark_palette.setColor(QPalette.Base, QColor(0, 0, 0))  # Black
    dark_palette.setColor(QPalette.AlternateBase, QColor(0, 0, 0))  # Black
    dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 255, 255))  # Cyan
    dark_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))  # Black
    dark_palette.setColor(QPalette.Text, QColor(0, 255, 255))  # Cyan
    dark_palette.setColor(QPalette.Button, QColor(0, 0, 0))  # Black
    dark_palette.setColor(QPalette.ButtonText, QColor(0, 255, 255))  # Cyan
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(0, 255, 255))  # Cyan
    dark_palette.setColor(QPalette.Highlight, QColor(0, 255, 255))  # Cyan
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0)) 
    QApplication.setPalette(dark_palette)
    QApplication.setStyle("Fusion")

    widget = QWidget()
    layout = QVBoxLayout()

    # Loading indicator
    loading_label = QLabel("Loading...", widget)
    loading_label.hide()
    layout.addWidget(loading_label)

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

    # Playback info layout
    playback_info_layout = QHBoxLayout()
    thumbnail_label_playback = QLabel()
    title_label_playback = AnimatedLabel()
    play_pause_button_playback = QPushButton()
    play_pause_button_playback.setIcon(QIcon("./icons/play.png"))

    song_database = {}

    app_dir = os.path.dirname(os.path.abspath(__file__))
    song_folder = os.path.join(app_dir, 'songs')
    if not os.path.exists(song_folder):
        os.makedirs(song_folder)

    for song in os.listdir(song_folder):
        if song.endswith('.mp3'):
            song_database[song] = os.path.join(song_folder, song)
            song_list_widget.addItem(song)

    artist_label = QLabel()
    layout.addWidget(artist_label)

    def get_thumbnail_as_pixmap(url):
        image = QImage()
        image.loadFromData(requests.get(url).content)
        pixmap = QPixmap(image)
        return pixmap
    
    def search_song():
        query = search_bar.text()
        song_list_widget.clear()
        loading_label.show()


        def add_song_to_list(title, icon):
            item = QListWidgetItem(title)
            item.setIcon(icon)
            song_list_widget.addItem(item)

        song_signal = SongFoundSignal()
        song_signal.song_found.connect(add_song_to_list)

        def search_worker(dummy_signal_instance):
            try:
                spotify_results = search_spotify(query)
                for title, spotify_thumbnail_url in spotify_results:
                    if title not in song_database:
                        video_id = get_youtube_video_id(title)
                        if video_id:
                            song_database[title] = video_id
                            thumbnail = get_thumbnail_as_pixmap(spotify_thumbnail_url)
                            icon = QIcon(thumbnail)
                            song_signal.song_found.emit(title, icon)
            except Exception as e:
                print(f"Error in search worker: {e}")
            finally:
                loading_label.hide()

        dummy_signal = SongFoundSignal()
        task = Worker(search_worker, dummy_signal)
        QThreadPool.globalInstance().start(task)

    instance = vlc.Instance()
    player = instance.media_player_new() 
    player.audio_set_volume(60)  # Set the default volume to 60
    is_playing = False

    progress_bar = CustomSlider(player, Qt.Horizontal)


    # Adjusting the Music bar layout
    music_bar_layout = QVBoxLayout()

        # Current time label
    current_time_label = QLabel("00:00")

    # Total duration label
    total_duration_label = QLabel("00:00")

        # Main controls
    controls_layout = QHBoxLayout()
    play_pause_button = QPushButton("▶️")
    next_song_button = QPushButton("⏭️")
    prev_song_button = QPushButton("⏮️")
    loop_control = QComboBox()
    loop_control.addItems(["No Loop", "Repeat One", "Repeat All"])

    container_widget = QWidget()
    container_widget.setFixedWidth(450)  # Set a width based on your requirements
# Create a layout for the container
    container_layout = QHBoxLayout()
    container_layout.setSpacing(10) 
    
    container_widget.setLayout(container_layout)
    container_layout.setContentsMargins(0, 0, 0, 0)  # This sets the margins to zero

    # Remove QLabel's internal margins
    thumbnail_label_playback.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

# Add the thumbnail and title to the container's layout
    container_layout.addWidget(thumbnail_label_playback)
    container_layout.addWidget(title_label_playback)

    # Playback info and controls
    playback_controls_layout = QHBoxLayout()
    playback_controls_layout.addWidget(container_widget) 
    playback_controls_layout.setContentsMargins(0, 0, 0, 0)
    playback_controls_layout.setSpacing(0)
    # Adding spacers to push the controls to the center a bit
    playback_controls_layout.addStretch(1.5)  # Adds some space before controls
    playback_controls_layout.addWidget(prev_song_button)
    playback_controls_layout.addWidget(play_pause_button_playback)
    playback_controls_layout.addWidget(next_song_button)
    playback_controls_layout.addWidget(loop_control)
    playback_controls_layout.addStretch(2)

    progress_layout = QHBoxLayout()
    progress_layout.addWidget(current_time_label)
    progress_layout.addWidget(progress_bar)
    progress_layout.addWidget(total_duration_label)

    music_bar_layout.addLayout(playback_controls_layout)
    music_bar_layout.addLayout(progress_layout)

    # Create a new QWidget for the music bar and set the layout
    music_bar_widget = QWidget()
    music_bar_widget.setLayout(music_bar_layout)

    # Add to main layout
    layout.addWidget(music_bar_widget)
    progress_timer = QTimer()

    # Update the progress bar and current time label every second
    def update_progress():
        current_time = player.get_time()
        progress_bar.setValue(current_time)
        current_time_label.setText(format_time(current_time))

    progress_timer.timeout.connect(update_progress)

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

                # Set the progress bar's maximum value to the song's duration
                song_duration = info_dict.get('duration', 0)  # in seconds
                progress_bar.setMaximum(song_duration * 1000)  # convert to ms
                total_duration_label.setText(format_time(song_duration * 1000))
                progress_timer.timeout.connect(lambda: progress_bar.setValue(player.get_time()))
                
                # Update playback info box
                thumbnail_url = f"http://i4.ytimg.com/vi/{video_id}/default.jpg"
                thumbnail = get_thumbnail_as_pixmap(thumbnail_url)
                thumbnail_label_playback.setPixmap(thumbnail.scaled(50, 50, Qt.KeepAspectRatio))  # Small thumbnail
                title_label_playback.setText(info_dict.get('title', 'Unknown Title'))
                dominant_color = dominant_color_from_url(thumbnail_url)
                music_bar_widget.setStyleSheet(f"background-color: rgb({dominant_color.red()}, {dominant_color.green()}, {dominant_color.blue()});")

                pythoncom.CoInitialize()

                media = instance.media_new(audio_url)
                player.set_media(media)
                player.play()

                is_playing = True
                progress_timer.start(1000)  # Update every second
                
        except Exception as e:
            print(f"Error playing song: {e}")

    def toggle_play_pause():
        nonlocal is_playing
        if player.get_state() == vlc.State.Playing:
            player.pause()
            play_pause_button_playback.setIcon(QIcon("./icons/play.png"))
            play_pause_button.setIcon(QIcon("./icons/play.png"))
            progress_timer.stop()
            is_playing = False
        else:
            if not player.get_media():
                play_selected_song()
            else:
                player.play()
                progress_timer.start(1000)
            play_pause_button_playback.setIcon(QIcon("./icons/pause.png"))
            play_pause_button.setIcon(QIcon("./icons/pause.png"))
            is_playing = True

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

    song_list_widget.itemDoubleClicked.connect(play_selected_song)
    play_pause_button.clicked.connect(toggle_play_pause)
    next_song_button.clicked.connect(play_next_song)
    prev_song_button.clicked.connect(play_prev_song)
    search_button.clicked.connect(search_song)
    play_pause_button_playback.clicked.connect(toggle_play_pause)
    next_song_button.setText("⏭️")
    prev_song_button.setText("⏮️")

    widget.setLayout(layout)
    return widget