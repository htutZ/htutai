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
from colorthief import ColorThief
from io import BytesIO
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


def dominant_color_from_url(url):
    """Get the dominant color from an image URL."""
    response = requests.get(url)
    color_thief = ColorThief(BytesIO(response.content))
    dominant_color = color_thief.get_color(quality=1)
    return QColor(*dominant_color)
    
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

def format_time(ms):
    """Convert milliseconds into MM:SS format."""
    s = ms // 1000
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"

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

    # Playback info layout
    playback_info_layout = QHBoxLayout()
    thumbnail_label_playback = QLabel()
    title_label_playback = QLabel()
    play_pause_button_playback = QPushButton("▶️")

    # Main controls
    controls_layout = QHBoxLayout()
    play_pause_button = QPushButton("▶️")
    next_song_button = QPushButton("⏭️")
    prev_song_button = QPushButton("⏮️")
    loop_control = QComboBox()
    loop_control.addItems(["No Loop", "Repeat One", "Repeat All"])
    
    controls_layout.addWidget(play_pause_button)
    controls_layout.addWidget(prev_song_button)
    controls_layout.addWidget(next_song_button)
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
    player.audio_set_volume(60)  # Set the default volume to 60
    is_playing = False

    progress_bar = QSlider(Qt.Horizontal)
    # Current time label
    current_time_label = QLabel("00:00")
    layout.addWidget(current_time_label, alignment=Qt.AlignLeft)

    # Total duration label
    total_duration_label = QLabel("00:00")
    layout.addWidget(total_duration_label, alignment=Qt.AlignRight)

    # Adjusting the Music bar layout
    music_bar_layout = QVBoxLayout()

    playback_controls_layout = QHBoxLayout()
    playback_controls_layout.addWidget(thumbnail_label_playback)
    playback_controls_layout.addWidget(title_label_playback, 1)
    playback_controls_layout.addWidget(play_pause_button_playback)

    music_bar_layout.addLayout(playback_controls_layout)
    music_bar_layout.addWidget(progress_bar)
    music_bar_layout.addWidget(current_time_label, alignment=Qt.AlignLeft)
    music_bar_layout.addWidget(progress_bar, alignment=Qt.AlignCenter)
    music_bar_layout.addWidget(total_duration_label, alignment=Qt.AlignRight)


    # Create a new QWidget for the music bar and set the layout
    music_bar_widget = QWidget()
    music_bar_widget.setLayout(music_bar_layout)

    # Removing old widgets and layouts
    layout.removeWidget(play_pause_button_playback)
    layout.removeWidget(title_label_playback)

    # Add to main layout
    layout.addWidget(music_bar_widget)
    progress_timer = QTimer()

    # Progress bar mouse click event
    def progress_bar_clicked(event):
        fraction = event.x() / progress_bar.width()
        new_value = fraction * progress_bar.maximum()
        progress_bar.setValue(int(new_value))
        player.set_time(int(new_value))

    progress_bar.mousePressEvent = progress_bar_clicked

    # Update the progress bar and current time label every second
    def update_progress():
        current_time = player.get_time()
        progress_bar.setValue(current_time)
        current_time_label.setText(format_time(current_time))

    progress_timer.timeout.connect(update_progress)

        # This function will be called when the user adjusts the progress bar
    def update_player_position(value):
        player.set_time(value)  # Convert from milliseconds to seconds

    progress_bar.sliderMoved.connect(update_player_position)

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
            play_pause_button.setText("▶️")
            progress_timer.stop()
            is_playing = False
        else:
            if not player.get_media():
                play_selected_song()
            else:
                player.play()
                progress_timer.start(1000)
            play_pause_button.setText("⏸︎")    
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

    play_pause_button.clicked.connect(toggle_play_pause)
    next_song_button.clicked.connect(play_next_song)
    prev_song_button.clicked.connect(play_prev_song)
    search_button.clicked.connect(search_song)
    play_pause_button_playback.clicked.connect(toggle_play_pause)
    play_pause_button.setText("▶️")
    next_song_button.setText("⏭️")
    prev_song_button.setText("⏮️")

    widget.setLayout(layout)
    return widget
