from PySide2.QtWidgets import (QListWidget, QVBoxLayout, QPushButton, QWidget, QTextEdit, QHBoxLayout, QComboBox,
                               QLineEdit, QLabel, QListWidgetItem, QStyle, QSlider, QApplication, QSizePolicy, QStackedLayout)
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtCore import QUrl, QFile, QTextStream, Qt, QTimer, QSize, QThreadPool, QRunnable, Signal, QObject, QThread
from PySide2.QtGui import QImage, QPixmap, QIcon, QPalette, QColor, QPainter, QMovie
import os
from ytmusicapi import YTMusic
import vlc
import pythoncom
import yt_dlp
import requests
import logging
import threading
import time
from fuzzywuzzy import fuzz
from Classes.Utilities import (dominant_color_from_url, search_youtube, search_spotify,search_musicapi,search_deezer_rapidapi, search_ytm,search_deezer, get_spotify_token, search_for_track_id, get_recommendations_by_track_id, get_spotify_recommendations, search_lastfm, 
                          get_stream_url, get_youtube_video_id, format_time)
from Classes.Widgets import CustomSlider, AnimatedLabel
from Classes.SignalsAndWorkers import Worker, SongFoundSignal

logging.getLogger('pytube').setLevel(logging.CRITICAL)

os.environ["PAFY_BACKEND"] = "yt_dlp"
ytmusic = YTMusic('headers_auth.json')
recommended_songs_queue = []

def filter_similar_songs(song_list):
    threshold = 85
    filtered_list = []
    
    for song in song_list:
        add_to_list = True
        for existing_song in filtered_list:
            if fuzz.ratio(song[0].split('\n')[0], existing_song[0].split('\n')[0]) > threshold and \
               fuzz.ratio(song[0].split('\n')[1], existing_song[0].split('\n')[1]) > threshold:
                add_to_list = False
                break
        if add_to_list:
            filtered_list.append(song)
    
    return filtered_list

recommended_songs_queue = []
recently_played = set()

def fetch_recommendations(song_name, artist_name):
    print(f"Fetching recommendations for {song_name} by {artist_name}")
    token = get_spotify_token()
    print(f"Spotify Token: {token}")
    track_id = search_for_track_id(token, song_name, artist_name)
    print(f"Track ID for {song_name}: {track_id}")
    if not track_id:
        return []
    recommendations = get_recommendations_by_track_id(token, track_id)
    songs = []
    for track in recommendations['tracks']:
        track_name = track['name']
        track_artist = track['artists'][0]['name']
        if (track_name, track_artist) not in recently_played:
            songs.append((track_name, track_artist))
            recently_played.add((track_name, track_artist))
    print(f"Recommendations: {songs}")
    return songs

# Example Usage
client_id = 'YOUR_SPOTIFY_CLIENT_ID'
client_secret = 'YOUR_SPOTIFY_CLIENT_SECRET'
user_id = 'SPOTIFY_USER_ID'

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

    # Search bar
    search_bar = QLineEdit(widget)
    search_bar.setPlaceholderText('Search for songs...')
    layout.addWidget(search_bar)

    # Search button
    search_button = QPushButton("Search", widget)
    layout.addWidget(search_button)

    stacked_layout = QStackedLayout()

    # Music list
    song_list_widget = QListWidget()
    layout.addWidget(song_list_widget)

   # Loading gif
    loading_gif = QMovie("./assets/loading.gif")
    loading_gif_label = QLabel(song_list_widget)
    loading_gif_label.setMovie(loading_gif)
    loading_gif.setScaledSize(QSize(350, 280))
    loading_gif_label.setAlignment(Qt.AlignCenter)
    loading_gif_label.setAttribute(Qt.WA_TranslucentBackground)
    loading_gif_label.hide()

    # playback info layout
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
    
    def get_channel_name_for_video(video_id):
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': True,
        }
    
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            return info_dict.get('uploader', None)
    
    def search_song():
        query = search_bar.text()
        song_list_widget.clear()
        loading_gif_label.resize(song_list_widget.size()) 
        loading_gif_label.show()
        loading_gif.start()

        final_results = []

        def add_song_to_list(title, icon=None):
            item = QListWidgetItem(title)
            if icon:
                item.setIcon(icon)
            song_list_widget.addItem(item)

        song_signal = SongFoundSignal()
        song_signal.song_found.connect(add_song_to_list)

        def is_song_similar(song1, song2):
            threshold_title = 80
            threshold_artist = 90 
            
            title1, artist1 = song1[0].split('\n')
            title2, artist2 = song2[0].split('\n')

            title_similarity = max(fuzz.partial_ratio(title1, title2), fuzz.token_sort_ratio(title1, title2))
            artist_similarity = max(fuzz.partial_ratio(artist1, artist2), fuzz.token_sort_ratio(artist1, artist2))

            return title_similarity > threshold_title and artist_similarity > threshold_artist

        def add_song_to_final_list(song, final_list):
            for existing_song in final_list:
                if is_song_similar(song, existing_song):
                   return 
            final_list.append(song)
            song_signal.song_found.emit(song[0], song[1])


         
        def youtube_search_worker(dummy_signal_instance): 
            youtube_results = search_youtube(query, max_results=5)
            for title, video_id in youtube_results:
                channel_name = get_channel_name_for_video(video_id)
                full_title = f"{title}\n{channel_name}"
                if full_title not in song_database and title != channel_name:
                    song_database[full_title] = video_id
                    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                    thumbnail = get_thumbnail_as_pixmap(thumbnail_url)
                    icon = QIcon(thumbnail)
                    add_song_to_final_list((full_title, icon), final_results)

        def spotify_search_worker(dummy_signal_instance):
            spotify_results = search_spotify(query)
            print(spotify_results)
            for title, artist_name, spotify_thumbnail_url in spotify_results:
                full_title = f"{title}\n{artist_name}"
                if full_title not in song_database:
                    video_id = get_youtube_video_id(f"{title} {artist_name}")
                    if video_id:
                        song_database[full_title] = video_id
                        thumbnail = get_thumbnail_as_pixmap(spotify_thumbnail_url)
                        icon = QIcon(thumbnail)
                        add_song_to_final_list((full_title, icon), final_results)

        def deezer_search_worker(dummy_signal_instance):                    
            deezer_results = search_deezer_rapidapi(query)
            for title, artist_name, album_cover in deezer_results:
                full_title = f"{title}\n{artist_name}"
                if full_title not in song_database:
                    video_id = get_youtube_video_id(f"{title} {artist_name}")
                    if video_id:
                        song_database[full_title] = video_id
                        thumbnail = get_thumbnail_as_pixmap(album_cover)
                        icon = QIcon(thumbnail)
                        add_song_to_final_list((full_title, icon), final_results)

        completed_threads_counter = 0 

        def worker_finished():
            nonlocal completed_threads_counter
            completed_threads_counter += 1
            if completed_threads_counter == len(search_threads):
                loading_gif.stop()
                loading_gif_label.hide()   

        class SearchThread(QThread):
            finished_signal = Signal()

            def __init__(self, func):
                super().__init__()
                self.func = func

            def run(self):
                self.func(None) 
                self.finished_signal.emit()                    
                    
        search_threads = [
            SearchThread(youtube_search_worker),
            SearchThread(spotify_search_worker),
            SearchThread(deezer_search_worker),
        ]

        for thread in search_threads:
            thread.finished_signal.connect(worker_finished)
            thread.start()   

    instance = vlc.Instance()
    player = instance.media_player_new() 
    player.audio_set_volume(60) 
    is_playing = False

    progress_bar = CustomSlider(player, Qt.Horizontal)

    music_bar_layout = QVBoxLayout()

    # current time label
    current_time_label = QLabel("00:00")

    # total duration label
    total_duration_label = QLabel("00:00")

    # controls
    controls_layout = QHBoxLayout()
    play_pause_button = QPushButton("▶️")
    next_song_button = QPushButton("⏭️")
    prev_song_button = QPushButton("⏮️")
    loop_control = QComboBox()
    loop_control.addItems(["No Loop", "Repeat One", "Repeat All"])

    container_widget = QWidget()
    container_widget.setFixedWidth(450)

    # a layout for container
    container_layout = QHBoxLayout()
    container_layout.setSpacing(10) 
    
    container_widget.setLayout(container_layout)
    container_layout.setContentsMargins(0, 0, 0, 0) 

    thumbnail_label_playback.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

    container_layout.addWidget(thumbnail_label_playback)
    container_layout.addWidget(title_label_playback)

    # Playback controls
    playback_controls_layout = QHBoxLayout()
    playback_controls_layout.addWidget(container_widget) 
    playback_controls_layout.setContentsMargins(0, 0, 0, 0)
    playback_controls_layout.setSpacing(0)

    playback_controls_layout.addStretch(1.5) 
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

    music_bar_widget = QWidget()
    music_bar_widget.setLayout(music_bar_layout)

    layout.addWidget(music_bar_widget)
    progress_timer = QTimer()

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

                song_duration = info_dict.get('duration', 0) 
                progress_bar.setMaximum(song_duration * 1000) 
                total_duration_label.setText(format_time(song_duration * 1000))
                progress_timer.timeout.connect(lambda: progress_bar.setValue(player.get_time()))
                
                thumbnail_url = f"http://i4.ytimg.com/vi/{video_id}/default.jpg"
                thumbnail = get_thumbnail_as_pixmap(thumbnail_url)
                thumbnail_label_playback.setPixmap(thumbnail.scaled(50, 50, Qt.KeepAspectRatio)) 
                title_label_playback.setText(info_dict.get('title', 'Unknown Title'))
                dominant_color = dominant_color_from_url(thumbnail_url)
                music_bar_widget.setStyleSheet(f"background-color: rgb({dominant_color.red()}, {dominant_color.green()}, {dominant_color.blue()});")

                pythoncom.CoInitialize()

                media = instance.media_new(audio_url)
                player.set_media(media)
                player.play()

                is_playing = True
                progress_timer.start(1000) 
                
                global recommended_songs_queue
                recommended_songs_queue.clear()
                recommended_songs_queue = fetch_recommendations(info_dict.get('title'), info_dict.get('uploader'))
                print(recommended_songs_queue)
                
        except Exception as e:
            print(f"Error playing song: {e}")

    def play_next_song():
        global recommended_songs_queue
        if recommended_songs_queue:
            next_song, next_artist = recommended_songs_queue.pop(0)
            print(f"Playing next recommended song: {next_song} by {next_artist}")
        elif len(recommended_songs_queue) <= 2:
            recommended_songs_queue.extend(fetch_recommendations(next_song, next_artist))

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
        global recommended_songs_queue
        if recommended_songs_queue:
            next_recommended_song = recommended_songs_queue.pop(0) 
            video_id = song_database.get(next_recommended_song.split('\n')[0])  # video_id from song title
            if video_id:
                selected_song = next_recommended_song

            else:
                print(f"No video ID found for song: {next_recommended_song}")
        else:

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