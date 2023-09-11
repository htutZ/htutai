import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from ytmusicapi import YTMusic
import pytube
from pytube import Search
from colorthief import ColorThief
from io import BytesIO
import requests
from PySide2.QtGui import QImage, QPixmap, QIcon, QPalette, QColor, QPainter
ytmusic = YTMusic('headers_auth.json')

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
    results = sp.search(q=query, limit=23)
    return [(track['name'], track['album']['images'][0]['url']) for track in results['tracks']['items']]

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

def get_youtube_video_id(song_name):
    search_results = search_youtube(song_name, max_results=1)
    if search_results:
        return search_results[0][1]
    return None

def format_time(ms):
    """Convert milliseconds into MM:SS format."""
    s = ms // 1000
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"
