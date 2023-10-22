import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from ytmusicapi import YTMusic
import pytube
from pytube import Search
import yt_dlp
from colorthief import ColorThief
from io import BytesIO
import requests
from PySide2.QtGui import QImage, QPixmap, QIcon, QPalette, QColor, QPainter
ytmusic = YTMusic('headers_auth.json')
import deezer

SPOTIFY_CREDENTIALS = [
    {'client_id': '92b308edd46a4134894df230bf543670', 'client_secret': '9ef2001ccf21447f937679801d935b3d'},
    {'client_id': 'a04abef1ba5d4fc5a031fce0f094e5f8', 'client_secret': 'ae13a92da3f7438490c277acc29e9f84'},
    # ... add more credentials as needed
]

RAPIDAPI_KEYS = [
    '2d4e76f6dfmshe4465090fbe5c4ap159ae1jsn957f235ebcc4',  # replace with your actual key
    'key2',  # replace with your actual key
    # ... add more keys as needed
]

CURRENT_KEY_INDEX = 0
RAPIDAPI_DEEZER_HOST = "deezerdevs-deezer.p.rapidapi.com"

def rotate_rapidapi_key():
    """Rotate to the next available RapidAPI key."""
    global CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(RAPIDAPI_KEYS)

def get_current_rapidapi_key():
    """Get the current RapidAPI key."""
    return RAPIDAPI_KEYS[CURRENT_KEY_INDEX]

LASTFM_API_KEY = '27125370b7dd847e69b366dd3dc97993'

DEEZER_API_ENDPOINT = "https://api.deezer.com/search"

SPOTIFY_AUTH_URL = 'https://accounts.spotify.com/api/token'
SPOTIFY_BASE_URL = 'https://api.spotify.com/v1'

current_spotify_credential_index = 0\

def rotate_spotify_credentials():
    """Rotate to the next available Spotify credentials."""
    global current_spotify_credential_index
    current_spotify_credential_index = (current_spotify_credential_index + 1) % len(SPOTIFY_CREDENTIALS)

def spotify_authenticate():
    global current_spotify_credential_index
    retry_count = len(SPOTIFY_CREDENTIALS)
    
    while retry_count > 0:
        credentials = SPOTIFY_CREDENTIALS[current_spotify_credential_index]
        try:
            auth_response = requests.post(SPOTIFY_AUTH_URL, {
                'grant_type': 'client_credentials',
                'client_id': credentials['client_id'],
                'client_secret': credentials['client_secret'],
            })
            
            auth_response_data = auth_response.json()
            if auth_response.status_code == 200:
                access_token = auth_response_data['access_token']
                headers = {
                    'Authorization': 'Bearer {token}'.format(token=access_token)
                }
                return headers
            elif auth_response.status_code == 429:  # Rate limiting
                print("Rate limited. Retrying...")
                rotate_spotify_credentials()
                retry_count -= 1
            elif auth_response.status_code == 401:  # Unauthorized
                print("Credentials unauthorized. Trying next...")
                rotate_spotify_credentials()
                retry_count -= 1
            else:
                print(f"Error: {auth_response.status_code} - {auth_response_data.get('error_description')}")
                rotate_spotify_credentials()
                retry_count -= 1
        except requests.RequestException as e:
            print(f"Request error: {e}")
            retry_count -= 1

    print("All credentials exhausted.")
    return None

def initialize_spotify(credentials):
    client_credentials_manager = SpotifyClientCredentials(client_id=credentials['client_id'],
                                                          client_secret=credentials['client_secret'])
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def search_spotify_for_track_id(track_name, artist_name):
    headers = spotify_authenticate()
    query = f"track:{track_name} artist:{artist_name}"
    search_url = SPOTIFY_BASE_URL + f"/search?q={query}&type=track&limit=1"
    response = requests.get(search_url, headers=headers)
    response_data = response.json()
    tracks = response_data['tracks']['items']
    if tracks:
        return tracks[0]['id']
    return None

def get_spotify_recommendations(sp, track_id):
    try:
        recommendations = sp.recommendations(seed_tracks=[track_id], limit=10)
        return recommendations['tracks']
    except Exception as e:
        print(f"Error getting Spotify recommendations: {e}")
        return []

def dominant_color_from_url(url):
    """Get the dominant color from an image URL."""
    response = requests.get(url)
    color_thief = ColorThief(BytesIO(response.content))
    dominant_color = color_thief.get_color(quality=1)
    return QColor(*dominant_color)
    
def search_youtube(query, max_results=5):
    search_results = Search(query).results[:max_results]
    return [(video.title, video.video_id) for video in search_results]

def get_spotify_token():
    global current_spotify_credential_index
    credentials = SPOTIFY_CREDENTIALS[current_spotify_credential_index]

    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': credentials['client_id'],
        'client_secret': credentials['client_secret']
    })

    auth_response_data = auth_response.json()
    return auth_response_data['access_token']

def search_for_track_id(token, track_name, artist_name=None):
    track_name = track_name.split('(')[0].strip()
    
    # First, try searching with both the song title and artist name
    query = f"{track_name} artist:{artist_name}" if artist_name else track_name
    track_id = perform_spotify_search(token, query)
    
    # If no results, try searching with the song title only
    if track_id is None and artist_name:
        query = track_name
        track_id = perform_spotify_search(token, query)
    
    return track_id

def perform_spotify_search(token, query):
    search_url = 'https://api.spotify.com/v1/search'
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
    if not results['tracks']['items']:
        print(f"No results for query: {query}")
        print(f"Full API response: {results}")
        return None
    return results['tracks']['items'][0]['id']


def get_recommendations_by_track_id(token, track_id):
    token = get_spotify_token()
    recommendations_url = 'https://api.spotify.com/v1/recommendations'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    params = {
        'seed_tracks': track_id
    }
    response = requests.get(recommendations_url, headers=headers, params=params)
    return response.json()


def search_spotify(query, limit=15):
    global current_spotify_credential_index
    results = None
    retry_count = len(SPOTIFY_CREDENTIALS)

    while retry_count > 0:
        sp = initialize_spotify(SPOTIFY_CREDENTIALS[current_spotify_credential_index])
        try:
            response = sp.search(q=query, limit=limit)
            results = [(track['name'], track['artists'][0]['name'], track['album']['images'][0]['url']) for track in response['tracks']['items']]
            break
        except spotipy.exceptions.SpotifyException:
            # If rate limited or any other Spotify exception, switch credentials
            current_spotify_credential_index = (current_spotify_credential_index + 1) % len(SPOTIFY_CREDENTIALS)
            retry_count -= 1

    return results

# def search_soundcloud_rapidapi(query, limit=10):
#     print("search_soundcloud_rapidapi called")

#     url = "https://soundcloud4.p.rapidapi.com/search"

#     headers = {
#         'x-rapidapi-host': "soundcloud4.p.rapidapi.com",
#         'x-rapidapi-key': RAPIDAPI_KEY
#     }

#     params = {
#         'query': query,
#         'type': 'all',
#         'limit': str(limit)
#     }

#     response = requests.get(url, headers=headers, params=params)
#     print(f"Response status code: {response.status_code}")

#     data = response.json()
#     print(data)

#     results = []
#     for track in data:
#         if 'type' in track and track['type'] == 'track':
#             title = f"{track.get('title', 'Unknown Title')} - {track['author'].get('name', 'Unknown Artist')}"
#             thumbnail_url = track.get('thumbnail', None)
#             track_id = track.get('id', None)
#             results.append((title, thumbnail_url, track_id))

#     return results

def search_musicapi(track: str, artist: str, sources=["appleMusic", "youtubeMusic", "tidal"]):
    url = "https://musicapi13.p.rapidapi.com/public/search"
    retry_count = len(RAPIDAPI_KEYS)

    while retry_count > 0:
        payload = {
            "track": track,
            "artist": artist,
            "type": "track",
            "sources": sources
        }
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": get_current_rapidapi_key(),
            "X-RapidAPI-Host": "musicapi13.p.rapidapi.com"
        }

        response = requests.post(url, json=payload, headers=headers)
        
        # If "Too Many Requests" which indicates rate limit, rotate the key and retry
        if response.status_code == 429:
            print("Rate limit reached for current RapidAPI key. Rotating to next key...")
            rotate_rapidapi_key()
            retry_count -= 1
            continue
        
        elif response.status_code == 200:
            data = response.json()
            if not data.get('tracks'):
                return []
            results = []
            for track in data['tracks']:  
                if track and 'data' in track and track['data']:
                   title = track['data'].get('name', None)
                else:
                   title = None
                artist_names = track['data'].get('artistNames')
                if artist_names and isinstance(artist_names, (list, tuple)):
                    artist_names = ', '.join(artist_names)
                else:
                    artist_names = ''
                album_name = track['data']['albumName']
                image_url = track['data']['imageUrl']
                track_url = track['data']['url']
                results.append((title, artist_names, album_name, image_url, track_url))
            return results
        
        elif response.status_code == 500:
            print("Internal server error from MusicAPI.")
            return []

    print(f"All RapidAPI keys exhausted. Please check rate limits or add more keys.")
    return []

def search_deezer_rapidapi(query, limit=10):
    url = f"https://{RAPIDAPI_DEEZER_HOST}/search"
    headers = {
        "X-RapidAPI-Key": get_current_rapidapi_key(),
        "X-RapidAPI-Host": RAPIDAPI_DEEZER_HOST
    }
    params = {
        "q": query
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 429:  # HTTP 429 is "Too Many Requests" which indicates rate limit
        rotate_rapidapi_key()  # Switch to the next key
        return search_deezer_rapidapi(query, limit)  # Retry with the new key
    
    elif response.status_code != 200:
        print(f"Failed to fetch data from Deezer. Status code: {response.status_code}")
        return []

    data = response.json()
    results = []
    for track in data['data'][:limit]:  # We limit the results to the specified limit
        title = track['title']
        artist = track['artist']['name']
        album_cover = track['album']['cover_medium']
        results.append((title, artist, album_cover))
    
    return results

def search_deezer(query):
    client = deezer.Client()  # Initialize the Deezer client
    results = []

    # Fetch search results
    search_results = client.search(query)
    if search_results:
        print(search_results[0].title)
        print(search_results[0].artist.name)
        print(search_results[0].album.cover_medium)
        print(f"Number of results from Deezer for '{query}': {len(search_results)}")
        
        for track in search_results[:10]:  # Limit to the first 10 results
            title = track.title
            artist = track.artist.name
            album_cover = track.album.cover_medium
            print(f"Processing - Title: {title}, Artist: {artist}, Album Cover: {album_cover}")  # Debug print
            
            if album_cover:  # Only add tracks with album covers
                results.append((title, artist, album_cover))
            print(f"Results so far: {results}") 
    else:
        print("search_results is empty!")

    return results

def search_lastfm(query, artist=None, limit=15):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.search",
        "track": query,
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": limit
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    results = []
    for track in data['results']['trackmatches']['track']:
        title = track['name']
        artist_name = track['artist']
        # Let's get the best available thumbnail
        image_url = None
        for image_data in reversed(track['image']):
            if image_data['#text']:
                image_url = image_data['#text']
                break
        if image_url:  # Only add tracks with thumbnails
            results.append((title, artist_name, image_url))
    
    return results


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