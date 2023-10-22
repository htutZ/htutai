import requests

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