import requests

# Sample configuration (replace with actual values)
SPOTIFY_API_KEYS = ["SPOTIFY_API_KEY_1", "SPOTIFY_API_KEY_2"]
DEEZER_API_KEY = "YOUR_DEEZER_API_KEY"
SOUNDCLOUD_API_KEYS = ["SOUNDCLOUD_API_KEY_1", "SOUNDCLOUD_API_KEY_2"]

API_CONFIGS = {
    "Spotify": {
        "endpoints": ["SPOTIFY_ENDPOINT_1", "SPOTIFY_ENDPOINT_2"],  # Multiple endpoints for different keys
        "keys": SPOTIFY_API_KEYS,
        "rate_limits": [1000, 1000]  # Sample rate limits for each key
    },
    "Deezer": {
        "endpoints": ["DEEZER_ENDPOINT"],
        "keys": [DEEZER_API_KEY],
        "rate_limits": [500]
    },
    "SoundCloud": {
        "endpoints": ["SOUNDCLOUD_ENDPOINT_1", "SOUNDCLOUD_ENDPOINT_2"],
        "keys": SOUNDCLOUD_API_KEYS,
        "rate_limits": [100, 100]
    }
}

def make_api_request(endpoint, key, query):
    headers = {
        'Authorization': f"Bearer {key}"
    }
    response = requests.get(endpoint, headers=headers, params={"q": query})
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

def get_music_data(query):
    for api_name, api_config in API_CONFIGS.items():
        for i, endpoint in enumerate(api_config["endpoints"]):
            if api_config['rate_limits'][i] > 0:
                try:
                    data = make_api_request(endpoint, api_config["keys"][i], query)
                    api_config['rate_limits'][i] -= 1  # Decrease the rate limit
                    return data
                except requests.HTTPError as e:
                    print(f"Error with {api_name} API: {e}")
                    continue  # if an error occurs, continue to the next API or endpoint
    return None  # if all APIs fail or rate limits are exceeded

# Usage
song_data = get_music_data("Imagine by John Lennon")
