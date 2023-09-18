import requests

def search_musicapi(track: str, artist: str, sources=["spotify", "appleMusic", "youtube", "deezer"]):
    url = "https://musicapi13.p.rapidapi.com/public/search"

    payload = {
        "track": track,
        "artist": artist,
        "type": "track",
        "sources": sources
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "2d4e76f6dfmshe4465090fbe5c4ap159ae1jsn957f235ebcc4",
        "X-RapidAPI-Host": "musicapi13.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)

    data = response.json()

    if not data.get('tracks'):
        return []

    results = []
    for track in data['tracks']:  
        title = track['data']['name']
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

# Sample usage
if __name__ == "__main__":
    results = search_musicapi("Bezos I", "Bo Burnham")
    for title, artists, album, cover, url in results:
        print(f"Title: {title}, Artists: {artists}, Album: {album}, Cover: {cover}, URL: {url}")
