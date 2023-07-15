import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Replace with your Spotify client ID and client secret
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Read the CSV file
df = pd.read_csv(r'C:\Users\kiett\Downloads\Project-main\Project-main\Spotify_Crawl\spotify_data.csv')


# Initialize empty lists
tempo = []
speechiness = []
loudness = []
liveness = []
instrumentalness = []
energy = []
acousticness = []
valence = []

# Loop over the rows of the DataFrame
for i, row in df.iterrows():
    artist = row['artist_name']
    track = row['track_name']
    print(track)

    # Search for the track using artist and track name
    results = sp.search(q='artist:' + artist + ' track:' + track, type='track')
    
    # Check if search results found any tracks
    if len(results['tracks']['items']) > 0:
        track_id = results['tracks']['items'][0]['id']
        
        try:
            # Get audio features for the track
            audio_features = sp.audio_features(track_id)[0]
            tempo.append(audio_features['tempo'])
            speechiness.append(audio_features['speechiness'])
            loudness.append(audio_features['loudness'])
            liveness.append(audio_features['liveness'])
            instrumentalness.append(audio_features['instrumentalness'])
            energy.append(audio_features['energy'])
            acousticness.append(audio_features['acousticness'])
            valence.append(audio_features['valence'])
            print(i)
        except:
            print(i, 'Error retrieving audio features')
    else:
        print(i, 'Track not found')

# Assign the new columns to the DataFrame
df['tempo'] = tempo
df['speechiness'] = speechiness
df['loudness'] = loudness
df['liveness'] = liveness
df['instrumentalness'] = instrumentalness
df['energy'] = energy
df['acousticness'] = acousticness
df['valence'] = valence

# Save the updated DataFrame to a new CSV file
df.to_csv('final_1.csv', index=False)
print(df.head(5))