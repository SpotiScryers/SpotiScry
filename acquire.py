import pandas as pd
import spotipy
import numpy as np

from spotipy.oauth2 import SpotifyClientCredentials 
from env import cid, c_secret

# Function to create spotipy client object
def create_spotipy_client():
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, 
    client_secret=c_secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
    sp.trace=False
    return sp

# Function to acquire playlist tracks and features
def analyze_playlist(creator, playlist_id, sp_client, offset=0):
    
    # Create empty dataframe
    playlist_features_list = ["artist","album","release_date","track_name","track_id","danceability","energy","key",
                              "loudness","mode", "speechiness","instrumentalness","liveness","valence",
                              "tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist df
    playlist = sp_client.user_playlist_tracks(creator, playlist_id, offset=offset)['items']
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata, skipping if track is missing artist data
        if track['track']['album']['artists'] == []:
            continue
        else:
            playlist_features['artist'] = track['track']['album']['artists'][0]['name']
            playlist_features["album"] = track["track"]["album"]["name"]
            playlist_features["release_date"] = track["track"]["album"]["release_date"]
            playlist_features["track_name"] = track["track"]["name"]
            playlist_features["track_id"] = track["track"]["id"]
            playlist_features['explicit'] = track['track']['explicit']
            playlist_features["popularity"] = track["track"]["popularity"]
            playlist_features['disc_number'] = track['track']['disc_number']
        
            # Get audio features
            audio_features = sp_client.audio_features(playlist_features["track_id"])
            if audio_features is None:
                for feature in playlist_features_list[5:]:
                    playlist_features[feature] = None
            elif audio_features[0] is None:
                    for feature in playlist_features_list[5:]:
                        playlist_features[feature] = None
            else:
                for feature in playlist_features_list[5:]:
                    playlist_features[feature] = audio_features[0][feature]

            # Concat the dfs
            track_df = pd.DataFrame(playlist_features, index = [0])
            playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df