import pandas as pd
import spotipy
import numpy as np
import os

from spotipy.oauth2 import SpotifyClientCredentials 
from env import cid, c_secret

###################################################### Create Spotipy Client ######################################################

# Function to create spotipy client object
def create_spotipy_client():
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, 
    client_secret=c_secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
    sp.trace=False
    return sp

###################################################### Analyze First 100 Tracks From Playlist/Offset In Playlist ######################################################

# Function to acquire playlist tracks and features
def analyze_playlist(creator, playlist_id, sp_client, offset=0):
    
    # Create empty dataframe
    playlist_features_list = ["artist","album","release_date","track_name","track_id", 'label',
                              "danceability","energy","key","loudness","mode", "speechiness","instrumentalness",
                              "liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist df
    playlist = sp_client.user_playlist_tracks(creator, playlist_id, offset=offset)['items']
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
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
            playlist_features['track_number'] = track['track']['track_number']
            playlist_features['album_id'] = track['track']['album']['id']
            playlist_features['album_type'] = track['track']['album']['album_type']
        
            # Get audio features
            audio_features = sp_client.audio_features(playlist_features["track_id"])
            if audio_features is None:
                for feature in playlist_features_list[6:]:
                    playlist_features[feature] = None
            elif audio_features[0] is None:
                    for feature in playlist_features_list[6:]:
                        playlist_features[feature] = None
            else:
                for feature in playlist_features_list[6:]:
                    playlist_features[feature] = audio_features[0][feature]
            
            # Get album popularity
            album_features = sp_client.album(playlist_features['album_id'])
            if album_features is None:
                for feature in playlist_features_list[5:6]:
                    playlist_features[feature] = None
            else:
                playlist_features['album_popularity'] = album_features['popularity']
                playlist_features['label'] = album_features['label']

            # Concat the dfs
            track_df = pd.DataFrame(playlist_features, index = [0])
            playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df

###################################################### Concat CSV Files ######################################################

def concat_csv_files():

    '''
    Loops through each csv file of acquired data to combine into one df.
    No parameters needed, only needs the files saved in the working directory.
    Returns the one df.
    '''

    # sets initial df as file of first 100 observations
    df = pd.read_csv('data/playlist-offset-0.csv', index_col=0)

    # loops through 100 - 6000 by one hundreds, matching the csv file names
    # as it loops, it combines the csv file to the original df
    for offset in range(100, 5901, 100):

        # saves next csv file as a df
        add_df = pd.read_csv(f'data/playlist-offset-{offset}.csv', index_col=0)

        # adds the new df to the original df 
        df = pd.concat([df, add_df], ignore_index=True)

    # returns the csv files combined in one dataframe, should be 6_000 observations
    return df

###################################################### Gather Entire Capstone Playlist ######################################################

# sp is the spotipy client you created
def get_capstone_playlist(sp):
    if os.path.exists('data/full-playlist.csv'):
        df = pd.read_csv('data/full-playlist.csv', index_col=0)
    else:
    # Let this loop run as it gathers the tracks from the playlist
        for offset in range(0, 6000, 100):
            # Prints out how many pages in the loop is. Each page is 100 tracks + or - a few if nulls appear
            print(f'Making page with offset = {offset}')
            # Analyze the first 100 tracks past the offset
            playlist_df = analyze_playlist('spotify:user:afrodeezeemusic', '3P6Pr6iEqvK5fl4UkgdQ7T', sp, offset)
            # Write each dataframe of 100 tracks to a csv. If the function ends early in an error you will still have some data
            playlist_df.to_csv('data/playlist-offset-' + str(offset) + '.csv')
        # use the concat_csv_files function to concat all the dataframes together into one complete dataframe    
        df = concat_csv_files()
        df.to_csv('data/full-playlist.csv')
    return df