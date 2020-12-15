# Module: prepare.py
# Functions to prep data acquired from the Spotify API

import pandas as pd
import numpy as np
from preprocessing import create_features

###################################################### Handle Nulls ######################################################

def handle_nulls(df):
    '''
    This function takes in a DataFrame and returns a DataFrame with the nulls handled.
    Release dates that don't have a month or day are assign a day and month of '01-01'.
    '''
    # handle nulls in release data
    df['release_date'] = np.where(df['release_date'].str.len()==4, df.release_date.astype(str) + '-01-01', df['release_date'])

    # drop observations that contain nulls
    df = df.dropna()

    return df

def set_index(df):
    df = df.set_index('track_id')
    return df

def change_dtypes(df):
    # change explicit column to int
    df['explicit'] = df.explicit.astype('int')
    df['is_featured_artist'] = df.is_featured_artist.astype('int')
    df['disc_number'] = df.disc_number.astype('int')
    df['mode'] = df['mode'].astype('int')
    df['key'] = df.key.astype('int')
    df['duration_seconds'] = df.duration_seconds.astype('int')
    df['duration_minutes'] = df.duration_minutes.astype('int')
    df['duration_ms'] = df.duration_ms.astype('int')
    df['popularity'] = df.popularity.astype('int')
    df['time_signature'] = df.time_signature.astype('int')
    df['track_number'] = df.track_number.astype('int')

    return df

def prepare_df(df):
    df = create_features(df)
    df = handle_nulls(df)
    df = change_dtypes(df)
    df = set_index(df)
    return df