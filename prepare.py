# Module: prepare.py
# Functions to prep data acquired from the Spotify API

import pandas as pd
import numpy as np

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
