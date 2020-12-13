import pandas as pd
import numpy as np

def get_model_features(df):
    '''
    This function takes in a DataFrame and returns a DataFrame with features to use in predictive modeling.
    '''
    df = df.drop(columns=['artist', 'album', 'release_date', 'track_name', 'track_id'])
    return df