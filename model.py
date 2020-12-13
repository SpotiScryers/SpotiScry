import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestRegressor

def get_model_features(df):
    '''
    This function takes in a DataFrame and returns a DataFrame with features to use in predictive modeling.
    '''
    df = df.drop(columns=['artist', 'album', 'release_date', 'track_name'])
    return df

def OLS_model(X, y, X_v, y_v):
    '''
    This function creates, fits, and evaluates an OLS using linear regression model.
    '''
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data.
    lm.fit(X, y)

    # predict on train
    lm_pred = lm.predict(X)
    # compute root mean squared error
    lm_rmse = mean_squared_error(y, lm_pred)**1/2

    # predict on validate
    lm_pred_v = lm.predict(X_v)
    # compute root mean squared error
    lm_rmse_v = mean_squared_error(y_v, lm_pred_v)**1/2

    print("RMSE for OLS using Linear Regression\n\nOn train data:\n", round(lm_rmse, 6), '\n\n', 
        "On validate data:\n", round(lm_rmse_v, 6))

    return lm_pred, lm_rmse, lm_pred_v, lm_rmse_v