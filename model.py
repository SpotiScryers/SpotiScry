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
    lm_rmse = sqrt(mean_squared_error(y, lm_pred))

    # predict on validate
    lm_pred_v = lm.predict(X_v)
    # compute root mean squared error
    lm_rmse_v = sqrt(mean_squared_error(y_v, lm_pred_v))

    print("RMSE for OLS using Linear Regression\n\nOn train data:\n", round(lm_rmse, 6), '\n\n', 
        "On validate data:\n", round(lm_rmse_v, 6))

    return lm_pred, lm_rmse, lm_pred_v, lm_rmse_v


def get_baseline_metrics(y_tr):
    bl = np.mean(y_tr)
    
    bl_train_rmse = round(sqrt(mean_squared_error(y_tr, np.full(len(y_tr), baseline))), 6)
    print('RMSE (Root Mean Square Error) of Baseline on train data:\n', baseline_train_rmse)

    #bl_validate_rmse = round(sqrt(mean_squared_error(y_v, np.full(len(y_v), baseline))), 6)
    #print('RMSE (Root Mean Square Error) of Baseline on validate data:\n', baseline_validate_rmse)

    #bl_test_rmse = round(sqrt(mean_squared_error(y_te, np.full(len(y_te), baseline))), 6)
    #print('RMSE (Root Mean Square Error) of Baseline on test data:\n', baseline_test_rmse)

    return bl, bl_train_rmse

def linear_regression_model(X_tr, y_tr, X_v, y_v, normalize=True, fit_intercept=True):
    lm = LinearRegression(normalize=True)
    lm.fit(X_tr, y_tr)
    lm_pred = lm.predict(X_tr)
    lm_rmse = sqrt(mean_squared_error(y_tr, lm_pred))

    lm_pred_v = lm.predict(X_v)
    lm_rmse_v = sqrt(mean_squared_error(y_v, lm_pred_v))

    print('RMSE for OLS using Linear Regression \n')
    print('On train data:\n', round(lm_rmse, 6), '\n')
    return lm_rmse, lm_rmse_v

def lasso_lars(X_tr, y_tr, X_v, y_v, X_te, y_te, alpha=.00001, fit_intercept=True):
    lars = LassoLars(alpha=.00001, fit_intercept=True)
    lars.fit(X_tr, y_tr)

    lars_pred = lars.predict(X_tr)
    lars_rmse = sqrt(mean_squared_error(y_tr, lars_pred))

    lars_pred_v = lars.predict(X_v)
    lars_rmse_v = sqrt(mean_squared_error(y_v, lars_pred_v))

    lars_pred_t = lars.predict(X_te)
    lars_rmse_t = sqrt(mean_squared_error(y_te, lars_pred_t))

    print('RMSE for LASSO + LARS \n')
    print('On train data:\n', round(lars_rmse, 6), '\n') 
    return  lars_rmse, lars_rmse_v, lars_rmse_t

def polynomial_regression(X_tr, y_tr, X_v, y_v, degree=2):
    pf = PolynomialFeatures(degree=2)
    X_train_sq = pf.fit_transform(X_tr)
    X_validate_sq = pf.transform(X_v)

    lm_sq = LinearRegression()
    lm_sq.fit(X_train_sq, y_tr)

    lm_sq_pred = lm_sq.predict(X_train_sq)
    lm_sq_rmse = sqrt(mean_squared_error(y_tr, lm_sq_pred))

    lm_sq_pred_v = lm_sq.predict(X_validate_sq)
    lm_sq_rmse_v = sqrt(mean_squared_error(y_v, lm_sq_pred_v))

    print('RMSE for Polynomial Squared + Linear Regression \n')
    print('On train data:\n', round(lm_sq_rmse, 6), '\n')
    return lm_sq_rmse, lm_sq_rmse_v

def evaluate_df(bl_train_rmse, lm_rmse, lars_rmse_v, lars_rmse_t, lm_sq_rmse, lm_sq_rmse_v, lm_cb_rmse, lm_cb_rmse_v):
    columns = ['train_rmse', 'validate_rmse', 'test_rmse']
    index = ['baseline', 'ols', 'lassolars', 'pf2_lr', 'pf3_lr']
    data = [[bl_train_rmse, '-', '-'],
            [lm_rmse, '-', '-'],
            [lars_rmse, lars_rmse_v, lars_rmse_t],
            [lm_sq_rmse, lm_sq_rmse_v, '-'],
            [lm_cb_rmse, lm_cb_rmse_v, '-']]
    print(f'Model beat baseline by {abs((lars_rmse_t - bl_train_rmse)/bl_train_rmse)*100:.2f}%')
    return pd.DataFrame(columns=columns, data=data, index=index).sort_values(by='train_rmse')

