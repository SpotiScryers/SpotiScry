import numpy as np
import pandas as pd 

from sklearn.model_selection import GridSearchCV

def crossval_GSCV(params, model_object, X, y):
    from sklearn.model_selection import GridSearchCV
    '''
    This function takes in a model_object, parameters to 
    run through, X variables and a y target variable.
    The algorithm is using cross validation to find the 
    best hyper parameters for the model. Scoring is based
    on regression evaluation metric RMSE.
    '''
    # set variables for inputs
    parameters = params
    model = model_object
    # create the grid search object
    grid = GridSearchCV(model_object, params, 
                        scoring= 'neg_root_mean_squared_error', 
                        cv=3, iid=True)
    # fit the alogrithm to the data
    grid.fit(X, y)
    # set the list of dictionaries to a variable
    results = grid.cv_results_
    # all the parameter combinations
    params = results['params']
    # the 'test score' - RMSE for each combination
    test_scores = results['mean_test_score']
    # return a dataframe of param combos with score results
    for p, s in zip(params, test_scores):
        p['RMSE'] = s
    # sort by the RMSE
    return pd.DataFrame(params).sort_values(by='RMSE', ascending=False)