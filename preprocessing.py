# Module: preprocessing.py
# Functions to reproduce the pre-processing of the data

import pandas as pd

################################################## Feature Engineering ###################################################

def create_features(df):

    '''
    Creates features based on the original dataframe. Converts track duration into new columns for length in seconds and
    length in minutes. Creates a feature for a boolean value if the track features an artist or not. Lower cases all 
    characters in text data columns. Returns the df with these features created at the end and string conversions.
    '''

    # Feature Engineering
    # converting track length from ms to seconds as new column
    df['duration_seconds'] = df.duration_ms / 1_000
    # converting track length from seconds to minutes as new column
    df['duration_minutes'] = df.duration_seconds / 60
    # creating boolean if track has a featured artist
    df['is_featured_artist'] = df.track_name.str.contains('feat')

    # Lowercasing String
    # using string function to convert all characters to lowercase
    df['artist'] = df.artist.str.lower()
    df['album'] = df.album.str.lower()
    df['track_name'] = df.track_name.str.lower()

    return df

##################################################### Split the Data #####################################################

def split_df(df):

    '''
    Splits dataframe into train, validate, and test - 70%, 20%, 10% respectively.
    Prints out the percentage shape and row/column shape of the split dataframes.
    Returns train, validate, test.
    '''

    # Import to use split function, can only split two at a time
    from sklearn.model_selection import train_test_split

    # First, split into train + validate together and test by itself
    # Test will be %10 of the data, train + validate is %70 for now
    # Set random_state so we can reproduce the same 'random' data
    train_validate, test = train_test_split(df, test_size = .10, random_state = 666)

    # Second, split train + validate into their seperate dataframes
    # Train will be %70 of the data, Validate will be %20 of the data
    # Set random_state so we can reproduce the same 'random' data
    train, validate = train_test_split(train_validate, test_size = .22, random_state = 666)

    # These two print functions allow us to ensure the date is properly split
    # Will print the shape of each variable when running the function
    print("train shape: ", train.shape, ", validate shape: ", validate.shape, ", test shape: ", test.shape)

    # Will print the shape of each variable as a percentage of the total data set
    # Variable to hold the sum of all rows (total observations in the data)
    total = df.count()[0]
    
    #calculating percentages of the split df to the original df
    train_percent = round(((train.shape[0])/total),2) * 100
    validate_percent = round(((validate.shape[0])/total),2) * 100
    test_percent = round(((test.shape[0])/total),2) * 100
    
    print("\ntrain percent: ", train_percent, ", validate percent: ", validate_percent, 
            ", test percent: ", test_percent)

    return train, validate, test

##################################################### Scale the Data #####################################################

def scale_data(train, validate, test, predict, scaler):

    '''
    Scales a df based on scaler chosen: 'MinMax', 'Standard', or 'Robust'. 
    Needs three dfs: train, validate, and test. Fits the scaler object to train 
    only, transforms on all 3. Returns the three dfs scaled.
    '''
    
    import sklearn.preprocessing
    
    # removing predictive feature
    X_train = train.drop(predict, axis=1)
    X_validate = validate.drop(predict, axis=1)
    X_test = test.drop(predict, axis=1)
    
    if scaler == 'MinMax':

        # create scaler object for MinMax Scaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        
    elif scaler == 'Standard':
        
        # create scaler object for Standard Scaler
        scaler = sklearn.preprocessing.StandardScaler()
        
    elif scaler == 'Robust':
        
        # create scaler object for Robust Scaler
        scaler = sklearn.preprocessing.StandardScaler()
        
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)

    # transforming all three dfs with the scaler object
    # this turns it into an array
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # converting scaled array back to df
    # first by converting to a df, it will not have the original index and column names
    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_validate_scaled = pd.DataFrame(X_validate_scaled)
    X_test_scaled = pd.DataFrame(X_test_scaled)
        
    # setting index to original dfs
    X_train_scaled.index = X_train.index
    X_validate_scaled.index = X_validate.index
    X_test_scaled.index = X_test.index
        
    # renaming columns to original dfs
    X_train_scaled.columns = X_train.columns
    X_validate_scaled.columns = X_validate.columns
    X_test_scaled.columns = X_test.columns

    return X_train_scaled, X_validate_scaled, X_test_scaled

################################################# Create KMeans Clusters #################################################

def create_clusters(X_train_scaled, X_validate_scaled, X_test_scaled, features, n, cluster_name):
    
    '''
    Create clusters based on features specified. n is amount of groups within the cluster.
    Best when used on scaled dfs. Returns dfs with dummy variables of clusters appended.
    '''
    
    from sklearn.cluster import KMeans

    X = X_train_scaled[features]
    Y = X_validate_scaled[features]
    Z = X_test_scaled[features]
    
    # create object with clusters chosen by n parameter
    kmeans = KMeans(n_clusters=n, random_state = 666)

    # fit to train only and the features chosen
    kmeans.fit(X)
    
    # add a column to the dfs of the prediction of cluster group
    X_train_scaled[cluster_name] = kmeans.predict(X)
    X_validate_scaled[cluster_name] = kmeans.predict(Y)
    X_test_scaled[cluster_name] = kmeans.predict(Z)
    
    
    # naming the cluster groups by cluster name plus numbers 1 through n for each group
    columns = []
    for x in range(1, n+1):
        columns.append(f'{cluster_name}_{x}')
    
    
    # create dataframe of dummy variables of cluster group created for each train, validate, test
    # train cluster dummy variables
    dummies = pd.get_dummies(X_train_scaled[cluster_name])
    dummies.columns = columns

    # validate cluster dummy variables
    dummies2 = pd.get_dummies(X_validate_scaled[cluster_name])
    dummies2.columns = columns

    # test cluster dummy variables
    dummies3 = pd.get_dummies(X_test_scaled[cluster_name])
    dummies3.columns = columns
    
    # add cluster dummy variables to scaled df
    # adding train cluster dummies to train scaled df
    X_train_scaled = pd.concat([X_train_scaled,dummies], axis=1)
    
    # adding validate cluster dummies to validate scaled df
    X_validate_scaled = pd.concat([X_validate_scaled,dummies2], axis=1)
    
    # adding test cluster dummies to test scaled df
    X_test_scaled = pd.concat([X_test_scaled,dummies3], axis=1)

    return X_train_scaled, X_validate_scaled, X_test_scaled