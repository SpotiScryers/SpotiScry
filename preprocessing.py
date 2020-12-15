# Module: preprocessing.py
# Functions to reproduce the pre-processing of the data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    df['is_featured_artist'] = df.track_name.str.contains('feat').astype('int')

    # Lowercasing String
    # using string function to convert all characters to lowercase
    df['artist'] = df.artist.str.lower()
    df['album'] = df.album.str.lower()
    df['track_name'] = df.track_name.str.lower()

    # Create Seperate Columns for Year, Month, and Day
    # creating dataframe of release day split by '-'
    dates = df.release_date.str.split('-', expand=True)
    # renaming columns to respective year, month, and day
    dates.columns = ['release_year','release_month','release_day']
    # ensuring index is the same as the df
    dates.index = df.index
    # adding to the dataframe with axis=1 to add column-wise
    df = pd.concat([df,dates], axis=1)

    df.release_year = df.release_year.astype('int')

    # bins set edge points for range of label
    # goes from 1980-1989, 1990-1999, 2000-2009, 2019-2019, 2020-2029
    df['decade'] = pd.cut(x=df.release_year, bins=[1979,1989,1999,2009,2019,2029], 
                                                labels=['80s','90s','2000s','2010s','2020s'])

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

def spotify_split(df, target):
    '''
    This function takes in a dataframe and the string name of the target variable
    and splits it into test (15%), validate (15%), and train (70%). 
    It also splits test, validate, and train into X and y dataframes.
    Returns X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test.
    '''
    # first, since the target is a continuous variable and not a categorical one,
    # in order to use stratification, we need to turn it into a categorical variable with binning.
    bin_labels_5 = ['Low', 'Moderately Low', 'Moderate', 'Moderately High', 'High']
    df['pop_strat_bin'] = pd.qcut(df['popularity'], q=5, precision=0, labels=bin_labels_5)

    # split df into test (15%) and train_validate (85%)
    train_validate, test = train_test_split(df, test_size=.15, stratify=df['pop_strat_bin'], random_state=666)

    # drop column used for stratification
    train_validate = train_validate.drop(columns=['pop_strat_bin'])
    test = test.drop(columns=['pop_strat_bin'])

    # split train_validate off into train (82.35% of 85% = 70%) and validate (17.65% of 85% = %15)
    train, validate = train_test_split(train_validate, test_size=.1765, random_state=666)

    # split train into X & y
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X & y
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X & y
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    print('Shape of train:', X_train.shape, '| Shape of validate:', X_validate.shape, '| Shape of test:', X_test.shape)
    print('Percent train:', round(((train.shape[0])/df.count()[0]),2) * 100, '       | Percent validate:', round(((validate.shape[0])/df.count()[0]),2) * 100, '      | Percent test:', round(((test.shape[0])/df.count()[0]),2) * 100)

    return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test

def encode_features(df):
    '''
    This function encodes non-numeric features for use in modeling.
    Takes in df and returns df.
    '''
    # encode 'explicit'
    df['is_explicit'] = df.explicit.map({True: 1, False: 0})
    df = df.drop(columns=['explicit'])
    return df

##################################################### Scale the Data #####################################################

def scale_data(train, validate, test, predict, scaler):

    '''
    Scales a df based on scaler chosen: 'MinMax', 'Standard', or 'Robust'. 
    Needs three dfs: train, validate, and test. Fits the scaler object to train 
    only, transforms on all 3. Returns the three dfs scaled.
    'predict' is the target variable name.
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

################################################# Create Top Ten Labels Feature #################################################

def get_top_ten_labels(df):
    # Create a dataframe of the mean of popularity and label count grouped by label
    biggest_labels = df.groupby('label').popularity.agg(['mean', 'count']).sort_values(by=['count', 'mean'], ascending=False).head(20)
    # Create a list of the top ten labels by popularity
    top_ten_labels = list(biggest_labels.sort_values(by='mean', ascending=False).head(10).index)
    # Make a pattern by joining every label in top_ten_labels
    pattern = '|'.join(top_ten_labels)
    # Make new column with boolean variable for if the label is contained in the pattern
    df['top_ten_label'] = df.label.str.contains(pattern)
    # Convert boolean to int
    df['top_ten_label'] = df.top_ten_label.astype('int')
    return df

################################################# Create Record Label Features #################################################

def get_labels_features(df):
    # Create a dataframe of the mean of popularity and label count grouped by label
    biggest_labels = df.groupby('label').popularity.agg(['mean', 'count']).sort_values(by=['count', 'mean'], ascending=False).head(20)
    # Create a list of the top/bottom ten/five labels by popularity
    top_ten_labels = list(biggest_labels.sort_values(by='mean', ascending=False).head(10).index)
    top_five_labels = list(biggest_labels.sort_values(by='mean', ascending=False).head(5).index)
    worst_ten_labels = list(biggest_labels.sort_values(by='mean').head(10).index)
    worst_five_labels = list(biggest_labels.sort_values(by='mean').head(5).index)
    # Make a pattern by joining every label in our features
    pattern1 = '|'.join(top_ten_labels)
    pattern2 = '|'.join(top_ten_labels)
    pattern3 = '|'.join(top_ten_labels)
    pattern4 = '|'.join(top_ten_labels)
    # Make new column with boolean variable for if the label is contained in the pattern
    df['top_ten_label'] = df.label.str.contains(pattern1)
    df['top_five_label'] = df.label.str.contains(pattern1)
    df['worst_ten_label'] = df.label.str.contains(pattern1)
    df['worst_five_label'] = df.label.str.contains(pattern1)
    # Convert boolean to int
    df['top_ten_label'] = df.top_ten_label.astype('int')
    df['top_five_label'] = df.top_five_label.astype('int')
    df['worst_ten_label'] = df.worst_ten_label.astype('int')
    df['worst_five_label'] = df.worst_five_label.astype('int')
    
def modeling_prep():
    
    df = pd.read_csv('full-playlist.csv', index_col=0)
    # handle nulls in release data
    df['release_date'] = np.where(df['release_date'].str.len()==4, df.release_date.astype(str) + '-01-01', df['release_date'])
    # drop observations that contain nulls
    df = df.dropna()
    df = encode_features(df)
    df = get_top_ten_labels(df)

    album_dummies = pd.get_dummies(df.album_type, drop_first=True)
    df = pd.concat([df, album_dummies], axis=1)
    df[['compilation', 'single']]= df[['compilation', 'single']].astype('int')

    df = df.drop(columns=['album_popularity','label', 'artist', 
                        'album', 'release_date', 'track_name', 'album_id', 'album_type'])
    return df