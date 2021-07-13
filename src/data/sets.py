import pandas as pd
from pathlib import Path, WindowsPath
from dotenv import find_dotenv
from typing import Tuple
from sklearn.model_selection import train_test_split

project_dir = Path(find_dotenv()).parent

def subset_x_y(target, features, start_index:int, end_index:int):

    '''
    '''
    return features[start_index:end_index], target[start_index:end_index]

def split_sets_by_time(df, target_col, test_ratio=0.2, to_numpy=False):
    '''
    '''
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(target)/5)

    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(target))

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_sets_random(df, target_col, test_ratio=0.2, to_numpy=False):
    """Split sets randomly
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)
    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    features, target = pop_target(df=df, target_col=target_col,
                                  to_numpy=to_numpy)

    X_data, X_test, y_data, y_test = train_test_split(features, target,
                                                      test_size=test_ratio,
                                                      random_state=8)

    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data,
                                                      test_size=val_ratio,
                                                      random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_sets(X_train: pd.DataFrame = None,
              y_train: pd.Series = None,
              X_val: pd.DataFrame = None,
              y_val: pd.Series = None,
              X_test: pd.DataFrame = None,
              y_test: pd.Series = None,
              path: WindowsPath = None):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    if X_train is not None:
        X_train.to_csv(path.joinpath('X_train.csv'), index=False)
    if y_train is not None:
        y_train.to_csv(path.joinpath('y_train.csv'), index=False)
    if X_val is not None:
        X_val.to_csv(path.joinpath('X_val.csv'), index=False)
    if y_val is not None:
        y_val.to_csv(path.joinpath('y_val.csv'), index=False)
    if X_test is not None:
        X_test.to_csv(path.joinpath('X_test.csv'), index=False)
    if y_test is not None:
        y_test.to_csv(path.joinpath('y_test.csv'), index=False)

# def load_sets(path: WindowsPath = project_dir / 'data/processed') -> \
# def load_sets(path ='../data/processed') -> \

# def load_sets(path ='../../data/processed') -> str:
#     import os
#     return (os.path.join(path, "X_train.csv"))


def load_sets(path: WindowsPath = project_dir / 'data/processed') -> \
        Tuple[pd.DataFrame,
              pd.DataFrame,
              pd.Series,
              pd.Series]:
    """
    :param path:
    :return:
    """
    import os
   
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(path,"X_test.csv"))
    X_val = pd.read_csv(os.path.join(path,"X_val.csv"))
    y_train = pd.read_csv(os.path.join(path,"y_train.csv"))
    y_test = pd.read_csv(os.path.join(path,"y_test.csv"))
    y_val = pd.read_csv(os.path.join(path,"y_val.csv"))

    return X_train, X_test, X_val, y_train, y_test, y_val


def pop_target(df, target_col, to_numpy=False):
    """Extract target variable from dataframe and convert to nympy arrays if required

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable
    to_numpy : bool
        Flag stating to convert to numpy array or not

    Returns
    -------
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing all features
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    
    if to_numpy:
        df_copy = df_copy.to_numpy()
        target = target.to_numpy()
    
    return df_copy, target