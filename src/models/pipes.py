from pathlib import Path
from category_encoders.binary import BinaryEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from joblib import load
from pathlib import Path


def create_preprocessing_pipe(X: pd.DataFrame,
                              y: pd.Series = None) -> Pipeline:
    """
    Create a pipeline object with elements fitted to the training data.
    :param X: The dataframe of features
    :param y: The target series
    :return: a pipeline object
    """
    pipe = Pipeline([
        ('bin_encoder', BinaryEncoder(cols=['brewery_name'])),
        ('scaler', StandardScaler())
    ])

    pipe.fit(X)

    return pipe


def load_preprocessing_pipe(pipe_name: str) -> Pipeline:
    pipe = load(f'../models/{pipe_name}')
    return pipe


def load_label_encoder(label_encoder_name: str) -> LabelEncoder:
    le = load(f'../models/{label_encoder_name}')
    return le