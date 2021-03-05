# Solution:
import pandas as pd
import numpy as np

class NullModel:
    """
    Class used as baseline model for both regression and classification
    ...

    Attributes
    ----------
    target_type : str
        Type of ML problem (default regression)
    y : Numpy Array-like
        Target variable
    pred_value : Float
        Value to be used for prediction
    preds : Numpy Array
        Predicted array

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the predicted value to be used based on the problem type
    get_length
        Calculate the number of observations from the target variable
    predict(y)
        Generate the predictions
    fit_predict(y)
        Perform a fit followed by predict
    """
        
    
    def __init__(self, target_type: str = "regression"):
        self.target_type = target_type
        self.y = None
        self.pred_value = None
        self.preds = None
        
    def fit(self, y):
        self.y = y
        if self.target_type == "regression":
            self.pred_value = y.mean()
        else:
            from scipy.stats import mode
            self.pred_value = mode(y)[0][0]
            
    def get_length(self):
        return len(self.y)
    
    def predict(self, y):
        self.preds = np.full((self.get_length(), 1), self.pred_value)
        return self.preds
    
    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)