from fastapi import FastAPI, Query
from typing import List, Optional
from starlette.responses import JSONResponse
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
import torch as torch
from sklearn.pipeline import Pipeline
from joblib import load
import numpy as np
import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.abspath('../'))

app = FastAPI()

model = torch.load('../models/pytorch_multi_beer_type_prediction_nn_pipeline.pt', encoding='ascii')

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

def make_predictions(obs):

    # Use saved Pipeline object - 
    # applying datatransformation - scaling and same encoding used on training data
    trained_encoder = load( '../models/pipeline/trained_encoder.sav')
    obs_trans = trained_encoder.transform(obs)

    #  Convert data to tensor format
    device = get_device()
    df_tensor= torch.Tensor(np.array(obs_trans)).to(device)

    #  Make Prediction
    prediction = model(df_tensor).argmax(1)

    # Decode the predicted encoded target using inverse_transform method   
    le = load('../models/pipeline/te_pipeline.sav')
    pred_name = le.inverse_transform(prediction.tolist())[0]
    return pred_name

@app.get("/")
def read_root():
    return 'Welcome To Beer Prediction'

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer Type Prediction all ready to go!'

@app.post('/beer/type/single')
def predict_single(brewery_name: str, review_aroma: int, review_appearance:int, review_palate: int, review_taste: int):
    #features = format_feature(brewery_name, review_aroma, review_appearance, review_palate, review_taste )

    obs = pd.DataFrame({'brewery_name': [brewery_name],
                       'review_aroma': [review_aroma],
                       'review_appearance': [review_appearance],
                       'review_palate': [review_palate],
                       'review_taste': [review_taste]})

    
    trained_encoder = load( '../models/pipeline/trained_encoder.sav')
    obs_trans = trained_encoder.transform(obs)

    #  Convert data to tensor format
    device = get_device()
    df_tensor= torch.Tensor(np.array(obs_trans)).to(device)

    #  Make Prediction
    prediction = model(df_tensor).argmax(1)

    # Decode the predicted encoded target using inverse_transform method   
    le = load('../models/pipeline/te_pipeline.sav')
    pred_name = le.inverse_transform(prediction.tolist())[0]
    # pred = make_predictions(obs)
    return pred_name

### Target
@app.post('/beer/type/many')
def predict_many(brewery_name: List[str] = Query(None), review_aroma: List[int] = Query(None), 
                review_appearance: List[int] = Query(None),review_palate: List[int] = Query(None),
                review_taste: List[int] = Query(None)):

    obs = pd.DataFrame({'brewery_name': [brewery_name],
                        'review_aroma': [review_aroma],
                        'review_appearance': [review_appearance],
                        'review_palate': [review_palate],
                        'review_taste': [review_taste]})

    # pred_name = make_predictions(obs)
    trained_encoder = load( '../models/pipeline/trained_encoder.sav')
    obs_trans = trained_encoder.transform(obs)

    #  Convert data to tensor format
    device = get_device()
    df_tensor= torch.Tensor(np.array(obs_trans)).to(device)

    #  Make Prediction
    prediction = model(df_tensor).argmax(1)

    # Decode the predicted encoded target using inverse_transform method   
    le = load('../models/pipeline/te_pipeline.sav')
    pred_name = le.inverse_transform(prediction.tolist())
    return JSONResponse(pred_name.tolist())
                

@app.get('model/architechture')
def get_model_architecture():
    return {model}


    




