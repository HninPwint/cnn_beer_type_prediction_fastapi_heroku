from fastapi import FastAPI, Query, Body
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
from src.models.pytorch import New_LabelEncoder

app = FastAPI()

model = torch.load('../models/pytorch_multi_beer_type_prediction_nn_pipeline.pt', encoding='ascii')

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

def format_feature(brewery_name: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int ):
    return {
        'Brewery Name' : [brewery_name],
        'Review Aroma' : [review_aroma],
        'Review Appearance': [review_appearance],
        'Review Palate': [review_palate],
        'Review Taste': [review_taste]
    }

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
    pred_name = le.inverse_transform(prediction.tolist())
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
    pred_name = make_predictions(obs)
    return pred_name[0]

### Target
@app.post('/beer/type/many')
def predict_many(brewery_name: List[str] = Query(None), review_aroma: List[int] = Query(None), 
                review_appearance: List[int] = Query(None),review_palate: List[int] = Query(None),
                review_taste: List[int] = Query(None)):

     obs = pd.DataFrame({'brewery_name': tuple(brewery_name),
                        'review_aroma': tuple(review_aroma),
                        'review_appearance': tuple(review_appearance),
                        'review_palate': tuple(review_palate),
                        'review_taste': tuple(review_taste)}, 
                        columns= ['brewery_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste'])

     pred_name = make_predictions(obs)  
     return JSONResponse(pred_name.tolist())
                

@app.get('model/architechture')
def get_model_architecture():
    return {model}
    

### Seems working
# @app.post('/beer/type/muli')
# def predict_many_list(brewery_name: Optional[List[str]], review_aroma: Optional[List[int]], review_appearance:Optional[List[int]], review_palate: Optional[List[int]], review_taste: Optional[List[int]]):
#     features = format_feature(brewery_name, review_aroma, review_appearance, review_palate, review_taste )
#     obs = pd.DataFrame(features)
#     pred = model.predict(obs)
#     return JSONResponse(pred.tolist())


### Seems working
class Item(BaseModel):
    brewery_name: str
    review_aroma: int
    review_appearance: int
    review_palate: int
    review_taste: int

@app.post("/beer/type/many/json")
def predict_many_json(item: Item = Body(..., embed=True)):
    #feature = format_feature(item.brewery_name, item.review_aroma, item.review_appearance, item.review_palate, item.review_taste)
    # obs = pd.DataFrame (feature,columns=['brewery_name','review_aroma','review_appearance', 'review_palate', 'review_taste'])

    obs = pd.DataFrame({'brewery_name': item.brewery_name,
                        'review_aroma': item.review_aroma,
                        'review_appearance': item.review_appearance,
                        'review_palate': item.review_palate,
                        'review_taste': item.review_taste}, columns= ['brewery_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste'])

    pred_name = make_predictions(obs)  
    return JSONResponse(pred_name.tolist())