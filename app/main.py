from typing import Optional, List
import pandas as pd
import torch
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, Query
from starlette.responses import JSONResponse
from joblib import load

import uvicorn

from src.models.pipes import load_preprocessing_pipe
from src.models.pipes import load_label_encoder
from src.models.pytorch import get_device

app = FastAPI()

model = torch.load('../models/model.torch',
                   map_location=lambda storage, loc: storage)


@app.get('/')
def read_root():
    root_dict = {
        'Project Objectives': 'The objective of this project is to predict '
                              'the beer style given the features '
                              '`brewery_name`, `review_aroma`, '
                              '`review_performance`, `review_appearance`, '
                              '`review_palate`, and `review_taste',
        'List of endpoints': ["/", "/health/", "/beer/type/", "/beers/type/",
                              "/model/architecture/"],
        'Inputs': {'`brewery_name`': 'str or List[str]',
                   '`review_performance`': 'float or List[float]',
                   '`review_appearance`': 'float or List[float]',
                   '`review_palate`': 'float or List[float]',
                   '`review_palate`': 'float or List[float]'},
        'GitHub link': 'https://github.com/roger-yu-ds/assignment_2'
    }

    response = JSONResponse(content=root_dict, status_code=200)

    return response


@app.get('/health', status_code=200)
def healthcheck():
    return 'PyTorch NN app up and running‼'


@app.post('/beer/type')
def predict_one(brewery_name: str,
                review_aroma: float,
                review_appearance: float,
                review_palate: float,
                review_taste: float):
    """
    Produces a single prediction for a single observation.
    - **brewery_name**:      a single brewery name
    - **review_aroma**:      a single review aroma
    - **review_appearance**: a single review appearance
    - **review_palate**:     a single review palate
    - **review_taste**:      a single review taste
    - **return**:            JSON of the prediction
    """
    device = get_device()
    df = pd.DataFrame({'brewery_name': [brewery_name],
                       'review_aroma': [review_aroma],
                       'review_appearance': [review_appearance],
                       'review_palate': [review_palate],
                       'review_taste': [review_taste]})

    # Encoding the data for prediction
    pipe = load_preprocessing_pipe('pipe.sav')
    df_trans = pipe.transform(df)
    df_tensor = torch.Tensor(np.array(df_trans)).to(device)

    # Prediction
    pred = model(df_tensor).argmax(1)

    # Decode results to produce human readable classes
    le = load_label_encoder('label_encoder.sav')
    pred_name = le.inverse_transform(pred.tolist())[0]

    return JSONResponse(pred_name)


@app.post('/beer/types')
def predict_many(brewery_name: List[str] = Query(None),
                 review_aroma: List[float] = Query(None),
                 review_appearance: List[float] = Query(None),
                 review_palate: List[float] = Query(None),
                 review_taste: List[float] = Query(None)):
    """
    Predict multiple observations.
    - **brewery_name**:      list of brewery names
    - **review_aroma**:      list of review aroma
    - **review_appearance**: list of review appearance
    - **review_palate**:     list of review palate
    - **review_taste**:      list of review taste
    - **return**:            JSON of predictions
    """
    device = get_device()
    df = pd.DataFrame({'brewery_name': brewery_name,
                       'review_aroma': review_aroma,
                       'review_appearance': review_appearance,
                       'review_palate': review_palate,
                       'review_taste': review_taste})

    # Encoding the data for prediction
    pipe = load_preprocessing_pipe('pipe.sav')
    df_trans = pipe.transform(df)
    df_tensor = torch.Tensor(np.array(df_trans)).to(device)

    # Prediction
    pred = model(df_tensor).argmax(1)

    # Decode results to produce human readable classes
    le = load_label_encoder('label_encoder.sav')
    pred_names = list(le.inverse_transform(pred.tolist()))

    return JSONResponse(pred_names)


@app.get('/model/architecture/')
def get_architecture():
    architecture_dict = {
        layer.split(':')[0][2:]: layer.split(':')[1]
        for layer in model.__str__().split('\n')[1:-1]
    }

    return JSONResponse(architecture_dict)


if __name__ == '__main__':
    uvicorn.run(app)
else:
    pass
© 2021 GitHub, Inc.