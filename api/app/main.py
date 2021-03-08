from fastapi import FastAPI
from starlette import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

nn_pipe = load('../models/pytorch_multi_beer_type_prediction.pt')
@app.get("/")
def read_root():
    return{"Welcome To" : "Beer Prediction"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer Type Prediction all ready to go!'


def format_feature(brewery_name: int, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int ):
    return {
        'Brewery Name' : [brewery_name],
        'Review Aroma' : [review_aroma],
        'Review Appearance': [review_appearance],
        'Review Palate': [review_palate],
        'Review Taste': [review_taste]
    }

@app.post('/beer/type')
def predict(brewery_name: int, review_aroma: int, review_appearance:int, review_palate: int, review_taste: int):
    features = format_feature(brewery_name, review_aroma, review_appearance, review_palate, review_taste )
    obs = pd.DataFrame(features)
    pred = nn_pipe.predict(obs)
    return JSONResponse(pred.tolist())

