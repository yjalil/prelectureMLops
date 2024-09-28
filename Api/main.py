from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from starlette.responses import Response

app = FastAPI()
@app.get('/')
def read_root():
    return {'message': 'Api is running'}
@app.post('/predict')
def predict(distance_km : int, passenger_count: int, is_day: int):
    model = tf.keras.models.load_model('./results.keras')
    X= np.array([[distance_km], [passenger_count], [is_day]])
    X= np.reshape(X,(-1,3))
    prediction = model.predict(X)
    df =pd.DataFrame(prediction)
    print(f"MMMMMMMMMMMMMMMMMMMMMMM, {df.head()}")
    return  Response(content=df.to_json(orient= 'records'))
