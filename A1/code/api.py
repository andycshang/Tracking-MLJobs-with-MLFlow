from typing import Union
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np


# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

w = model["w"]
b = model["b"]
x_scaler = model["x_scaler"]
features = model["features"]

app = FastAPI()

class InputData(BaseModel):
    data: list[float]  # eg. [0.00632, 18.0, 2.31, ...]

@app.post("/predict")
def predict_endpoint(input: InputData):
    X = np.array([input.data])
    X_scaled = x_scaler.transform(X)
    y_pred = np.dot(X_scaled, w) + b
    return {"prediction": float(y_pred[0])}

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# predict_example:
# {
#   "data": [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]
# }