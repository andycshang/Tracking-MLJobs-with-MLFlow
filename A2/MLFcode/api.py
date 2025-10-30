from typing import Union
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np
from pydantic import BaseModel, Field, conlist

# ======================
# Load model
# ======================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

w = model["w"]
b = model["b"]
x_scaler = model["x_scaler"]
features = model["features"]

app = FastAPI()

# ======================
# Define input schema
# ======================
class HouseData(BaseModel):
    CRIM: float = Field(..., ge=0)
    ZN: float = Field(..., ge=0)
    INDUS: float
    CHAS: int = Field(..., ge=0, le=1)
    NOX: float = Field(..., ge=0, le=1)
    RM: float = Field(..., ge=0)
    AGE: float = Field(..., ge=0, le=100)
    DIS: float = Field(..., ge=0)
    RAD: int = Field(..., ge=1)
    TAX: float = Field(..., ge=0)
    PTRATIO: float = Field(..., ge=0)
    B: float = Field(..., ge=0)
    LSTAT: float = Field(..., ge=0)
# ======================
# Predict endpoint
# ======================
@app.post("/predict")
def predict_endpoint(input: HouseData):
    X = np.array([[input.CRIM, input.ZN, input.INDUS, input.CHAS, input.NOX,
                   input.RM, input.AGE, input.DIS, input.RAD, input.TAX,
                   input.PTRATIO, input.B, input.LSTAT]])
    X_scaled = x_scaler.transform(X)
    y_pred = np.dot(X_scaled, w) + b
    return {"prediction": float(y_pred[0])}

# ======================
# Other routes
# ======================
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# ======================
# Example Input:
# ======================
# {
#   "CRIM": 0.03,
#   "ZN": 18.0,
#   "INDUS": 2.3,
#   "CHAS": 0,
#   "NOX": 0.4,
#   "RM": 6.5,
#   "AGE": 45.0,
#   "DIS": 5.2,
#   "RAD": 1,
#   "TAX": 290.0,
#   "PTRATIO": 17.8,
#   "B": 390.0,
#   "LSTAT": 12.5
# }
