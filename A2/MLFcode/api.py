from typing import Union
from fastapi import FastAPI, Query
import pickle
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
import os

# --------------------------
# Constants 版本信息
# --------------------------
APP_VERSION = "0.0"
MODEL_DIR = "models"  # 存放不同版本模型的目录
VERSION = "1.0"  # 默认最新模型版本
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"model_v{VERSION}.pkl")


# ======================
# Load model function
# ======================
def load_model(version: str):
    model_path = os.path.join(MODEL_DIR, f"model_v{version}.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"Model version {version} not found at {model_path}")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

# 默认加载最新模型
model_data = load_model(VERSION)
w = model_data["w"]
b = model_data["b"]
x_scaler = model_data["x_scaler"]
features = model_data["features"]

# ======================
# Initialize FastAPI
# ======================
app = FastAPI(title="Boston Housing API", version=APP_VERSION)

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
def predict_endpoint(
    input: HouseData,
    model_version: str = Query(None, description="Optional model version")
):
    version_to_use = model_version if model_version else VERSION

    # 加载指定版本模型
    model_data = load_model(version_to_use)
    w = model_data["w"]
    b = model_data["b"]
    x_scaler = model_data["x_scaler"]
    features = model_data["features"]

    # 构造输入特征
    X = np.array([[getattr(input, f) for f in features]])
    X_scaled = x_scaler.transform(X)

    # 预测
    y_pred = float(np.dot(X_scaled, w) + b)

    # metadata
    metadata = {
        "app_version": APP_VERSION,
        "model_version": version_to_use,
        "prediction_time": datetime.utcnow().isoformat() + "Z"
    }

    return {"prediction": y_pred, "metadata": metadata}

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
