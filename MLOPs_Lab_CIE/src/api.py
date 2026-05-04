from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import json
from datetime import datetime
app = FastAPI()

# Load model
MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)

# Input validation
class InputData(BaseModel):
    wind_speed_kmph: float = Field(..., ge=0, le=40)
    humidity_pct: float = Field(..., ge=30, le=90)
    payload_mass_kg: float = Field(..., ge=500, le=20000)
    vehicle_type_index: int = Field(..., ge=1, le=4)

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "best_model",
        "version": "1.0"
    }

# Prediction endpoint
@app.post("/estimate")
def estimate(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = float(model.predict(df)[0])

    log_prediction(data.dict(), prediction)

    return {"prediction": prediction}

def log_prediction(input_data, prediction):
    os.makedirs("logs", exist_ok=True)

    log_entry = {
        "timestamp": str(datetime.now()),
        "input": input_data,
        "prediction": prediction,
        "endpoint": "/estimate"
    }

    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")