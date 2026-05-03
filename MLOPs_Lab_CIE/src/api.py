from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
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
class JobInput(BaseModel):
    gpu_memory_gb: int = Field(..., ge=8, le=80)
    batch_size: int = Field(..., ge=8, le=256)
    model_params_millions: int = Field(..., ge=10, le=7000)
    queue_depth: int = Field(..., ge=1, le=20)

# Health endpoint
@app.get("/status")
def status():
    return {
        "status": "running",
        "model": "best_model",
        "version": "1.0"
    }

# Prediction endpoint
@app.post("/estimate")
def estimate(input: JobInput):
    df = pd.DataFrame([input.dict()])
    prediction = float(model.predict(df)[0])

    log_prediction(input.dict(), prediction)

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