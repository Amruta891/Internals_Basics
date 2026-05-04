import os
import json
import pickle
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions.jsonl")

app = FastAPI(title="HerdWatch Milk Yield API")

model = None

class FarmInput(BaseModel):
    feed_quality_score: float = Field(..., ge=1.0, le=10.0)
    animal_age_years: float = Field(..., ge=2.0, le=12.0)
    temperature_c: float = Field(..., ge=15.0, le=42.0)
    lactation_month: int = Field(..., ge=1, le=10)

class PredictionOutput(BaseModel):
    prediction: float

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

@app.get("/heartbeat")
def heartbeat():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/forecast")
def forecast(data: FarmInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = [[
        data.feed_quality_score,
        data.animal_age_years,
        data.temperature_c,
        data.lactation_month
    ]]
    prediction = float(round(model.predict(features)[0], 4))

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data.dict(),
        "prediction": prediction
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": prediction}

def save_step2_result(test_input, prediction):
    result = {
        "health_endpoint": "/heartbeat",
        "predict_endpoint": "/forecast",
        "port": 8500,
        "health_response": {"status": "healthy", "model_loaded": True},
        "test_input": test_input,
        "prediction": prediction
    }
    with open(os.path.join(RESULTS_DIR, "step2_s4.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("Saved: results/step2_s4.json")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8500, reload=False)
