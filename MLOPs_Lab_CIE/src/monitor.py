import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
LOGS_PATH = os.path.join(BASE_DIR, "logs", "predictions.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLDS = {
    "temperature_c": 7.73,
    "lactation_month": 1.65
}

def load_live_data():
    records = []
    with open(LOGS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                records.append(entry["input"])
    return pd.DataFrame(records)

def main():
    train_df = pd.read_csv(DATA_PATH)
    live_df = load_live_data()

    total_preds = len(live_df)
    
    # Load predictions for mean
    preds = []
    with open(LOGS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                preds.append(entry["prediction"])
    
    mean_pred = round(sum(preds) / len(preds), 4) if preds else 0.0

    alerts = []
    drift_detected = False

    for feature, threshold in THRESHOLDS.items():
        train_mean = round(train_df[feature].mean(), 2)
        live_mean = round(live_df[feature].mean(), 2)
        shift = round(abs(live_mean - train_mean), 2)
        status = "ALERT" if shift > threshold else "OK"
        if status == "ALERT":
            drift_detected = True
        alerts.append({
            "feature": feature,
            "train_mean": train_mean,
            "live_mean": live_mean,
            "shift": shift,
            "threshold": threshold,
            "status": status
        })
        print(f"{feature}: train={train_mean}, live={live_mean}, shift={shift}, threshold={threshold} -> {status}")

    result = {
        "total_predictions": total_preds,
        "mean_prediction": mean_pred,
        "drift_detected": drift_detected,
        "alerts": alerts
    }
    with open(os.path.join(RESULTS_DIR, "step3_s5.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDrift detected: {drift_detected}")
    print("Saved: results/step3_s5.json")

if __name__ == "__main__":
    main()
