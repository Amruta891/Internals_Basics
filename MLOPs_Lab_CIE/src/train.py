import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["feed_quality_score", "animal_age_years", "temperature_c", "lactation_month"]
TARGET = "milk_yield_litres"
EXPERIMENT_NAME = "herdwatch-milk-yield-litres"

def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return round(mae, 4), round(rmse, 4)

def main():
    mlflow.set_tracking_uri(f"file://{os.path.join(BASE_DIR, 'mlruns')}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    models_config = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3),
    }

    results = []
    for name, model in models_config.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("team", "ml_engineering")
            params = model.get_params()
            for k, v in params.items():
                mlflow.log_param(k, v)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae, rmse = compute_metrics(y_test, y_pred)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, artifact_path="model")
            results.append({"name": name, "mae": mae, "rmse": rmse, "model": model})
            print(f"{name} -> MAE: {mae}, RMSE: {rmse}")

    best = min(results, key=lambda x: x["rmse"])
    best_name = best["name"]
    best_model = best["model"]

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    step1 = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
        "best_model": best_name,
        "best_metric_name": "rmse",
        "best_metric_value": best["rmse"]
    }
    with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "w") as f:
        json.dump(step1, f, indent=2)

    print(f"\nBest model: {best_name} with RMSE={best['rmse']}")
    print("Saved: results/step1_s1.json")

if __name__ == "__main__":
    main()
