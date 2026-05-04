import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(BASE_DIR, "data", "training_data.csv")
DATA_NEW = os.path.join(BASE_DIR, "data", "new_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["feed_quality_score", "animal_age_years", "temperature_c", "lactation_month"]
TARGET = "milk_yield_litres"
MIN_IMPROVEMENT = 1.0

def get_model_instance(name):
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
    return LinearRegression()

def main():
    orig_df = pd.read_csv(DATA_TRAIN)
    new_df = pd.read_csv(DATA_NEW)
    combined_df = pd.concat([orig_df, new_df], ignore_index=True)

    X_orig = orig_df[FEATURES]
    y_orig = orig_df[TARGET]
    X_orig_train, X_test, y_orig_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

    # Load champion model
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        champion = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "r") as f:
        best_model_name = f.read().strip()

    champion_pred = champion.predict(X_test)
    champion_mae = round(mean_absolute_error(y_test, champion_pred), 4)

    # Retrain same model type on combined data
    X_combined = combined_df[FEATURES]
    y_combined = combined_df[TARGET]
    X_comb_train, _, y_comb_train, _ = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    retrained = get_model_instance(best_model_name)
    retrained.fit(X_comb_train, y_comb_train)

    retrained_pred = retrained.predict(X_test)
    retrained_mae = round(mean_absolute_error(y_test, retrained_pred), 4)

    improvement = round(champion_mae - retrained_mae, 4)

    if improvement >= MIN_IMPROVEMENT:
        action = "promoted"
        with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
            pickle.dump(retrained, f)
        print("Retrained model promoted!")
    else:
        action = "kept_champion"
        print("Champion retained.")

    result = {
        "original_data_rows": len(orig_df),
        "new_data_rows": len(new_df),
        "combined_data_rows": len(combined_df),
        "champion_mae": champion_mae,
        "retrained_mae": retrained_mae,
        "improvement": improvement,
        "min_improvement_threshold": MIN_IMPROVEMENT,
        "action": action,
        "comparison_metric": "mae"
    }
    with open(os.path.join(RESULTS_DIR, "step4_s8.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Champion MAE: {champion_mae}, Retrained MAE: {retrained_mae}, Improvement: {improvement}")
    print(f"Action: {action}")
    print("Saved: results/step4_s8.json")

if __name__ == "__main__":
    main()
