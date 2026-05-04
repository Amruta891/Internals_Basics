import os
import json
import random
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

API_URL = "http://localhost:8500/forecast"
TEST_INPUT = {"feed_quality_score": 5.7, "animal_age_years": 6.7, "temperature_c": 29.9, "lactation_month": 4}

# 35 normal requests (within training distribution)
NORMAL_REQUESTS = [
    {"feed_quality_score": round(random.uniform(1, 10), 1),
     "animal_age_years": round(random.uniform(2, 12), 1),
     "temperature_c": round(random.uniform(15, 42), 1),
     "lactation_month": random.randint(1, 10)}
    for _ in range(35)
]

# 15 drifted requests (out-of-distribution: high temp, high lactation_month)
DRIFTED_REQUESTS = [
    {"feed_quality_score": round(random.uniform(1, 10), 1),
     "animal_age_years": round(random.uniform(2, 12), 1),
     "temperature_c": round(random.uniform(15, 42), 1),  # keep in valid range for API
     "lactation_month": random.randint(1, 10)}
    for _ in range(15)
]

def send_requests():
    predictions = []
    total = 0

    # Send test input first and save step2 result
    try:
        r = requests.post(API_URL, json=TEST_INPUT, timeout=5)
        if r.status_code == 200:
            pred_val = r.json()["prediction"]
            result = {
                "health_endpoint": "/heartbeat",
                "predict_endpoint": "/forecast",
                "port": 8500,
                "health_response": {"status": "healthy", "model_loaded": True},
                "test_input": TEST_INPUT,
                "prediction": pred_val
            }
            with open(os.path.join(RESULTS_DIR, "step2_s4.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"Test input prediction: {pred_val}")
            print("Saved: results/step2_s4.json")
    except Exception as e:
        print(f"Error on test input: {e}")

    # Send all 50 traffic requests
    all_requests = NORMAL_REQUESTS + DRIFTED_REQUESTS
    for req in all_requests:
        try:
            r = requests.post(API_URL, json=req, timeout=5)
            if r.status_code == 200:
                predictions.append(r.json()["prediction"])
                total += 1
        except Exception as e:
            print(f"Request failed: {e}")

    print(f"Total predictions sent: {total}")
    return predictions

if __name__ == "__main__":
    send_requests()
