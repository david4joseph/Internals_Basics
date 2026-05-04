import pandas as pd
import json
import os
import numpy as np

# Load training data
train = pd.read_csv("data/training_data.csv")

# Load logs
logs = []
with open("logs/predictions.jsonl") as f:
    for line in f:
        logs.append(json.loads(line))

live_df = pd.DataFrame([l["input"] for l in logs])
preds = [l["prediction"] for l in logs]

alerts = []

def check(feature, threshold):
    train_mean = train[feature].mean()
    live_mean = live_df[feature].mean()
    shift = abs(train_mean - live_mean)

    status = "ALERT" if shift > threshold else "OK"

    alerts.append({
        "feature": feature,
        "train_mean": train_mean,
        "live_mean": live_mean,
        "shift": shift,
        "threshold": threshold,
        "status": status
    })

# thresholds from question
check("wind_speed_kmph", 10.74)
check("payload_mass_kg", 3516.61)

output = {
    "total_predictions": len(preds),
    "mean_prediction": float(np.mean(preds)),
    "drift_detected": any(a["status"] == "ALERT" for a in alerts),
    "alerts": alerts
}

os.makedirs("results", exist_ok=True)

with open("results/step4_s5.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Monitoring complete")