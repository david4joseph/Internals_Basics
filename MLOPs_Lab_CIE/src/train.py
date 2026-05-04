import pandas as pd
import numpy as np
import json
import os
import mlflow
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Load data
# =========================
df = pd.read_csv("data/training_data.csv")

X = df.drop("countdown_hold_min", axis=1)
y = df["countdown_hold_min"]

# Required split (from question)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MLflow setup
# =========================
mlflow.set_experiment("launchpredict-countdown-hold-min")

results = []
models_dict = {}

# =========================
# Function to train + log
# =========================
def run_model(model, name):
    with mlflow.start_run():
        mlflow.set_tag("priority", "high")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Log params + metrics
        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        models_dict[name] = model

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

# =========================
# Train models
# =========================
results.append(run_model(LinearRegression(), "LinearRegression"))
results.append(run_model(RandomForestRegressor(), "RandomForest"))

# =========================
# Select best (lowest RMSE)
# =========================
best = min(results, key=lambda x: x["rmse"])
best_model_name = best["name"]
best_model = models_dict[best_model_name]

# =========================
# Save model
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")

# =========================
# Save JSON output
# =========================
os.makedirs("results", exist_ok=True)

output = {
    "experiment_name": "launchpredict-countdown-hold-min",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Task 1 complete")