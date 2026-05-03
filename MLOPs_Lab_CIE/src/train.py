import pandas as pd
import numpy as np
import json
import os
import mlflow
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Load data
# =========================
df = pd.read_csv("data/training_data.csv")

X = df.drop("job_completion_min", axis=1)
y = df["job_completion_min"]

# IMPORTANT (from question paper)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MLflow setup
# =========================
mlflow.set_experiment("gpuforge-job-completion")

results = []
trained_models = {}

# =========================
# Function to train + log
# =========================
def run_model(model, name):
    with mlflow.start_run():
        mlflow.set_tag("experiment_type", "baseline_comparison")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        # Log to MLflow
        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        # Store model
        trained_models[name] = model

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        }

# =========================
# Train both models
# =========================
results.append(run_model(Ridge(), "Ridge"))
results.append(run_model(GradientBoostingRegressor(), "GradientBoosting"))

# =========================
# Select best model (RMSE)
# =========================
best = min(results, key=lambda x: x["rmse"])
best_model_name = best["name"]
best_model = trained_models[best_model_name]

# =========================
# Save best model
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")

# =========================
# Save JSON output (VERY IMPORTANT)
# =========================
os.makedirs("results", exist_ok=True)

output = {
    "experiment_name": "gpuforge-job-completion",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open("results/step1_tracking.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Task 1 complete")