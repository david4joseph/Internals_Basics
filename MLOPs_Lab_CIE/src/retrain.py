import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# =========================
# Load data
# =========================
old_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined_df = pd.concat([old_df, new_df], ignore_index=True)

# =========================
# Prepare data
# =========================
X_old = old_df.drop("job_completion_min", axis=1)
y_old = old_df["job_completion_min"]

X_combined = combined_df.drop("job_completion_min", axis=1)
y_combined = combined_df["job_completion_min"]

# SAME split (important for fair comparison)
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(
    X_old, y_old, test_size=0.2, random_state=42
)

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# =========================
# Load champion model type
# =========================
# Read which model was best in Task 1
with open("results/step1_tracking.json") as f:
    step1 = json.load(f)

best_model_name = step1["best_model"]

if best_model_name == "Ridge":
    champion_model = Ridge()
    retrain_model = Ridge()
else:
    champion_model = GradientBoostingRegressor()
    retrain_model = GradientBoostingRegressor()

# =========================
# Train old (champion)
# =========================
champion_model.fit(X_train_old, y_train_old)
pred_old = champion_model.predict(X_test_old)

champion_rmse = np.sqrt(mean_squared_error(y_test_old, pred_old))

# =========================
# Train retrained model
# =========================
retrain_model.fit(X_train_new, y_train_new)
pred_new = retrain_model.predict(X_test_new)

retrained_rmse = np.sqrt(mean_squared_error(y_test_new, pred_new))

# =========================
# Compare
# =========================
improvement = champion_rmse - retrained_rmse

if improvement >= 0.5:
    action = "promoted"
    final_model = retrain_model
else:
    action = "kept_champion"
    final_model = champion_model

# =========================
# Save final model
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/model.pkl")

# =========================
# Save JSON output
# =========================
output = {
    "original_data_rows": len(old_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": float(champion_rmse),
    "retrained_rmse": float(retrained_rmse),
    "improvement": float(improvement),
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)

with open("results/step4_retraining.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Task 4 complete")