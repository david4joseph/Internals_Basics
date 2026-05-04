import argparse
import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--wind_speed_kmph", type=float, required=True)
parser.add_argument("--humidity_pct", type=float, required=True)
parser.add_argument("--payload_mass_kg", type=float, required=True)
parser.add_argument("--vehicle_type_index", type=int, required=True)

args = parser.parse_args()

# Prepare input
data = pd.DataFrame([{
    "wind_speed_kmph": args.wind_speed_kmph,
    "humidity_pct": args.humidity_pct,
    "payload_mass_kg": args.payload_mass_kg,
    "vehicle_type_index": args.vehicle_type_index
}])

# Predict
prediction = model.predict(data)[0]

print(f"Prediction: {prediction}")