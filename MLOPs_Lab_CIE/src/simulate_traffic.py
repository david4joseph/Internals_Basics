import requests
import random

url = "http://127.0.0.1:8000/estimate"

# 30 normal requests (keep tight)
for _ in range(30):
    payload = {
        "wind_speed_kmph": random.uniform(5, 15),
        "humidity_pct": random.uniform(40, 60),
        "payload_mass_kg": random.uniform(1000, 5000),
        "vehicle_type_index": random.randint(1, 4)
    }
    requests.post(url, json=payload)

# 20 VERY STRONG drifted requests
for _ in range(20):
    payload = {
        "wind_speed_kmph": random.uniform(38, 40),
        "humidity_pct": random.uniform(85, 90),
        "payload_mass_kg": random.uniform(18000, 20000),
        "vehicle_type_index": random.randint(1, 4)
    }
    requests.post(url, json=payload)
print("✅ Sent 50 requests")