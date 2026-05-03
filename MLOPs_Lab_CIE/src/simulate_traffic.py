import requests
import random

url = "http://127.0.0.1:8500/estimate"

# 40 normal requests
for _ in range(40):
    payload = {
        "gpu_memory_gb": random.randint(8, 80),
        "batch_size": random.randint(8, 256),
        "model_params_millions": random.randint(10, 1000),
        "queue_depth": random.randint(1, 10)
    }
    requests.post(url, json=payload)

# 10 drifted requests (LLM surge)
for _ in range(10):
    payload = {
        "gpu_memory_gb": 80,
        "batch_size": 256,
        "model_params_millions": random.randint(5000, 7000),
        "queue_depth": random.randint(15, 20)
    }
    requests.post(url, json=payload)

print("✅ Sent 50 requests")