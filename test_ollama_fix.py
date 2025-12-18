
import requests
import json
import numpy as np

BASE_URL = "http://localhost:11434"
MODEL = "mxbai-embed-large"
GEN_MODEL = "llama3"

def test_endpoint(name, url, payload):
    print(f"\n--- Testing {name} ---")
    print(f"URL: {url}")
    print(f"Payload keys: {list(payload.keys())}")
    try:
        resp = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print("SUCCESS")
            return True
        else:
            print(f"FAILURE: {resp.text}")
            return False
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

# 1. Test /api/embed (New Standard) with keep_alive
payload_new = {
    "model": MODEL,
    "input": "Hello world",
    "keep_alive": 0
}
test_endpoint("/api/embed (with keep_alive)", f"{BASE_URL}/api/embed", payload_new)

# 2. Test /api/embeddings (Legacy) with keep_alive
payload_legacy = {
    "model": MODEL,
    "prompt": "Hello world",
    "keep_alive": 0
}
test_endpoint("/api/embeddings (Legacy with keep_alive)", f"{BASE_URL}/api/embeddings", payload_legacy)

# 3. Test /api/generate (Llama3)
payload_gen = {
    "model": GEN_MODEL,
    "prompt": "Say hi",
    "stream": False,
    "keep_alive": 0
}
test_endpoint("/api/generate", f"{BASE_URL}/api/generate", payload_gen)
