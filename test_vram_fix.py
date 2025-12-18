
import requests
import time
import json

BASE_URL = "http://localhost:11434"

def step(name, func):
    print(f"\n STEP: {name}...")
    try:
        start = time.time()
        func()
        print(f" SUCCESS ({time.time() - start:.2f}s)")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def load_embedding():
    # Simulate the MLP Adapter running
    url = f"{BASE_URL}/api/embeddings"
    payload = {
        "model": "mxbai-embed-large:latest",
        "prompt": "This is a resume text to embed for the MLP.",
    }
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"Status {resp.status_code}: {resp.text}")

def force_unload():
    # The FIX: Tell Ollama to immediately dump the embedding model
    print("   (Sending unload signal...)")
    # Corrected: Use the embeddings endpoint for an embedding model
    url = f"{BASE_URL}/api/embeddings"
    payload = {
        "model": "mxbai-embed-large:latest",
        "prompt": "unload",
        "keep_alive": 0
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except:
        pass

def invoke_consultant():
    # Simulate the Consultant Button
    url = f"{BASE_URL}/api/generate"
    payload = {
        "model": "gemma:2b",
        "prompt": "Are you working? Reply with YES.",
        "stream": False,
        "options": {
            "num_ctx": 2048 # Reduce context window to save VRAM
        }
    }
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"Status {resp.status_code}: {resp.text}")
    print(f"   Response: {resp.json().get('response')}")

print("STARTING VRAM SWAP TEST")
print("---------------------------------")

if step("1. Run MLP Embedding (mxbai)", load_embedding):
    
    # Run the fix
    step("2. FORCE UNLOAD (The Fix)", force_unload)
    
    # Give it a tiny breathing room to verify stability
    time.sleep(5)
    
    step("3. Invoke Consultant (Gemma 2B)", invoke_consultant)
