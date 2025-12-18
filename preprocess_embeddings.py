
import csv
import torch
import numpy as np
from tqdm import tqdm
from ollama_module import LLMModel
import os

INPUT_FILE = "synthetic_training_dataset.csv"
OUTPUT_FILE = "dataset_tensors.pt"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run generate_jds.py first.")
        return

    print("Loading text data...")
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    print(f"Processing {len(data)} samples...")
    
    resume_embeddings = []
    jd_embeddings = []
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Will save tensors for device: {device}")
    
    # Initialize Ollama Client
    # Ensure this matches the embedding model you want to use
    client = LLMModel(model_name="mxbai-embed-large")

    for row in tqdm(data):
        resume_text = row['Resume']
        jd_text = row['Job_Description']
        
        # Get embeddings (returns numpy array)
        r_emb = client.get_vector(resume_text)
        j_emb = client.get_vector(jd_text)
        
        if r_emb is not None and j_emb is not None:
            resume_embeddings.append(r_emb)
            jd_embeddings.append(j_emb)

    if not resume_embeddings:
        print("No valid embeddings generated.")
        return

    # Convert to Tensors
    print("Converting to PyTorch tensors...")
    r_tensor = torch.tensor(np.array(resume_embeddings), dtype=torch.float32)
    j_tensor = torch.tensor(np.array(jd_embeddings), dtype=torch.float32)

    # Save
    torch.save({
        'resume_embeddings': r_tensor,
        'jd_embeddings': j_tensor
    }, OUTPUT_FILE)
    
    print(f"Saved dataset to {OUTPUT_FILE}")
    print(f"Resume Tensor Shape: {r_tensor.shape}")
    print(f"JD Tensor Shape: {j_tensor.shape}")

if __name__ == "__main__":
    main()
