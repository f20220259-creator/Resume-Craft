
import torch
import torch.nn as nn
import torch.optim as optim
from adapter_model import ResumeMLPAdapter
import os
from tqdm import tqdm

# Configuration
DATA_FILE = "dataset_tensors.pt"
MODEL_SAVE_PATH = "mlp_model.pth"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run preprocess_embeddings.py first.")
        return

    # Check Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading dataset...")
    data = torch.load(DATA_FILE)
    resume_emb = data['resume_embeddings'].to(device) # Shape: (N, 1024)
    jd_emb = data['jd_embeddings'].to(device)     # Shape: (N, 1024)
    
    dataset_size = len(resume_emb)
    input_dim = resume_emb.shape[1]
    
    print(f"Dataset Size: {dataset_size}, Embedding Dim: {input_dim}")

    # Initialize Model
    model = ResumeMLPAdapter(input_dim=input_dim, hidden_dim=2048, output_dim=input_dim).to(device)
    
    # We want the output to be close to the JD embedding
    # CosineEmbeddingLoss takes (input1, input2, target)
    # But here:
    #   Input = MLP(Resume, JD)
    #   Target = JD
    # We want Distance(Input, Target) to be minimized (Similarity maximized)
    # So we can use CosineSimilarity directly in the loss or MSE.
    
    # Loss Function: 1 - CosineSimilarity(Output, Target)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    
    # Target label for CosineEmbeddingLoss: 1 means inputs should be similar
    # loss(x, y) = 1 - cos(x, y)
    target_labels = torch.ones(BATCH_SIZE).to(device)

    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        permutation = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_r = resume_emb[indices]
            batch_j = jd_emb[indices]
            
            current_batch_size = len(indices)
            if current_batch_size != BATCH_SIZE:
                target = torch.ones(current_batch_size).to(device)
            else:
                target = target_labels

            optimizer.zero_grad()
            
            # Forward Pass: We pass BOTH Resume and JD to the adapter
            # The adapter learns to "transform" the Resume given the context of JD
            output = model(batch_r, batch_j, use_skip_connection=False) # Train the MLP weights
            
            # Loss: Output should be similar to JD
            loss = criterion(output, batch_j, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training Complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
