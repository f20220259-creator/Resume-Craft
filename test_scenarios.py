"""
ResumeCraft - Isolated Test Scenarios
Tests the MLP Adapter + Decoder + Gemma Consultant against 4 JD scenarios.
"""
import os
import sys
import torch
import numpy as np

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollama_module import LLMModel
from adapter_model import ResumeMLPAdapter
from decoder_module import ResumeDecoder
from utils import extract_text_from_pdf

# Configuration
RESUME_PATH = r"d:\ResumeCraft-main\ResumeCraft-main\Ayush Kumar CV 2025 (9).pdf"
MODEL_PATH = "mlp_model.pth"

# Test JDs
SCENARIOS = {
    "1. AI Engineer (Perfect Match)": """
Job Title: AI Engineer
We are looking for an AI Engineer to build Generative AI applications.
Responsibilities:
- Develop and deploy LLM-based applications using Python and PyTorch.
- Fine-tune open-source models (Llama3, Mistral) for specific tasks.
- Build RAG pipelines and integrate vector databases.
- Optimize deep learning models for inference on GPUs.
Requirements:
- Strong proficiency in Python, PyTorch, and CUDA.
- Experience with HuggingFace, LangChain, and Ollama.
- Knowledge of backend APIs (FastAPI/Streamlit).
""",
    "2. Full Stack Developer (Pivot)": """
Job Title: Senior Full Stack Developer
Seeking a developer to build scalable web applications.
Responsibilities:
- Design and implement responsive user interfaces using React.js.
- Build RESTful APIs and microservices using Node.js or Python.
- Manage database schemas (PostgreSQL/MongoDB).
- Ensure application performance and responsiveness.
Requirements:
- 3+ years experience in Full Stack Development.
- Proficiency in HTML, CSS, JavaScript, and React.
- Experience with cloud platforms (AWS/Azure) and CI/CD pipelines.
""",
    "3. Product Manager (Management)": """
Job Title: Technical Product Manager
We need a TPM to bridge the gap between engineering and product.
Responsibilities:
- Define product roadmap and technical requirements.
- Collaborate with engineering teams to deliver AI features.
- Translate business goals into technical specifications.
- Prioritize backlog and manage sprint planning.
Requirements:
- Strong technical background in software engineering.
- Ability to communicate complex technical concepts to stakeholders.
- Experience leading projects from conception to launch.
""",
    "4. Head Chef (Stress Test)": """
Job Title: Head Chef
Looking for an experienced Chef to lead our kitchen using AI-driven inventory.
Responsibilities:
- Create new menu items and oversee food preparation.
- Manage kitchen staff and supply ordering.
- Ensure high standards of food safety.
"""
}

def load_models():
    """Load all required models."""
    print("Loading models...")
    
    # Ollama Client (for embeddings and critique)
    ollama_client = LLMModel(model_name="mxbai-embed-large")
    
    # MLP Adapter
    device = torch.device("cpu")  # Keep on CPU to save VRAM for Ollama
    mlp = ResumeMLPAdapter(input_dim=1024, hidden_dim=2048, output_dim=1024).to(device)
    
    # Load trained weights
    use_skip = True
    if os.path.exists(MODEL_PATH):
        try:
            mlp.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            mlp.eval()
            use_skip = False
            print(f"  [OK] Loaded trained MLP from {MODEL_PATH}")
        except Exception as e:
            print(f"  [WARN] Error loading model: {e}")
    else:
        print(f"  [WARN] Model not found at {MODEL_PATH}, using skip connection")
    
    # Decoder
    decoder = ResumeDecoder()
    print("  [OK] Models loaded\n")
    
    return ollama_client, mlp, decoder, device, use_skip

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_scenario(name, jd_text, resume_text, ollama_client, mlp, decoder, device, use_skip):
    """Run a single test scenario."""
    print("=" * 60)
    print(f"SCENARIO: {name}")
    print("=" * 60)
    
    # Step 1: Generate embeddings
    print("[1/4] Generating embeddings...")
    resume_vec = ollama_client.get_vector(resume_text)
    jd_vec = ollama_client.get_vector(jd_text)
    
    if resume_vec is None or jd_vec is None:
        print("  [ERROR] Failed to generate embeddings")
        return
    
    original_similarity = cosine_similarity(resume_vec, jd_vec)
    print(f"  Original Alignment: {original_similarity:.2%}")
    
    # Step 2: Run MLP Adapter
    print("[2/4] Running MLP Adapter...")
    resume_tensor = torch.tensor(resume_vec, dtype=torch.float32).to(device)
    jd_tensor = torch.tensor(jd_vec, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        if use_skip:
            tailored_vec = resume_tensor.numpy()
        else:
            tailored_vec = mlp(resume_tensor.unsqueeze(0), jd_tensor.unsqueeze(0)).squeeze(0).numpy()
    
    tailored_similarity = cosine_similarity(tailored_vec, jd_vec)
    delta = tailored_similarity - original_similarity
    print(f"  Tailored Alignment: {tailored_similarity:.2%} (Delta: {delta:+.2%})")
    
    # Step 3: Decode to text
    print("[3/4] Decoding to text...")
    decoded_resume = decoder.decode(tailored_vec, resume_text)
    print(f"  Decoded Resume (first 500 chars):\n{decoded_resume[:500]}...")
    
    # Step 4: Generate Critique
    print("\n[4/4] Generating AI Consultant Critique (Gemma 2B on GPU)...")
    critique = ollama_client.generate_critique(resume_text, jd_text)
    print(f"\n--- CONSULTANT ADVICE ---\n{critique}\n")
    
    print("-" * 60)
    print()

def main():
    print("\n" + "=" * 60)
    print("ResumeCraft - Isolated Scenario Testing")
    print("=" * 60 + "\n")
    
    # Load resume
    print(f"Loading resume from: {RESUME_PATH}")
    if not os.path.exists(RESUME_PATH):
        print(f"[ERROR] Resume not found at {RESUME_PATH}")
        return
    
    resume_text = extract_text_from_pdf(RESUME_PATH)
    if not resume_text:
        print("[ERROR] Failed to extract text from resume")
        return
    print(f"  [OK] Extracted {len(resume_text)} characters\n")
    
    # Load models
    ollama_client, mlp, decoder, device, use_skip = load_models()
    
    # Run each scenario
    for name, jd_text in SCENARIOS.items():
        run_scenario(name, jd_text, resume_text, ollama_client, mlp, decoder, device, use_skip)
    
    print("\n" + "=" * 60)
    print("All scenarios completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
