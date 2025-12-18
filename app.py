
import streamlit as st
import torch
import numpy as np
import os
from utils import extract_text_from_pdf
from ollama_module import LLMModel
from adapter_model import ResumeMLPAdapter
from decoder_module import ResumeDecoder

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="ResumeCraft AI",
    page_icon="✨",
    layout="wide"
)

# -------------------------------------------------
# ADVANCED CSS (Colorful + Modern)
# -------------------------------------------------
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Main container */
.block-container {
    padding-top: 2rem;
}

/* Card style */
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

/* Title */
.title {
    font-size: 48px;
    font-weight: 800;
    color: white;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    color: #e0e0ff;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #ff512f, #f09819);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

/* Text areas */
textarea {
    border-radius: 12px !important;
}

/* Section headers */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #4b0082;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# INITIALIZATION & STATE
# -------------------------------------------------
@st.cache_resource
def load_models():
    """Load and cache the Ollama client, MLP Adapter, and Decoder."""
    # Ollama Client
    ollama_client = LLMModel(model_name="mxbai-embed-large")
    
    # MLP Adapter
    # Critical Optimization: Force CPU for MLP to save VRAM for Ollama (Gemma 2B)
    # The MLP is tiny (1024->1024), so CPU inference is instant.
    device = torch.device("cpu") 
    # mxbai-embed-large has 1024 dimensions. Training uses hidden_dim=2048 (from train.py)
    mlp = ResumeMLPAdapter(input_dim=1024, hidden_dim=2048, output_dim=1024).to(device)
    
    # Load Trained Weights if available
    model_path = "mlp_model.pth"
    if os.path.exists(model_path):
        try:
            mlp.load_state_dict(torch.load(model_path, map_location=device))
            mlp.eval() # Set to eval mode for inference
            use_skip = False # Use actual MLP
            print(f"Loaded trained model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            use_skip = True # Fallback
    else:
        use_skip = True # Fallback for untrained
    
    # Decoder
    decoder = ResumeDecoder()
    
    return ollama_client, mlp, decoder, device, use_skip

try:
    ollama_client, mlp_model, decoder, device, use_skip_connection = load_models()
    st.toast(f"System initialized on {device}", icon="⚡")
    if not use_skip_connection:
        st.toast("Trained Model Loaded!", icon="🧠")
except Exception as e:
    st.error(f"Failed to initialize models: {e}")
    st.stop()

if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="title">✨ ResumeCraft AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Deep Learning Resume Tailoring (Frozen LLM + MLP Adapter)</div>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("⚙️ System Status")
st.sidebar.info(f"**MLP Adapter:** CPU (VRAM saving)")
st.sidebar.success("**Ollama (Gemma 2B):** GPU 🚀")

st.sidebar.markdown("---")
# Hidden hardcoding of model to ensure pipeline consistency
ollama_client.model_name = "mxbai-embed-large" 
# Just show the user what's happening
st.sidebar.info("🧠 Embedding: mxbai-embed-large")
st.sidebar.info("🤖 Consultant: gemma:2b (GPU)")

st.sidebar.markdown("---")
st.sidebar.caption("Project Phase: 3 (Refinement & UX)")

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
col1, col2 = st.columns(2)

# -------------------------------------------------
# RESUME INPUT
# -------------------------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Resume Input</div>', unsafe_allow_html=True)

    resume_option = st.radio(
        "Choose input method",
        ["📂 Upload PDF", "✍️ Paste Text"],
        horizontal=True
    )

    if resume_option == "📂 Upload PDF":
        pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        if pdf:
            extracted = extract_text_from_pdf(pdf)
            if "Error" in extracted:
                st.error(extracted)
            else:
                st.session_state.resume_text = extracted
                st.success("✅ Resume extracted successfully")
    else:
        st.session_state.resume_text = st.text_area(
            "Paste Resume Content",
            value=st.session_state.resume_text,
            height=260,
            placeholder="Paste your resume here..."
        )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# JOB DESCRIPTION INPUT
# -------------------------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Job Description</div>', unsafe_allow_html=True)

    jd_text = st.text_area(
        "Paste Job Description",
        height=300,
        placeholder="Paste the job description here..."
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# GENERATE BUTTON
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚀 Deep Tailoring Pipeline</div>', unsafe_allow_html=True)

generate = st.button("✨ Run MLP Adapter")

if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False

if generate:
    st.session_state.pipeline_run = True

if st.session_state.pipeline_run:
    if not st.session_state.resume_text.strip():
        st.error("❌ Please provide a resume.")
    elif not jd_text.strip():
        st.error("❌ Please provide a job description.")
    else:
        with st.status("Running Deep Learning Pipeline...", expanded=True) as status:
            
            # 1. Embeddings
            st.write("🧠 Generating Embeddings (Ollama)...")
            resume_emb = ollama_client.get_vector(st.session_state.resume_text)
            jd_emb = ollama_client.get_vector(jd_text)
            
            if resume_emb is None or jd_emb is None:
                st.error("Failed to get embeddings. Ensure Ollama is running.")
                status.update(label="Pipeline Failed", state="error")
            else:
                # 2. MLP Transformation
                st.write(f"⚡ Running MLP Adapter on {device}...")
                
                try:
                    with torch.no_grad():
                        # Pass to MLP
                        output_vector = mlp_model(resume_emb, jd_emb, use_skip_connection=use_skip_connection)
                    
                    st.success("Transformation Complete!")
                    status.update(label="Pipeline Completed!", state="complete")
                    
                    
                    # Store variables for tab display
                    resume_embedding = resume_emb
                    tailored_embedding = output_vector.cpu().numpy().flatten()
                    jd_embedding = jd_emb
                    
                    st.markdown("### 📊 Analysis & Results")
                    
                    # Create Tabs
                    tab1, tab2, tab3 = st.tabs(["📄 Tailored Resume", "🤖 AI Consultant (Gemma)", "🔧 Debug Info"])
                    
                    with tab1:
                        st.subheader("Tailored Resume Content")
                        if use_skip_connection:
                            st.info("ℹ️ Using UNTRAINED model (Skip Connection). Run training to improve results.")
                        else:
                            st.success("✅ Using TRAINED MLP Model.")

                        # Alignment Score
                        try:
                            from sklearn.metrics.pairwise import cosine_similarity
                            # Reshape for sklearn: (1, n_features)
                            r_vec = resume_embedding.reshape(1, -1)
                            j_vec = jd_embedding.reshape(1, -1)
                            t_vec = tailored_embedding.reshape(1, -1)
                            
                            original_score = cosine_similarity(r_vec, j_vec)[0][0]
                            new_score = cosine_similarity(t_vec, j_vec)[0][0]
                            
                            col_score1, col_score2 = st.columns(2)
                            col_score1.metric("Original Alignment", f"{original_score:.4f}", help="Similarity between original resume and JD")
                            col_score2.metric("Tailored Alignment 🎯", f"{new_score:.4f}", delta=f"{new_score-original_score:.4f}", help="Similarity after MLP transformation")
                        except Exception as e:
                            st.warning(f"Could not calculate alignment score: {e}")

                        # Decoding
                        with st.spinner("Decoding Output (RAG)..."):
                            # FIX 2: Correct Argument Order (Vector, Text) and remove invalid args
                            decoded_resume = decoder.decode(
                                tailored_embedding, # Vector first
                                st.session_state.resume_text # Text second
                            )
                        st.text_area("Final Result", decoded_resume, height=600)

                    with tab2:
                        st.subheader("💡 Strategic Advice (Gemma 2B)")
                        st.markdown("This analysis is generated by **Gemma 2B** to give you qualitative feedback.")
                        
                        if st.button("Generate Expert Critique"):
                            with st.spinner("Asking Gemma for advice..."):
                                st.session_state.critique = ollama_client.generate_critique(st.session_state.resume_text, jd_text)
                        
                        if "critique" in st.session_state:
                            st.markdown(st.session_state.critique)

                    with tab3:
                        st.subheader("Transformation Results")
                        colA, colB = st.columns(2)
                        colA.metric("Resume Vector Size", f"{resume_emb.shape[0]}")
                        colB.metric("Transformed Vector Size", f"{tailored_embedding.shape[0]}")

                    
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    status.update(label="Inference Error", state="error")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "🚀 ResumeCraft AI | Frozen LLM + MLP Adapter Architecture"
)
