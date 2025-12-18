# ResumeCraft - Handover & Setup Guide

This guide explains how to set up and run **ResumeCraft**, the local AI Resume Consultant and Tailoring tool. This application uses **Gemma 2B** (via Ollama) and a custom **MLP Adapter** to optimize resumes for specific job descriptions.

## 💻 System Requirements
- **OS**: Windows 10/11
- **GPU**: NVIDIA GTX 1060 (6GB) or better recommended
- **Storage**: ~5GB free space for models (can be on D: drive)
- **Software**: Python 3.11+ and Ollama

---

## 🚀 Step 1: Install Ollama (The AI Engine)
1. Download **Ollama for Windows** from [ollama.com](https://ollama.com).
2. Install it. When it finishes, you'll see an Ollama icon in your system tray.
3. Open a terminal (PowerShell or Command Prompt) and run:
   ```powershell
   ollama --version
   ```
   If it shows a version number, you're good to go.

### ⚠️ Important: Storage Space (Optional)
Ollama models are stored on C: drive by default. If your C: drive is full, you can move them to D: drive:
1. Close Ollama (Quit from system tray).
2. Open **Settings** > **System** > **About** > **Advanced system settings**.
3. Click **Environment Variables**.
4. Under **User variables**, click **New**.
   - Variable name: `OLLAMA_MODELS`
   - Variable value: `D:\OllamaModels` (or your preferred path)
5. Restart Ollama.

---

## 📦 Step 2: Download AI Models
Open your terminal and run these commands one by one to download the required brains:

1. **Gemma 2B** (The Consultant - smaller, faster):
   ```powershell
   ollama pull gemma:2b
   ```

2. **Mxbai Embed Large** (The Vectorizer - for resume tailoring):
   ```powershell
   ollama pull mxbai-embed-large
   ```

---

## 🐍 Step 3: Install Python & Dependencies
1. Ensure **Python 3.11 or newer** is installed.
2. Open this project folder in a terminal.
3. Create a virtual environment (recommended to keep things clean):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. Install all required code libraries:
   ```powershell
   pip install -r requirements.txt
   ```

---

## ▶️ Step 4: Run the Application
1. Make sure Ollama is running (check system tray).
2. In your terminal (inside the project folder):
   ```powershell
   streamlit run app.py
   ```
3. A browser window will open automatically at `http://localhost:8501`.

---

## 🛠️ How to Use
1. **Upload Resume**: PDF format.
2. **Paste Job Description**: Copy text from LinkedIn/portal.
3. **Run One-Click Tailor**:
   - Generates "Tailored Resume" content.
   - Shows Contextual Similarity Score.
4. **AI Consultant Tab** (The "Gemma" feature):
   - Click "Generate Expert Critique".
   - Get a **Skill Translation Table**, **Reframed Experience**, and **Pivots Pitch**.
   - **Note**: This runs on your GPU. It may take 30-60 seconds.

---

## ❓ Troubleshooting
- **500 Server Error**: Usually means Ollama isn't running. Open Ollama from Start Menu.
- **Out of Memory (OOM) / Slow**: 
   - The app automatically manages VRAM by unloading the embedding model before loading Gemma.
   - Ensure no other heavy games/apps are using your GPU.
- **Port Error**: If `localhost:8501` is taken, Streamlit use `localhost:8502`. Check the terminal output.

---
*Generated for ResumeCraft Handover | Dec 2025*
