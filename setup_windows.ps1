# setup_windows.ps1

Write-Host "Starting ResumeCraft Environment Setup..."

# 1. Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python detected: $pythonVersion"
}
catch {
    Write-Host "Python not found! Please install Python 3.10+"
    exit
}

# 2. Install Python Dependencies
Write-Host "Installing Python Dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "Dependencies installed!"
}
else {
    Write-Host "Failed to install dependencies."
    exit
}

# 3. Check/Install Ollama
Write-Host "Checking for Ollama..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ollama is already installed!"
}
else {
    Write-Host "Ollama not found. Downloading installer..."
    $installerUrl = "https://ollama.com/download/OllamaSetup.exe"
    $installerPath = "$PWD\OllamaSetup.exe"
    
    try {
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
        Write-Host "Download complete: $installerPath"
        Write-Host "Launching Installer... Please complete the setup in the popup window."
        Start-Process -FilePath $installerPath -Wait
        
        Write-Host "NOTE: You may need to restart your terminal after installation."
    }
    catch {
        Write-Host "Failed to download Ollama. Please download manually from https://ollama.com"
    }
}

# 4. Pull Model
Write-Host "Attempting to pull LLaMA-3 Model..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Running: ollama pull llama3"
    ollama pull llama3
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Model downloaded successfully!"
    }
    else {
        Write-Host "Model pull failed. Ensure Ollama is running."
    }
}
else {
    Write-Host "Ollama command not found yet. Please restart your terminal and run 'ollama pull llama3'."
}

Write-Host "Setup Phase 1 Complete!"
Write-Host "To start the app, run: streamlit run app.py"
