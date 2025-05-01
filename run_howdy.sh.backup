#!/bin/bash

# HowdyTTS Launcher Script for Conda Environment
# This script is specifically for running HowdyTTS in a conda environment

# Define your conda environment name
CONDA_ENV="howdy310"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Conda first."
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
# Source conda to ensure conda activate works in script
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
else
    echo "❌ Cannot find conda.sh. Please make sure conda is properly installed."
    exit 1
fi

# Activate the environment
conda activate $CONDA_ENV
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

echo "✅ Conda environment activated: $CONDA_ENV"

# Run the fix script for ONNX Runtime if needed
if ! python -c "import onnxruntime_silicon" 2>/dev/null; then
    echo "⚠️ onnxruntime-silicon not properly installed, running fix script..."
    python fix_onnx_runtime.py
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 2
fi

# Check if FastWhisperAPI is running
if ! curl -s http://localhost:8000/info > /dev/null; then
    echo "Starting FastWhisperAPI..."
    cd FastWhisperAPI
    # Use the Python from our conda environment
    python -m uvicorn main:app --reload &
    FASTWHISPER_PID=$!
    cd ..
    sleep 2
fi

# Verify FastWhisperAPI is working
if ! curl -s http://localhost:8000/info > /dev/null; then
    echo "⚠️ FastWhisperAPI didn't start correctly. Running fix script..."
    python fix_fastwhisper_api.py
fi

# Run HowdyTTS
echo "Starting HowdyTTS..."
python run_voice_assistant.py

cleanup() {
    echo "Cleaning up background processes..."
    [ ! -z "$OLLAMA_PID" ] && kill $OLLAMA_PID
    [ ! -z "$FASTWHISPER_PID" ] && kill $FASTWHISPER_PID
}
trap cleanup EXIT