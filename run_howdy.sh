#!/bin/bash

# HowdyTTS Launcher Script (using Conda)
# Created by mac_setup.py

# Activate Conda environment
conda activate howdy310

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
    uvicorn main:app --reload &
    FASTWHISPER_PID=$!
    cd ..
    sleep 2
fi

# Run HowdyTTS
python run_voice_assistant.py

cleanup() {
    echo "Cleaning up background processes..."
    [ ! -z "$OLLAMA_PID" ] && kill $OLLAMA_PID
    [ ! -z "$FASTWHISPER_PID" ] && kill $FASTWHISPER_PID
}
trap cleanup EXIT
