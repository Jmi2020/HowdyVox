#!/bin/bash

# This script activates the Python environment for HowdyTTS
# It creates a new virtualenv if one doesn't exist

ENV_NAME="howdy_env"
ENV_DIR="$HOME/.howdy_env"

# Check if virtual environment exists
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating new Python virtual environment at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
fi

# Activate the virtual environment
echo "Activating Python environment..."
source "$ENV_DIR/bin/activate"

# Install requirements if needed
if [ ! -f "$ENV_DIR/.installed" ]; then
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch "$ENV_DIR/.installed"
else
    echo "Environment already set up."
fi

echo "Environment activated. You can now run HowdyTTS."