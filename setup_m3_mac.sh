#!/bin/bash
# Automated Setup Script for HowdyVox on M3 Mac (Apple Silicon)
# This script handles all M3-specific installation steps automatically

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   HowdyVox M3 Mac Automated Setup${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}✗ This script is for Apple Silicon (M3/M2/M1) Macs only${NC}"
    echo -e "${YELLOW}  Your system: $(uname -m)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Detected Apple Silicon ($(uname -m))${NC}"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}✗ Homebrew not found${NC}"
    echo -e "${YELLOW}  Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo -e "${GREEN}✓ Homebrew installed${NC}"
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Conda not found${NC}"
    echo -e "${YELLOW}  Please install Anaconda or Miniconda first:${NC}"
    echo -e "${YELLOW}  https://www.anaconda.com/download${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Conda found: $(which conda)${NC}"
echo ""

# Install system dependencies via Homebrew
echo -e "${BLUE}[1/7] Installing system libraries...${NC}"

if brew list portaudio &> /dev/null; then
    echo -e "${GREEN}✓ PortAudio already installed${NC}"
else
    echo -e "${YELLOW}  Installing PortAudio...${NC}"
    brew install portaudio
    echo -e "${GREEN}✓ PortAudio installed${NC}"
fi

if brew list opus &> /dev/null; then
    echo -e "${GREEN}✓ Opus already installed${NC}"
else
    echo -e "${YELLOW}  Installing Opus...${NC}"
    brew install opus
    echo -e "${GREEN}✓ Opus installed${NC}"
fi

echo ""

# Create or activate conda environment
echo -e "${BLUE}[2/7] Setting up conda environment 'howdy310'...${NC}"

if conda env list | grep -q "^howdy310 "; then
    echo -e "${YELLOW}  Environment 'howdy310' already exists${NC}"
    read -p "  Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n howdy310 -y
        conda create -n howdy310 python=3.10 -y
        echo -e "${GREEN}✓ Environment recreated${NC}"
    else
        echo -e "${YELLOW}  Using existing environment${NC}"
    fi
else
    conda create -n howdy310 python=3.10 -y
    echo -e "${GREEN}✓ Environment created${NC}"
fi

echo ""

# Activate environment
echo -e "${BLUE}[3/7] Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate howdy310
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Install PyAudio with proper compilation flags
echo -e "${BLUE}[4/7] Installing PyAudio with M3-specific compilation flags...${NC}"

PORTAUDIO_PREFIX=$(brew --prefix portaudio)

# Check if PyAudio is already installed
if python -c "import pyaudio" 2>/dev/null; then
    echo -e "${YELLOW}  PyAudio already installed, reinstalling with proper flags...${NC}"
    pip uninstall -y pyaudio 2>/dev/null || true
fi

pip install --no-cache-dir \
    --global-option=build_ext \
    --global-option="-I${PORTAUDIO_PREFIX}/include" \
    --global-option="-L${PORTAUDIO_PREFIX}/lib" \
    pyaudio 2>&1 | grep -v "WARNING:" | grep -v "DEPRECATION:" || true

if python -c "import pyaudio" 2>/dev/null; then
    echo -e "${GREEN}✓ PyAudio installed successfully${NC}"
else
    echo -e "${RED}✗ PyAudio installation failed${NC}"
    exit 1
fi

echo ""

# Install all other requirements
echo -e "${BLUE}[5/7] Installing Python dependencies...${NC}"
echo -e "${YELLOW}  This may take several minutes...${NC}"

pip install -r requirements.txt 2>&1 | grep -E "(Requirement already satisfied|Successfully installed|Collecting)" | tail -20

echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Configure environment variables
echo -e "${BLUE}[6/7] Configuring environment...${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    cp example.env .env
    echo -e "${YELLOW}  Created .env file from example${NC}"
    echo -e "${YELLOW}  Please edit .env and add your PORCUPINE_ACCESS_KEY${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create/update conda environment activation script
CONDA_ENV_PATH=$(conda info --envs | grep "^howdy310 " | awk '{print $NF}')
ACTIVATE_SCRIPT="${CONDA_ENV_PATH}/etc/conda/activate.d/env_vars.sh"

mkdir -p "${CONDA_ENV_PATH}/etc/conda/activate.d"

cat > "${ACTIVATE_SCRIPT}" <<'ACTIVATE_EOF'
#!/bin/sh
# Auto-set library paths for HowdyVox on M3 Mac

if [[ $(uname -m) == "arm64" ]]; then
    # Set Opus library path for wireless audio support
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/opus/lib:${DYLD_LIBRARY_PATH}"

    # Suppress PyAudio/PortAudio warnings
    export PYAUDIO_SUPPRESS_WARNINGS=1
fi
ACTIVATE_EOF

chmod +x "${ACTIVATE_SCRIPT}"

echo -e "${GREEN}✓ Created conda activation script at:${NC}"
echo -e "${GREEN}  ${ACTIVATE_SCRIPT}${NC}"
echo ""

# Verify installation
echo -e "${BLUE}[7/7] Verifying installation...${NC}"

ERRORS=0

# Test PyAudio
if python -c "import pyaudio" 2>/dev/null; then
    echo -e "${GREEN}✓ PyAudio OK${NC}"
else
    echo -e "${RED}✗ PyAudio import failed${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Test opuslib
if python -c "import opuslib" 2>/dev/null; then
    echo -e "${GREEN}✓ Opuslib OK${NC}"
else
    echo -e "${RED}✗ Opuslib import failed${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Test ONNX Runtime
if python -c "import onnxruntime" 2>/dev/null; then
    echo -e "${GREEN}✓ ONNX Runtime OK${NC}"
else
    echo -e "${RED}✗ ONNX Runtime import failed${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Test Kokoro
if python -c "from voice_assistant.kokoro_manager import KokoroManager" 2>/dev/null; then
    echo -e "${GREEN}✓ Kokoro TTS OK${NC}"
else
    echo -e "${RED}✗ Kokoro TTS import failed${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Test torch
if python -c "import torch; import torchaudio" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch/TorchAudio OK${NC}"
else
    echo -e "${RED}✗ PyTorch import failed${NC}"
    ERRORS=$((ERRORS + 1))
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}   ✓ Setup completed successfully!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Edit .env file with your Porcupine access key"
    echo -e "  2. Run: ${GREEN}python quick_setup.py${NC}"
    echo -e "  3. Start HowdyVox: ${GREEN}./launch_howdy_m3.sh${NC}"
    echo ""
    echo -e "${YELLOW}Note: The conda environment will now automatically${NC}"
    echo -e "${YELLOW}      set the required library paths when activated${NC}"
else
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}   Setup completed with ${ERRORS} error(s)${NC}"
    echo -e "${RED}================================================${NC}"
    echo ""
    echo -e "${YELLOW}Please review the errors above and try:${NC}"
    echo -e "  - Rerunning this script"
    echo -e "  - Checking M3_MAC_SETUP.md for troubleshooting"
    exit 1
fi
