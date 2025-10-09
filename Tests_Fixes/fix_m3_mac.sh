#!/bin/bash

# Fix script for HowdyVox on M3 Mac Studio
# This script applies all necessary fixes to make HowdyVox work on M3 Apple Silicon Mac

echo "===== HowdyVox M3 Mac Fix Script ====="
echo "This script will fix issues with running HowdyVox on M3 Mac Studio"

# Ensure we're in a conda environment
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "❌ Not running in a conda environment"
    echo "Please activate your conda environment first:"
    echo "conda activate howdy310"
    exit 1
fi

echo "✅ Running in conda environment: $CONDA_PREFIX"

# Step 1: Fix ONNX Runtime issues
echo -e "\n===== Step 1: Fixing ONNX Runtime ====="
echo "This will repair the ONNX Runtime installation for Apple Silicon"

# Force reinstall of onnxruntime with correct version
echo "Uninstalling any existing onnxruntime packages..."
pip uninstall -y onnxruntime onnxruntime-silicon

echo "Installing standard onnxruntime..."
pip install --no-cache-dir onnxruntime==1.17.0

echo "Reinstalling kokoro-onnx..."
pip uninstall -y kokoro-onnx
pip install --no-cache-dir kokoro-onnx==0.4.8

# Step 2: Apply the fixed kokoro_manager.py
echo -e "\n===== Step 2: Applying fixed Kokoro manager ====="
if [ -f "voice_assistant/kokoro_manager_fixed.py" ]; then
    echo "Backing up original kokoro_manager.py..."
    cp voice_assistant/kokoro_manager.py voice_assistant/kokoro_manager.py.backup
    
    echo "Applying fixed version..."
    cp voice_assistant/kokoro_manager_fixed.py voice_assistant/kokoro_manager.py
    echo "✅ Applied fixed kokoro_manager.py"
else
    echo "❌ Fixed kokoro_manager file not found"
    exit 1
fi

# Step 3: Fix FastWhisperAPI if needed
echo -e "\n===== Step 3: Checking FastWhisperAPI ====="
if ! curl -s http://localhost:8000/info > /dev/null; then
    echo "FastWhisperAPI is not running, starting it now..."
    
    # Change to FastWhisperAPI directory
    cd FastWhisperAPI
    
    # Install dependencies
    echo "Installing FastWhisperAPI dependencies..."
    pip install -r requirements.txt
    
    # Start the server in background
    echo "Starting FastWhisperAPI server..."
    python -m uvicorn main:app --reload --port 8000 &
    FASTWHISPER_PID=$!
    
    # Return to original directory
    cd ..
    
    # Wait for server to start
    echo "Waiting for FastWhisperAPI to start..."
    for i in {1..10}; do
        sleep 1
        if curl -s http://localhost:8000/info > /dev/null; then
            echo "✅ FastWhisperAPI is now running"
            break
        fi
        echo "." 
    done
else
    echo "✅ FastWhisperAPI is already running"
fi

# Step 4: Prepare the run script
echo -e "\n===== Step 4: Updating run script ====="
if [ -f "run_howdy_conda.sh" ]; then
    echo "Making run_howdy_conda.sh executable..."
    chmod +x run_howdy_conda.sh
    
    echo "Creating new run_howdy.sh from conda version..."
    if [ -f "run_howdy.sh" ]; then
        cp run_howdy.sh run_howdy.sh.backup
    fi
    
    cp run_howdy_conda.sh run_howdy.sh
    chmod +x run_howdy.sh
    
    echo "✅ Updated run scripts"
else
    echo "❌ Conda run script not found"
fi

# Step 5: Create symbolic link for onnxruntime_silicon to ensure imports work
echo -e "\n===== Step 5: Creating onnxruntime_silicon compatibility ====="
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

if [ -d "$SITE_PACKAGES" ]; then
    echo "Site packages directory: $SITE_PACKAGES"
    
    # Create symbolic link from onnxruntime to onnxruntime_silicon
    if [ -d "$SITE_PACKAGES/onnxruntime" ] && [ ! -d "$SITE_PACKAGES/onnxruntime_silicon" ]; then
        echo "Creating symbolic link for onnxruntime_silicon..."
        ln -s "$SITE_PACKAGES/onnxruntime" "$SITE_PACKAGES/onnxruntime_silicon"
        echo "✅ Created symbolic link"
    elif [ -d "$SITE_PACKAGES/onnxruntime_silicon" ]; then
        echo "✅ onnxruntime_silicon already exists"
    else
        echo "❌ onnxruntime directory not found"
    fi
else
    echo "❌ Could not determine site-packages directory"
fi

# Step 6: Test the kokoro integration
echo -e "\n===== Step 6: Testing Kokoro TTS ====="
echo "Creating a simple test script..."

cat > test_kokoro_simple.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test for Kokoro TTS after fixes
"""
import os
import sys
import platform

def main():
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Test ONNX Runtime import
    try:
        import onnxruntime as ort
        print(f"✅ Successfully imported onnxruntime")
        
        if hasattr(ort, 'InferenceSession'):
            print(f"✅ InferenceSession is available")
        else:
            print(f"❌ InferenceSession is NOT available")
    except Exception as e:
        print(f"❌ Error importing onnxruntime: {e}")
        return
    
    # Test kokoro_onnx
    try:
        from kokoro_onnx import Kokoro
        print(f"✅ Successfully imported Kokoro")
    except Exception as e:
        print(f"❌ Error importing Kokoro: {e}")
        return
    
    # Test initialization
    try:
        model_path = os.path.join("models", "kokoro-v1.0.onnx")
        voices_path = os.path.join("models", "voices-v1.0.bin")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
            
        if not os.path.exists(voices_path):
            print(f"❌ Voices file not found: {voices_path}")
            return
            
        print("Initializing Kokoro (this may take a moment)...")
        kokoro = Kokoro(model_path, voices_path)
        print("✅ Successfully initialized Kokoro TTS")
        
        # Try synthesizing a short phrase
        voice = "am_michael"
        text = "Howdy partner! This is a test of the Kokoro TTS system."
        print(f"Synthesizing speech with voice '{voice}'...")
        audio = kokoro.predict(text, voice=voice)
        print(f"✅ Successfully synthesized {len(audio)} audio samples")
        
        # Save the audio
        import soundfile as sf
        output_path = "fix_test_result.wav"
        sf.write(output_path, audio, 24000)
        print(f"✅ Audio saved to {output_path}")
    except Exception as e:
        print(f"❌ Error testing Kokoro: {e}")
        return
    
    print("\n🎉 All tests passed! Kokoro TTS is working correctly.")

if __name__ == "__main__":
    main()
EOF

echo "Running test script..."
python test_kokoro_simple.py

# Summary
echo -e "\n===== Fix script complete ====="
echo "If the test was successful, you can now run HowdyVox with:"
echo "./run_howdy.sh"
echo ""
echo "If there were still errors, please run the standalone fix script:"
echo "python force_install_onnx.py"
echo ""
echo "Happy howdy-ing! 🤠"