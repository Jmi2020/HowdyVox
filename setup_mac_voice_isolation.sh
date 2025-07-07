#!/bin/bash

echo "Setting up macOS Voice Isolation for HowdyTTS"
echo "==========================================="

# Check macOS version
macos_version=$(sw_vers -productVersion)
major_version=$(echo $macos_version | cut -d. -f1)
minor_version=$(echo $macos_version | cut -d. -f2)

if [ $major_version -lt 12 ]; then
    echo "Error: macOS 12.0 or later is required"
    echo "Current version: $macos_version"
    exit 1
fi

echo "✓ macOS $macos_version detected"

# Check for Python 3.8+
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"

# Install PyObjC frameworks
echo "Installing PyObjC frameworks..."
echo "Note: These dependencies are also included in requirements.txt"
pip3 install -U \
    pyobjc-core \
    pyobjc-framework-AVFoundation \
    pyobjc-framework-CoreAudio \
    pyobjc-framework-CoreML

# Check installation
echo ""
echo "Verifying installation..."
python3 -c "
try:
    import AVFoundation
    import CoreAudio
    try:
        import AudioToolbox
    except ImportError:
        from CoreAudio import AudioToolbox
    print('✓ PyObjC frameworks installed successfully')
except ImportError as e:
    print(f'✗ Installation failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup complete! You can now test with:"
    echo "  python3 Tests_Fixes/test_mac_voice_isolation.py"
    echo ""
    echo "Voice isolation will be automatically enabled when you run HowdyTTS"
else
    echo ""
    echo "Setup failed. Please check the error messages above."
    exit 1
fi