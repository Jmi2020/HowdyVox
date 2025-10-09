#!/usr/bin/env python3
"""
Fix ONNX Runtime configuration for Apple Silicon Macs
"""

import os
import sys
import subprocess
import platform
import importlib

def main():
    print("\n===== Fixing ONNX Runtime for Apple Silicon =====\n")
    
    # Check if running on Apple Silicon
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("❌ This script is only for Apple Silicon Macs")
        return
    
    print("✅ Running on Apple Silicon Mac")
    
    # Uninstall any existing onnxruntime packages
    print("\n1. Removing existing ONNX Runtime packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], 
                  stdout=subprocess.PIPE)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-silicon"], 
                  stdout=subprocess.PIPE)
    
    # Install onnxruntime-silicon with the correct version
    print("\n2. Installing onnxruntime-silicon v1.16.3...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "onnxruntime-silicon==1.16.3"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error installing onnxruntime-silicon: {result.stderr}")
        return
    
    print("✅ Installed onnxruntime-silicon 1.16.3")
    
    # Make sure kokoro-onnx is also properly installed
    print("\n3. Reinstalling kokoro-onnx...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "kokoro-onnx==0.4.8"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error reinstalling kokoro-onnx: {result.stderr}")
        return
    
    print("✅ Reinstalled kokoro-onnx 0.4.8")
    
    # Verify installation
    print("\n4. Verifying installation...")
    
    try:
        # Clear the module cache to ensure we load the newly installed packages
        for key in list(sys.modules.keys()):
            if key.startswith('onnxruntime') or key == 'kokoro_onnx':
                del sys.modules[key]
        
        # Import onnxruntime_silicon
        import onnxruntime_silicon as ort
        print(f"✅ Successfully imported onnxruntime-silicon")
        print(f"Version: {ort.__version__ if hasattr(ort, '__version__') else 'Unknown'}")
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if "CoreMLExecutionProvider" in providers:
            print("✅ CoreML provider is available (best for Apple Silicon)")
        else:
            print("⚠️ CoreML provider not available")
        
        # Try importing kokoro
        import kokoro_onnx
        print("✅ Successfully imported kokoro_onnx")
        
        # Set environment variable to prefer CoreML
        os.environ["ORT_COREML_ALLOWED"] = "1"
        
        # Test creating a session (the part that's failing)
        try:
            from kokoro_onnx import Kokoro
            model_path = os.path.join("models", "kokoro-v1.0.onnx")
            voices_path = os.path.join("models", "voices-v1.0.bin")
            
            if os.path.exists(model_path) and os.path.exists(voices_path):
                print("\n5. Testing Kokoro initialization...")
                kokoro = Kokoro(model_path, voices_path)
                print("✅ Successfully initialized Kokoro!")
                
                # Test synthesis
                print("\n6. Testing speech synthesis...")
                test_text = "Howdy partner! This is a test of the fixed Kokoro TTS system."
                audio = kokoro.predict(test_text, voice="am_michael")
                print(f"✅ Successfully synthesized audio ({len(audio)} samples)")
                
                # Save the audio
                import soundfile as sf
                output_path = "fix_test.wav"
                sf.write(output_path, audio, 24000)
                print(f"✅ Audio saved to {output_path}")
            else:
                print(f"⚠️ Model files not found, skipping Kokoro test")
        except Exception as e:
            print(f"❌ Error initializing Kokoro: {e}")
            print("This is a more specific error that needs to be addressed")
    
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return
    
    print("\n===== ONNX Runtime Fix Complete =====")
    print("\nIf all checks passed, your system is now properly configured for HowdyVox.")
    print("You can now run the voice assistant:")
    print("  python run_voice_assistant.py")

if __name__ == "__main__":
    main()