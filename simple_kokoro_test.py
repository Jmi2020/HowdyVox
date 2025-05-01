#!/usr/bin/env python3
"""
Simple test script for Kokoro TTS on Apple Silicon Mac
"""

import os
import sys
import platform
import importlib.util

def check_module(module_name):
    """Check if a module is installed and return its version if available"""
    is_installed = importlib.util.find_spec(module_name) is not None
    version = None
    
    if is_installed:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, '__version__'):
                version = module.__version__
        except:
            pass
    
    return is_installed, version

def main():
    print("\n===== Basic System Information =====")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    print("\n===== Checking Critical Dependencies =====")
    modules_to_check = [
        "onnxruntime_silicon", 
        "onnxruntime", 
        "kokoro_onnx", 
        "numpy", 
        "soundfile", 
        "pygame"
    ]
    
    for module in modules_to_check:
        installed, version = check_module(module)
        if installed:
            print(f"✅ {module}: Installed{f' (version: {version})' if version else ''}")
        else:
            print(f"❌ {module}: Not installed")
    
    # Check ONNX Runtime providers
    print("\n===== Checking ONNX Runtime Providers =====")
    try:
        # First try silicon version
        try:
            import onnxruntime_silicon as ort
            print("Using onnxruntime-silicon")
        except ImportError:
            import onnxruntime as ort
            print("Using standard onnxruntime")
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if "CoreMLExecutionProvider" in providers:
            print("✅ CoreML provider is available (best for Apple Silicon)")
        else:
            print("⚠️ CoreML provider not available")
    except Exception as e:
        print(f"❌ Error checking ONNX providers: {e}")
    
    # Check model files
    print("\n===== Checking Model Files =====")
    model_path = os.path.join("models", "kokoro-v1.0.onnx")
    voices_path = os.path.join("models", "voices-v1.0.bin")
    
    if os.path.exists(model_path):
        print(f"✅ Kokoro model file found: {model_path}")
    else:
        print(f"❌ Kokoro model file missing: {model_path}")
    
    if os.path.exists(voices_path):
        print(f"✅ Voices file found: {voices_path}")
    else:
        print(f"❌ Voices file missing: {voices_path}")
    
    # Try minimal Kokoro test if modules are available
    if check_module("kokoro_onnx")[0]:
        print("\n===== Testing Kokoro Text-to-Speech =====")
        try:
            from kokoro_onnx import Kokoro
            
            # Check if model files exist before trying to load
            if os.path.exists(model_path) and os.path.exists(voices_path):
                print("Initializing Kokoro model (may take a few seconds)...")
                kokoro = Kokoro(model_path, voices_path)
                print("✅ Kokoro model initialized successfully")
                
                # Try synthesizing a simple phrase
                test_text = "This is a simple test."
                print(f"Synthesizing: '{test_text}'")
                audio = kokoro.predict(test_text, voice="am_michael")
                print(f"✅ Audio synthesized successfully (length: {len(audio)} samples)")
                
                # Save the audio
                import soundfile as sf
                output_path = "simple_test.wav"
                sf.write(output_path, audio, 24000)
                print(f"✅ Audio saved to {output_path}")
            else:
                print("⚠️ Skipping Kokoro test due to missing model files")
        except Exception as e:
            print(f"❌ Kokoro test failed: {e}")
    else:
        print("\n⚠️ Skipping Kokoro test since kokoro_onnx module is not available")
    
    print("\n===== Test Complete =====")

if __name__ == "__main__":
    main()