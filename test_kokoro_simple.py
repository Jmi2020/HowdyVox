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
