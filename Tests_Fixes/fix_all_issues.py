#!/usr/bin/env python3
"""
One-stop script to diagnose and fix all common issues with HowdyTTS on Apple Silicon
"""

import os
import sys
import subprocess
import importlib
import platform
import time
import shutil

def run_command(command, description=None):
    """Run a command and return the result"""
    if description:
        print(f"\n{description}...")
    
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    
    return result.returncode == 0, result.stdout, result.stderr

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def fix_onnx_runtime():
    """Fix ONNX Runtime issues"""
    print("\n===== Fixing ONNX Runtime =====")
    
    # Check if we're on Apple Silicon
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("⚠️ This fix is only for Apple Silicon Macs")
        return False
    
    # Uninstall existing ONNX Runtime packages
    success, _, _ = run_command(
        f"{sys.executable} -m pip uninstall -y onnxruntime onnxruntime-silicon",
        "Removing existing ONNX Runtime packages"
    )
    
    # Install onnxruntime-silicon
    success, _, err = run_command(
        f"{sys.executable} -m pip install onnxruntime-silicon==1.16.3",
        "Installing onnxruntime-silicon 1.16.3"
    )
    
    if not success:
        print(f"❌ Failed to install onnxruntime-silicon: {err}")
        return False
    
    # Reinstall kokoro-onnx
    success, _, err = run_command(
        f"{sys.executable} -m pip install --force-reinstall kokoro-onnx==0.4.8",
        "Reinstalling kokoro-onnx 0.4.8"
    )
    
    if not success:
        print(f"❌ Failed to reinstall kokoro-onnx: {err}")
        return False
    
    # Clear the module cache
    for key in list(sys.modules.keys()):
        if key.startswith('onnxruntime') or key == 'kokoro_onnx':
            del sys.modules[key]
    
    # Verify installation
    try:
        import onnxruntime_silicon as ort
        print(f"✅ Successfully imported onnxruntime-silicon")
        
        if hasattr(ort, 'get_available_providers'):
            providers = ort.get_available_providers()
            print(f"Available providers: {providers}")
            
            if "CoreMLExecutionProvider" in providers:
                print("✅ CoreML provider is available")
            else:
                print("⚠️ CoreML provider not available")
        else:
            print("⚠️ Cannot determine available providers")
        
        # Try importing kokoro_onnx
        import kokoro_onnx
        print("✅ Successfully imported kokoro_onnx")
        
        return True
    except Exception as e:
        print(f"❌ Error verifying ONNX Runtime installation: {e}")
        return False

def fix_fastwhisper_api():
    """Fix FastWhisperAPI issues"""
    print("\n===== Fixing FastWhisperAPI =====")
    
    api_dir = os.path.join(os.getcwd(), "FastWhisperAPI")
    if not os.path.isdir(api_dir):
        print("❌ FastWhisperAPI directory not found")
        return False
    
    # Install dependencies
    req_file = os.path.join(api_dir, "requirements.txt")
    success, _, err = run_command(
        f"{sys.executable} -m pip install -r {req_file}",
        "Installing FastWhisperAPI dependencies"
    )
    
    if not success:
        print(f"❌ Failed to install FastWhisperAPI dependencies: {err}")
        return False
    
    # Try to restart the service
    if platform.system() == "Windows":
        run_command("taskkill /f /im uvicorn.exe", "Stopping any running uvicorn processes")
    else:
        run_command(
            "ps aux | grep 'uvicorn main:app' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true",
            "Stopping any running uvicorn processes"
        )
    
    # Start the service
    os.chdir(api_dir)
    if platform.system() == "Windows":
        subprocess.Popen(
            ["start", "cmd", "/c", "uvicorn", "main:app", "--reload", "--port", "8000"],
            shell=True
        )
    else:
        subprocess.Popen(
            ["uvicorn", "main:app", "--reload", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    # Return to original directory
    os.chdir(os.path.dirname(api_dir))
    
    # Check if service started
    print("Waiting for FastWhisperAPI to start...")
    for _ in range(10):
        time.sleep(1)
        try:
            import requests
            response = requests.get("http://localhost:8000/info", timeout=2)
            if response.status_code == 200:
                print("✅ FastWhisperAPI started successfully")
                return True
        except:
            pass
        print(".", end="", flush=True)
    
    print("\n❌ Failed to start FastWhisperAPI")
    return False

def check_model_files():
    """Check if model files exist"""
    print("\n===== Checking Model Files =====")
    
    model_dir = os.path.join(os.getcwd(), "models")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        print(f"Created models directory at {model_dir}")
    
    # Check for required files
    required_files = [
        ("kokoro-v1.0.onnx", "Kokoro ONNX model"),
        ("voices-v1.0.bin", "Kokoro voices file"),
        ("Hey-howdy_en_mac_v3_0_0.ppn", "Porcupine wake word model")
    ]
    
    missing_files = []
    for filename, description in required_files:
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            print(f"✅ {description} found: {file_path}")
        else:
            print(f"❌ {description} missing: {file_path}")
            missing_files.append((filename, description))
    
    # Check voices directory
    voices_dir = os.path.join(model_dir, "voices")
    if not os.path.isdir(voices_dir):
        os.makedirs(voices_dir)
        print(f"Created voices directory at {voices_dir}")
    
    voice_files = os.listdir(voices_dir) if os.path.isdir(voices_dir) else []
    voice_files = [f for f in voice_files if f.endswith('.bin')]
    
    if voice_files:
        print(f"✅ Found {len(voice_files)} voice files")
        if "am_michael.bin" in voice_files:
            print("✅ Default cowboy voice (am_michael.bin) is available")
        else:
            print("⚠️ Default cowboy voice (am_michael.bin) not found")
            missing_files.append(("voices/am_michael.bin", "Default cowboy voice"))
    else:
        print("❌ No voice files found in voices directory")
        missing_files.append(("voices/am_michael.bin", "Default cowboy voice"))
    
    return missing_files

def fix_run_script():
    """Fix the run script"""
    print("\n===== Fixing Run Script =====")
    
    # Check if the conda run script exists
    conda_script = os.path.join(os.getcwd(), "run_howdy_conda.sh")
    if not os.path.exists(conda_script):
        print("❌ Conda run script not found")
        return False
    
    # Make it executable
    os.chmod(conda_script, 0o755)
    print(f"✅ Made conda run script executable")
    
    # Create a symlink or copy to run_howdy.sh
    run_script = os.path.join(os.getcwd(), "run_howdy.sh")
    if os.path.exists(run_script):
        os.rename(run_script, f"{run_script}.backup")
        print(f"✅ Backed up original run script to {run_script}.backup")
    
    shutil.copy2(conda_script, run_script)
    os.chmod(run_script, 0o755)
    print(f"✅ Created new run script at {run_script}")
    
    return True

def fix_environment_variables():
    """Fix environment variables"""
    print("\n===== Fixing Environment Variables =====")
    
    # Check if .env file exists
    env_file = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_file):
        # Create a new .env file
        with open(env_file, 'w') as f:
            f.write("# HowdyTTS Environment Variables\n")
            f.write("LOCAL_MODEL_PATH=models\n")
            f.write("# Add your Porcupine access key below\n")
            f.write("PORCUPINE_ACCESS_KEY=\n")
        
        print(f"✅ Created new .env file at {env_file}")
        print("⚠️ Please edit the .env file and add your Porcupine access key")
    else:
        # Read the .env file and check for required variables
        with open(env_file, 'r') as f:
            content = f.read()
        
        needs_update = False
        new_content = content
        
        if "LOCAL_MODEL_PATH" not in content:
            new_content += "\nLOCAL_MODEL_PATH=models\n"
            needs_update = True
        
        if "PORCUPINE_ACCESS_KEY" not in content:
            new_content += "\n# Add your Porcupine access key below\nPORCUPINE_ACCESS_KEY=\n"
            needs_update = True
        
        if needs_update:
            with open(env_file, 'w') as f:
                f.write(new_content)
            print(f"✅ Updated .env file with required variables")
            print("⚠️ Please edit the .env file and add your Porcupine access key if not already set")
        else:
            print(f"✅ .env file already contains required variables")
    
    return True

def main():
    print("\n===== HowdyTTS Comprehensive Fix Tool =====")
    print("This tool will diagnose and fix common issues with HowdyTTS on Apple Silicon Macs")
    
    # Check if we're running on Apple Silicon
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("⚠️ This tool is designed for Apple Silicon Macs")
        print(f"Your system: {platform.system()} {platform.machine()}")
    
    print(f"\nPython version: {sys.version}")
    
    # Fix ONNX Runtime
    onnx_fixed = fix_onnx_runtime()
    
    # Check model files
    missing_files = check_model_files()
    
    # Fix environment variables
    env_fixed = fix_environment_variables()
    
    # Fix FastWhisperAPI
    fastwhisper_fixed = fix_fastwhisper_api()
    
    # Fix run script
    script_fixed = fix_run_script()
    
    # Summary
    print("\n===== Fix Summary =====")
    
    if onnx_fixed:
        print("✅ ONNX Runtime fixed and working correctly")
    else:
        print("❌ Issues with ONNX Runtime remain")
    
    if not missing_files:
        print("✅ All model files are present")
    else:
        print("❌ Missing model files:")
        for filename, description in missing_files:
            print(f"  - {description} ({filename})")
    
    if env_fixed:
        print("✅ Environment variables configured")
    else:
        print("❌ Issues with environment variables remain")
    
    if fastwhisper_fixed:
        print("✅ FastWhisperAPI fixed and running")
    else:
        print("❌ Issues with FastWhisperAPI remain")
    
    if script_fixed:
        print("✅ Run script updated for conda environment")
    else:
        print("❌ Issues with run script remain")
    
    # Next steps
    print("\n===== Next Steps =====")
    
    if missing_files:
        print("1. You need to obtain the missing model files")
        print("   - For Kokoro model files: Check the project documentation")
        print("   - For Porcupine wake word: Register at https://console.picovoice.ai/ and download the model")
    
    if not onnx_fixed or not fastwhisper_fixed:
        print("2. Some components could not be automatically fixed")
        print("   - Try running the individual fix scripts manually")
    
    print("\nTo run HowdyTTS:")
    print("1. Make sure your conda environment is activated: conda activate howdy310")
    print("2. Run the updated script: ./run_howdy.sh")
    
    print("\n===== Fix Complete =====")

if __name__ == "__main__":
    main()