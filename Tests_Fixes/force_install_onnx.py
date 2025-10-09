#!/usr/bin/env python3
"""
Direct installation of required ONNX Runtime for Apple Silicon
"""

import os
import sys
import subprocess
import time
import platform

def run_pip_command(cmd):
    """Run a pip command and print output"""
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Get the return code
    return_code = process.poll()
    
    # Print any errors
    if return_code != 0:
        errors = process.stderr.read()
        print(f"Error: {errors}")
    
    return return_code == 0

def main():
    print("\n===== Direct ONNX Runtime Installation =====\n")
    
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("⚠️ This tool is specifically for Apple Silicon Macs")
        return
    
    print("✅ Running on Apple Silicon Mac")
    
    # Uninstall any existing ONNX Runtime packages
    print("\n1. Removing ALL existing ONNX Runtime packages...")
    run_pip_command(f"{sys.executable} -m pip uninstall -y onnxruntime")
    run_pip_command(f"{sys.executable} -m pip uninstall -y onnxruntime-silicon")
    
    # Also uninstall kokoro-onnx to avoid dependency issues
    print("\n2. Removing kokoro-onnx temporarily...")
    run_pip_command(f"{sys.executable} -m pip uninstall -y kokoro-onnx")
    
    # Clean pip cache to ensure clean installation
    print("\n3. Cleaning pip cache...")
    run_pip_command(f"{sys.executable} -m pip cache purge")
    
    # Install onnxruntime-silicon with specific version
    print("\n4. Installing onnxruntime-silicon v1.16.3...")
    success = run_pip_command(f"{sys.executable} -m pip install --no-cache-dir onnxruntime-silicon==1.16.3")
    
    if not success:
        print("\n❌ Failed to install onnxruntime-silicon. Trying alternate approach...")
        print("Attempting to install onnxruntime instead...")
        success = run_pip_command(f"{sys.executable} -m pip install --no-cache-dir onnxruntime==1.17.0")
        
        if not success:
            print("\n❌ Failed to install any version of ONNX Runtime.")
            print("Please try manually with: pip install onnxruntime==1.17.0")
            return
    
    # Reinstall kokoro-onnx
    print("\n5. Reinstalling kokoro-onnx...")
    run_pip_command(f"{sys.executable} -m pip install --no-cache-dir kokoro-onnx==0.4.8")
    
    # Verify installation
    print("\n6. Verifying installation...")
    
    # Create a test script to verify the installation
    test_script = """
import sys
print(f"Python version: {sys.version}")

# Try to import onnxruntime
try:
    import onnxruntime as ort
    print(f"Successfully imported onnxruntime")
    print(f"Version: {ort.__version__ if hasattr(ort, '__version__') else 'Unknown'}")
    print(f"Available attributes: {dir(ort)[:10]}...")
    
    if hasattr(ort, 'InferenceSession'):
        print("✅ ort.InferenceSession is available")
    else:
        print("❌ ort.InferenceSession is NOT available")
    
    # Check available providers if the function exists
    if hasattr(ort, 'get_available_providers'):
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
    else:
        print("⚠️ get_available_providers not available")
except Exception as e:
    print(f"❌ Error importing onnxruntime: {e}")

# Try to import onnxruntime_silicon
try:
    import onnxruntime_silicon as ort_si
    print(f"Successfully imported onnxruntime_silicon")
    # Additional checks can go here
except ImportError:
    print("⚠️ onnxruntime_silicon not importable")
except Exception as e:
    print(f"❌ Error with onnxruntime_silicon: {e}")

# Try to import kokoro_onnx
try:
    import kokoro_onnx
    print(f"Successfully imported kokoro_onnx")
    
    # Check if Kokoro class is correctly defined
    if hasattr(kokoro_onnx, 'Kokoro'):
        print("✅ kokoro_onnx.Kokoro class is available")
    else:
        print("❌ kokoro_onnx.Kokoro class is NOT available")
except Exception as e:
    print(f"❌ Error importing kokoro_onnx: {e}")
"""
    
    # Write the test script to a file
    with open("test_onnx_import.py", "w") as f:
        f.write(test_script)
    
    # Run the test script
    print("\nRunning verification script...")
    subprocess.run([sys.executable, "test_onnx_import.py"])
    
    # Clean up
    os.remove("test_onnx_import.py")
    
    print("\n===== ONNX Runtime Installation Complete =====")
    print("\nNext step: Try running HowdyVox with:")
    print("./run_howdy.sh")

if __name__ == "__main__":
    main()