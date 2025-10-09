#!/usr/bin/env python3
"""
Check if the Python environment has all required dependencies for HowdyVox
"""

import sys
import subprocess
import importlib.util
import pkg_resources
import platform

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def get_package_version(package_name):
    """Get the version of an installed package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def main():
    print("\n===== HowdyVox Environment Check =====\n")
    
    # Python version check
    py_version = sys.version.split()[0]
    print(f"Python version: {py_version}")
    
    py_major, py_minor, _ = py_version.split(".")
    if int(py_major) != 3 or int(py_minor) != 10:
        print(f"⚠️  Warning: HowdyVox is designed for Python 3.10, you're using {py_major}.{py_minor}")
    else:
        print("✅ Python 3.10 detected (correct version)")
    
    # System information
    print(f"\nSystem: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        print("✅ Apple Silicon Mac detected")
    else:
        print("⚠️ Not running on Apple Silicon Mac")
    
    # Check critical packages
    print("\n===== Checking Critical Dependencies =====")
    
    # Required packages and versions
    required_packages = {
        "python-dotenv": None,
        "PyAudio": "0.2.14",  # Specified in README
        "pvporcupine": None,
        "onnxruntime": None,
        "onnxruntime-silicon": None,
        "kokoro-onnx": None,
        "numpy": None,
        "soundfile": None,
        "pygame": None,
        "fastapi": None,
        "uvicorn": None,
        "faster_whisper": None,
        "ollama": None
    }
    
    missing_packages = []
    wrong_version_packages = []
    
    for package, required_version in required_packages.items():
        # Handle special case for package names with dashes
        import_name = package.replace("-", "_")
        version = get_package_version(package)
        
        if version:
            if required_version and version != required_version:
                print(f"⚠️ {package}: Installed (version {version}, but {required_version} recommended)")
                wrong_version_packages.append((package, version, required_version))
            else:
                print(f"✅ {package}: Installed (version {version})")
        else:
            # Special check for onnxruntime variants
            if package in ["onnxruntime", "onnxruntime-silicon"] and check_import(import_name):
                print(f"✅ {package}: Installed (imported as {import_name})")
            else:
                print(f"❌ {package}: Not installed or not found")
                missing_packages.append(package)
    
    # ONNX Runtime check
    print("\n===== Checking ONNX Runtime Configuration =====")
    ort_silicon_available = check_import("onnxruntime_silicon")
    ort_standard_available = check_import("onnxruntime")
    
    if ort_silicon_available:
        import onnxruntime_silicon as ort
        print("✅ Using onnxruntime-silicon (optimized for Apple Silicon)")
        if hasattr(ort, "get_available_providers"):
            providers = ort.get_available_providers()
            print(f"Available providers: {providers}")
            
            if "CoreMLExecutionProvider" in providers:
                print("✅ CoreML provider available (best for Apple Silicon)")
            else:
                print("⚠️ CoreML provider not available")
        else:
            print("⚠️ Cannot determine available providers")
    elif ort_standard_available:
        import onnxruntime as ort
        print("⚠️ Using standard onnxruntime (not optimized for Apple Silicon)")
        if hasattr(ort, "get_available_providers"):
            providers = ort.get_available_providers()
            print(f"Available providers: {providers}")
        else:
            print("⚠️ Cannot determine available providers")
    else:
        print("❌ No version of ONNX Runtime is installed")
    
    # Kokoro check
    print("\n===== Checking Kokoro TTS Availability =====")
    if check_import("kokoro_onnx"):
        print("✅ kokoro_onnx module is available")
    else:
        print("❌ kokoro_onnx module is not available")
    
    # Model files check
    import os
    print("\n===== Checking Model Files =====")
    model_paths = [
        ("models/kokoro-v1.0.onnx", "Kokoro ONNX model"),
        ("models/voices-v1.0.bin", "Kokoro voices file"),
        ("models/Hey-howdy_en_mac_v3_0_0.ppn", "Porcupine wake word model")
    ]
    
    for path, description in model_paths:
        if os.path.exists(path):
            print(f"✅ {description} found: {path}")
        else:
            print(f"❌ {description} missing: {path}")
    
    # Summary
    print("\n===== Summary =====")
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("\nInstallation commands:")
        for pkg in missing_packages:
            if pkg == "onnxruntime-silicon" and is_apple_silicon:
                print(f"pip install {pkg}==1.16.3")
            else:
                print(f"pip install {pkg}")
    else:
        print("✅ All required packages are installed")
    
    if wrong_version_packages:
        print("\n⚠️ Packages with version mismatches:")
        for pkg, current, required in wrong_version_packages:
            print(f"  - {pkg}: Current={current}, Required={required}")
            print(f"    To fix: pip install {pkg}=={required}")
    
    print("\n===== Environment Check Complete =====")

if __name__ == "__main__":
    main()