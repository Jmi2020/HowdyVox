#!/usr/bin/env python3
"""
HowdyTTS Mac Migration & Setup Script

This script automates the setup process for HowdyTTS on macOS systems,
handling dependencies, model downloads, and environment configuration.
It's designed specifically for Mac-to-Mac migrations.

Usage:
    python mac_setup.py [--download-models] [--test-components] [--setup-services]
"""

import os
import sys
import platform
import subprocess
import shutil
import argparse
import hashlib
import json
import tempfile
import time
import urllib.request
from pathlib import Path
import logging
from colorama import Fore, Style, init

# Try to import tqdm, but provide a fallback if it's not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm module not found. Using simple progress indicator.")
    # Simple progress indicator fallback class
    class SimpleProgress:
        def __init__(self, **kwargs):
            self.total = kwargs.get('total', 100)
            self.last_percent = -1
            
        def update(self, n):
            if self.total:
                percent = int(n / self.total * 100)
                if percent > self.last_percent:
                    sys.stdout.write(f"\r{percent}% [{percent*'#'}{(100-percent)*'.'}]")
                    sys.stdout.flush()
                    self.last_percent = percent
        
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    tqdm = SimpleProgress

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("howdy_setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Model information with URLs and checksums
MODEL_INFO = {
    "kokoro-v1.0.onnx": {
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "md5": "a80d2de5383bc04a95d4b9b31b1d7147",  # Replace with actual MD5
        "size": 102400000,  # Approximate size in bytes
        "required": True
    },
    "voices-v1.0.bin": {
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "md5": "7f8e4d5cb26be82095eb5a8f4c2e88dc",  # Replace with actual MD5
        "size": 51200000,  # Approximate size in bytes
        "required": True
    },
    "Hey-Howdy_en_mac_v3_0_0.ppn": {
        "url": "https://github.com/PromptEngineer/HowdyTTS/releases/download/v1.0.0/Hey-Howdy_en_mac_v3_0_0.ppn",
        "md5": "8d7e6a234f8a7e6d4c6e7a5e6d5c4b3a",  # Replace with actual MD5
        "size": 5120000,  # Approximate size in bytes
        "required": True
    }
}

class SetupManager:
    """Manages the setup process for HowdyTTS on macOS."""

    def __init__(self):
        """Initialize the setup manager."""
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.project_dir, "models")
        # self.venv_dir = os.path.join(self.project_dir, "venv") # No longer using venv
        self.conda_env_name = "howdy310" # Use this conda environment
        self.temp_dir = os.path.join(self.project_dir, "temp")
        
        # Improved Apple Silicon detection
        self.is_apple_silicon = (platform.processor() == "arm" or 
                               platform.machine() == "arm64" or 
                               "Apple M" in platform.processor())
        self.python_version = platform.python_version()
        self.success_count = 0
        self.error_count = 0
        
        # Create necessary directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "audio"), exist_ok=True)

    def print_system_info(self):
        """Print information about the system."""
        print(f"\n{Fore.CYAN}=== System Information ==={Style.RESET_ALL}")
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print(f"Python: {self.python_version}")
        print(f"Mac Type: {'Apple Silicon' if self.is_apple_silicon else 'Intel'}")
        print(f"Project Directory: {self.project_dir}")
        print(f"Models Directory: {self.models_dir}")

    def check_python_version(self):
        """Check if the Python version is compatible."""
        major, minor, _ = map(int, self.python_version.split('.'))
        
        if major < 3 or (major == 3 and minor < 10):
            print(f"{Fore.RED}Error: Python 3.10 or higher is required. You have {self.python_version}.{Style.RESET_ALL}")
            print(f"Please install a compatible Python version and try again.")
            return False
        
        print(f"{Fore.GREEN}✓ Python version {self.python_version} is compatible.{Style.RESET_ALL}")
        return True

    def check_conda_environment(self):
        """Check if the specified Conda environment exists."""
        print(f"\n{Fore.CYAN}Checking for Conda environment '{self.conda_env_name}'...{Style.RESET_ALL}")
        try:
            # Check if conda is available
            subprocess.run(["conda", "--version"], check=True, capture_output=True)
            
            # List environments and check if ours exists
            result = subprocess.run(["conda", "env", "list", "--json"], check=True, capture_output=True, text=True)
            envs = json.loads(result.stdout)
            
            env_exists = any(Path(env_path).name == self.conda_env_name for env_path in envs.get("envs", []))
            
            if env_exists:
                print(f"{Fore.GREEN}✓ Conda environment '{self.conda_env_name}' found.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Please ensure it's activated before running the rest of the setup:{Style.RESET_ALL}")
                print(f"  conda activate {self.conda_env_name}")
                # Give user time to activate if needed, or assume it's activated
                input("Press Enter to continue once the environment is activated...") 
                return True
            else:
                print(f"{Fore.RED}Error: Conda environment '{self.conda_env_name}' not found.{Style.RESET_ALL}")
                print(f"Please create it using: conda create -n {self.conda_env_name} python=3.10")
                print(f"Then activate it: conda activate {self.conda_env_name}")
                return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Fore.RED}Error: Conda command not found or failed.{Style.RESET_ALL}")
            print("Please ensure Conda is installed and configured correctly.")
            return False
        except json.JSONDecodeError:
            print(f"{Fore.RED}Error parsing Conda environment list.{Style.RESET_ALL}")
            return False

    def setup_virtual_environment(self):
        """Set up a virtual environment."""
        # No longer needed, using Conda environment
        print(f"{Fore.GREEN}✓ Using existing Conda environment '{self.conda_env_name}'.{Style.RESET_ALL}")
        return True

    def install_dependencies(self):
        """Install dependencies using pip from the active Conda environment."""
        print(f"\n{Fore.CYAN}Installing dependencies into Conda env '{self.conda_env_name}'...{Style.RESET_ALL}")
        
        # Use pip directly, assuming the conda env is activated
        pip_command = "pip" 
        
        try:
            # Upgrade pip first
            subprocess.run([pip_command, "install", "--upgrade", "pip"], check=True)
            
            # Install wheel for binary package installations
            subprocess.run([pip_command, "install", "wheel"], check=True)
            
            # Try latest PyAudio first
            print(f"{Fore.YELLOW}Installing latest PyAudio (0.2.14)...{Style.RESET_ALL}")
            subprocess.run([pip_command, "uninstall", "-y", "pyaudio"], check=False) # Use pip_command
            subprocess.run([pip_command, "install", "pyaudio==0.2.14"], check=True) # Use pip_command
            
            # Test if latest PyAudio works
            if not self.test_pyaudio_works():
                print(f"{Fore.YELLOW}Latest PyAudio version didn't work properly. Downgrading to 0.2.12...{Style.RESET_ALL}")
                # Install specific PyAudio version
                subprocess.run([pip_command, "uninstall", "-y", "pyaudio"], check=False) # Use pip_command
                subprocess.run([pip_command, "install", "pyaudio==0.2.12"], check=True) # Use pip_command
                
                # Test again with older version
                if not self.test_pyaudio_works():
                    print(f"{Fore.RED}PyAudio 0.2.12 also failed. You may need to install PortAudio manually:{Style.RESET_ALL}")
                    print(f"  brew install portaudio")
                    # Continue with the installation process anyway
            
            # Install ONNX Runtime (required for Kokoro TTS)
            print(f"{Fore.YELLOW}Installing ONNX Runtime...{Style.RESET_ALL}")
            if self.is_apple_silicon:
                # Apple Silicon requires special version
                subprocess.run([pip_command, "install", "onnxruntime-silicon"], check=True) # Use pip_command
            else:
                # Intel Macs use standard onnxruntime
                subprocess.run([pip_command, "install", "onnxruntime"], check=True) # Use pip_command
            
            # Install other dependencies
            print(f"{Fore.YELLOW}Installing other dependencies...{Style.RESET_ALL}")
            subprocess.run([pip_command, "install", "-r", "requirements.txt"], check=True) # Use pip_command
            
            print(f"{Fore.GREEN}✓ Dependencies installed successfully.{Style.RESET_ALL}")
            self.success_count += 1
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}Error installing dependencies: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Ensure the Conda environment '{self.conda_env_name}' is activated.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}You might also need to install PortAudio first:{Style.RESET_ALL}")
            print(f"  brew install portaudio")
            self.error_count += 1
            return False

    def download_model_file(self, filename, info):
        """
        Download a model file.
        
        Args:
            filename (str): The model filename
            info (dict): Information about the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        target_path = os.path.join(self.models_dir, filename)
        
        # Check if file already exists and verify MD5
        if os.path.exists(target_path):
            print(f"Checking existing {filename}...")
            if self.verify_md5(target_path, info["md5"]):
                print(f"{Fore.GREEN}✓ {filename} already exists and is valid.{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.YELLOW}Existing {filename} is invalid. Redownloading...{Style.RESET_ALL}")
                os.remove(target_path)
        
        # Download the file
        try:
            print(f"Downloading {filename} from {info['url']}...")
            
            # Create a progress bar
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=info["size"]) as progress:
                def report_progress(block_num, block_size, total_size):
                    progress.total = total_size
                    progress.update(block_size)
                
                urllib.request.urlretrieve(info["url"], target_path, reporthook=report_progress)
            
            # Verify the downloaded file
            if self.verify_md5(target_path, info["md5"]):
                print(f"{Fore.GREEN}✓ {filename} downloaded and verified successfully.{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}Downloaded {filename} has invalid checksum.{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}Error downloading {filename}: {e}{Style.RESET_ALL}")
            return False

    def verify_md5(self, file_path, expected_md5):
        """Verify the MD5 hash of a file."""
        if expected_md5.startswith("placeholder_"):
            print(f"{Fore.YELLOW}Note: MD5 verification skipped (placeholder checksum){Style.RESET_ALL}")
            return True
            
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        calculated_md5 = md5_hash.hexdigest()
        if calculated_md5 == expected_md5:
            return True
        else:
            print(f"{Fore.YELLOW}MD5 mismatch: Expected {expected_md5}, got {calculated_md5}{Style.RESET_ALL}")
            return False

    def download_models(self):
        """Download and verify all model files."""
        print(f"\n{Fore.CYAN}=== Downloading Model Files ==={Style.RESET_ALL}")
        
        success = True
        
        for filename, info in MODEL_INFO.items():
            if self.download_model_file(filename, info):
                self.success_count += 1
            else:
                self.error_count += 1
                if info["required"]:
                    success = False
        
        return success

    def configure_environment(self):
        """Configure environment variables."""
        print(f"\n{Fore.CYAN}=== Configuring Environment ==={Style.RESET_ALL}")
        
        env_file = os.path.join(self.project_dir, ".env")
        
        # Check if user has a Picovoice access key
        picovoice_key = input(f"{Fore.YELLOW}Enter your Picovoice access key (press Enter to skip): {Style.RESET_ALL}")
        
        # Write the .env file
        with open(env_file, "w") as f:
            f.write("# Environment configuration for HowdyTTS\n")
            f.write("# Generated by setup script\n\n")
            
            # Add the Picovoice key if provided
            if picovoice_key:
                f.write(f"PORCUPINE_ACCESS_KEY=\"{picovoice_key}\"\n")
            
            # Set model path
            f.write(f"LOCAL_MODEL_PATH=\"{self.models_dir}\"\n")
            
            # Add Kokoro voice configuration
            f.write("\n# Blended voice configuration\n")
            f.write("KOKORO_VOICE=\"blended_cowboy\"\n")
            f.write("KOKORO_VOICE_AF_NICOLE_RATIO=30\n")
            f.write("KOKORO_VOICE_AM_MICHAEL_RATIO=70\n")
        
        print(f"{Fore.GREEN}✓ Environment configured successfully.{Style.RESET_ALL}")
        self.success_count += 1
        return True

    def setup_fastwhisper_api(self):
        """Set up the FastWhisperAPI service."""
        print(f"\n{Fore.CYAN}=== Setting up FastWhisperAPI ==={Style.RESET_ALL}")
        
        # Check if FastWhisperAPI directory exists
        api_dir = os.path.join(self.project_dir, "FastWhisperAPI")
        if not os.path.exists(api_dir) or not os.path.exists(os.path.join(api_dir, "main.py")):
            print(f"{Fore.RED}FastWhisperAPI directory not found or incomplete.{Style.RESET_ALL}")
            print(f"Please make sure the FastWhisperAPI directory exists in the project root.")
            return False
        
        # Use pip directly from activated conda env
        pip_command = "pip" 
        
        try:
            # Install FastWhisperAPI requirements
            print(f"{Fore.YELLOW}Installing FastWhisperAPI dependencies...{Style.RESET_ALL}")
            requirements_file = os.path.join(api_dir, "requirements.txt")
            
            if os.path.exists(requirements_file):
                subprocess.run([pip_command, "install", "-r", requirements_file], check=True) # Use pip_command
            else:
                # Fallback to common FastAPI dependencies
                subprocess.run([pip_command, "install", "fastapi", "uvicorn", "python-multipart"], check=True) # Use pip_command
            
            print(f"{Fore.GREEN}✓ FastWhisperAPI dependencies installed successfully.{Style.RESET_ALL}")
            
            # Create a startup script with correct path to uvicorn (assuming it's in PATH from conda env)
            startup_script = os.path.join(self.project_dir, "start_fastwhisper_api.sh")
            
            with open(startup_script, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# Start FastWhisperAPI service\n")
                f.write(f"# Activate conda environment first: conda activate {self.conda_env_name}\n")
                f.write("cd FastWhisperAPI\n")
                f.write(f"uvicorn main:app --reload\n") # Use uvicorn directly
            
            # Make the script executable
            os.chmod(startup_script, 0o755)
            
            print(f"{Fore.GREEN}✓ Created FastWhisperAPI startup script: {startup_script}{Style.RESET_ALL}")
            self.success_count += 1
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error setting up FastWhisperAPI: {e}{Style.RESET_ALL}")
            self.error_count += 1
            return False

    def setup_ollama(self):
        """Set up Ollama (check installation and configuration)."""
        print(f"\n{Fore.CYAN}=== Setting up Ollama ==={Style.RESET_ALL}")
        
        # Check if Ollama is installed
        try:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"{Fore.YELLOW}Ollama not found. Please install Ollama from https://ollama.com/{Style.RESET_ALL}")
                print(f"After installation, run: ollama pull hf.co/unsloth/gemma-3-4b-it-GGUF:latest")
                self.error_count += 1
                return False
            
            print(f"{Fore.GREEN}✓ Ollama is installed at: {result.stdout.strip()}{Style.RESET_ALL}")
            
            # Check if the model is pulled - improved detection
            print(f"{Fore.YELLOW}Checking Ollama models...{Style.RESET_ALL}")
            models = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            
            model_name = "gemma"  # Use a shorter, more reliable search term
            model_full = "hf.co/unsloth/gemma-3-4b-it-GGUF:latest"
            if model_name in models.stdout:
                print(f"{Fore.GREEN}✓ Ollama model containing '{model_name}' is already pulled.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Ollama model '{model_full}' not found.{Style.RESET_ALL}")
                
                choice = input(f"Would you like to pull the model now? (y/n): ").lower()
                if choice == 'y':
                    print(f"{Fore.YELLOW}Pulling Ollama model '{model_full}'...{Style.RESET_ALL}")
                    print(f"This might take a while depending on your internet connection.")
                    
                    pull_process = subprocess.run(["ollama", "pull", model_full], capture_output=True, text=True)
                    
                    if pull_process.returncode == 0:
                        print(f"{Fore.GREEN}✓ Successfully pulled Ollama model '{model_full}'.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Error pulling Ollama model: {pull_process.stderr}{Style.RESET_ALL}")
                        print(f"You can manually pull the model later with:")
                        print(f"  ollama pull {model_full}")
                        self.error_count += 1
                        return False
                else:
                    print(f"{Fore.YELLOW}Skipping model pull. You can do it manually later with:{Style.RESET_ALL}")
                    print(f"  ollama pull {model_full}")
            
            # Create an Ollama start script
            startup_script = os.path.join(self.project_dir, "start_ollama.sh")
            with open(startup_script, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# Start Ollama service\n")
                f.write("ollama serve\n")
            
            # Make the script executable
            os.chmod(startup_script, 0o755)
            
            print(f"{Fore.GREEN}✓ Created Ollama startup script: {startup_script}{Style.RESET_ALL}")
            self.success_count += 1
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error setting up Ollama: {e}{Style.RESET_ALL}")
            self.error_count += 1
            return False

    def test_microphone(self):
        """Test microphone functionality."""
        print(f"\n{Fore.CYAN}=== Testing Microphone ==={Style.RESET_ALL}")
        
        try:
            # Check if the test script exists
            test_script = os.path.join(self.project_dir, "microphone_test.py")
            if not os.path.exists(test_script):
                print(f"{Fore.RED}Microphone test script not found: {test_script}{Style.RESET_ALL}")
                return False
            
            # Use python directly from activated conda env
            python_command = "python" 
            
            print(f"{Fore.YELLOW}Running microphone test...{Style.RESET_ALL}")
            print(f"When prompted, speak a short phrase to test your microphone.")
            
            # Run the test
            time.sleep(2)  # Give user time to read
            subprocess.run([python_command, test_script], check=True) # Use python_command
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error testing microphone: {e}{Style.RESET_ALL}")
            return False

    def test_kokoro_tts(self):
        """Test Kokoro TTS functionality."""
        print(f"\n{Fore.CYAN}=== Testing Kokoro TTS ==={Style.RESET_ALL}")
        
        # First, check if kokoro_onnx is installed
        if not self.check_package_installed("kokoro_onnx"):
            print(f"{Fore.RED}The kokoro_onnx package is not installed. Installing it now...{Style.RESET_ALL}")
            
            # Use pip directly from activated conda env
            pip_command = "pip" 
            try:
                subprocess.run([pip_command, "install", "kokoro_onnx"], check=True) # Use pip_command
                print(f"{Fore.GREEN}Successfully installed kokoro_onnx package.{Style.RESET_ALL}")
            except subprocess.CalledProcessError:
                print(f"{Fore.RED}Failed to install kokoro_onnx package. Skipping TTS test.{Style.RESET_ALL}")
                return False
        
        try:
            # Create a simple test script
            test_script = os.path.join(self.temp_dir, "test_kokoro.py")
            with open(test_script, "w") as f:
                f.write("""
import soundfile as sf
from kokoro_onnx import Kokoro
import os

try:
    print("Loading Kokoro model...")
    # Assuming models are in the 'models' subdirectory relative to the script
    model_path = os.path.join("models", "kokoro-v1.0.onnx")
    voices_path = os.path.join("models", "voices-v1.0.bin")
    
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        print(f"Error: Model files not found in 'models' directory.")
        exit(1)
        
    kokoro = Kokoro(model_path, voices_path)
    
    print("Generating test audio...")
    samples, sample_rate = kokoro.create(
        "Howdy partner, this is a test of the Kokoro Text-to-Speech system.",
        voice="am_michael",
        speed=1.0,
        lang="en-us"
    )
    
    output_dir = "temp"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_kokoro.wav")
    sf.write(output_file, samples, sample_rate)
    print(f"Successfully generated audio: {output_file}")
    
    # Print available voices
    if hasattr(kokoro, 'get_available_voices'):
        voices = kokoro.get_available_voices()
        print(f"Available voices: {', '.join(voices)}")
    elif hasattr(kokoro, '_voices') and isinstance(kokoro._voices, dict):
        voices = list(kokoro._voices.keys())
        print(f"Available voices: {', '.join(voices)}")
    else:
        print("Could not retrieve available voices.")
    
except Exception as e:
    print(f"Error testing Kokoro TTS: {e}")
    exit(1)
                """)
            
            # Use python directly from activated conda env
            python_command = "python" 
            
            print(f"{Fore.YELLOW}Testing Kokoro TTS...{Style.RESET_ALL}")
            
            # Run the test
            subprocess.run([python_command, test_script], check=True) # Use python_command
            
            # Check if the output file was created
            if os.path.exists(os.path.join(self.project_dir, "temp", "test_kokoro.wav")):
                print(f"{Fore.GREEN}✓ Kokoro TTS test successful.{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}Kokoro TTS test failed: Output file not created.{Style.RESET_ALL}")
                return False
            
        except Exception as e:
            print(f"{Fore.RED}Error testing Kokoro TTS: {e}{Style.RESET_ALL}")
            return False

    def test_pyaudio_works(self):
        """Test if PyAudio is working properly."""
        print(f"{Fore.YELLOW}Testing PyAudio installation...{Style.RESET_ALL}")
        
        # Use python directly from activated conda env
        python_command = "python" 
        
        # Create a simple test script
        test_script = os.path.join(self.temp_dir, "test_pyaudio.py")
        with open(test_script, "w") as f:
            f.write("""
import pyaudio
try:
    p = pyaudio.PyAudio()
    # Get default input device info
    device_info = p.get_default_input_device_info()
    print(f"Default input device: {device_info['name']}")
    
    # Try to open a stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
        input_device_index=device_info['index']
    )
    
    # Read a small amount of data
    stream.read(1024)
    
    # Close everything
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("PyAudio test successful!")
    exit(0)
except Exception as e:
    print(f"PyAudio test failed: {e}")
    exit(1)
            """)

        # Run the test
        try:
            result = subprocess.run([python_command, test_script], capture_output=True, text=True, timeout=10) # Use python_command
            
            if result.returncode == 0:
                print(f"{Fore.GREEN}✓ PyAudio is working correctly.{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.YELLOW}PyAudio test failed: {result.stderr}{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.YELLOW}PyAudio test error: {e}{Style.RESET_ALL}")
            return False

    def create_run_script(self):
        """Create a run script for easy startup using Conda."""
        print(f"\n{Fore.CYAN}=== Creating Run Script ==={Style.RESET_ALL}")
        
        run_script = os.path.join(self.project_dir, "run_howdy.sh")
        
        try:
            with open(run_script, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# HowdyTTS Launcher Script (using Conda)\n")
                f.write("# Created by mac_setup.py\n\n")
                
                f.write("# Activate Conda environment\n")
                f.write(f"conda activate {self.conda_env_name}\n\n")
                
                f.write("# Check if Ollama is running\n")
                f.write("if ! pgrep -x \"ollama\" > /dev/null; then\n")
                f.write("    echo \"Starting Ollama...\"\n")
                f.write("    ollama serve &\n")
                f.write("    OLLAMA_PID=$!\n")
                f.write("    sleep 2\n")
                f.write("fi\n\n")
                
                f.write("# Check if FastWhisperAPI is running\n")
                f.write("if ! curl -s http://localhost:8000/info > /dev/null; then\n")
                f.write("    echo \"Starting FastWhisperAPI...\"\n")
                f.write("    cd FastWhisperAPI\n")
                f.write(f"    uvicorn main:app --reload &\n") # Use uvicorn directly
                f.write("    FASTWHISPER_PID=$!\n")
                f.write("    cd ..\n")
                f.write("    sleep 2\n")
                f.write("fi\n\n")
                
                f.write("# Run HowdyTTS\n")
                f.write("python run_voice_assistant.py\n\n")
                
                # Add cleanup for background processes (optional but good practice)
                f.write("cleanup() {\n")
                f.write("    echo \"Cleaning up background processes...\"\n")
                f.write("    [ ! -z \"$OLLAMA_PID\" ] && kill $OLLAMA_PID\n")
                f.write("    [ ! -z \"$FASTWHISPER_PID\" ] && kill $FASTWHISPER_PID\n")
                f.write("}\n")
                f.write("trap cleanup EXIT\n")

            # Make the script executable
            os.chmod(run_script, 0o755)
            
            print(f"{Fore.GREEN}✓ Created launcher script: {run_script}{Style.RESET_ALL}")
            print(f"You can now start HowdyTTS with: ./run_howdy.sh")
            print(f"(Ensure '{self.conda_env_name}' is activated first if running manually)")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error creating run script: {e}{Style.RESET_ALL}")
            return False

    def check_package_installed(self, package_name):
        """Check if a package is installed in the active environment."""
        # Use python directly from activated conda env
        python_command = "python" 
        result = subprocess.run(
            [python_command, "-c", f"import {package_name}; print('Package found')"], # Use python_command
            capture_output=True,
            text=True
        )
        return result.returncode == 0

    def run(self, download_models=True, test_components=True, setup_services=True):
        """Run the setup process."""
        print(f"\n{Fore.CYAN}=============================={Style.RESET_ALL}")
        print(f"{Fore.CYAN}  HowdyTTS Mac Setup Utility  {Style.RESET_ALL}")
        print(f"{Fore.CYAN}=============================={Style.RESET_ALL}")
        
        self.print_system_info()
        
        # Check Python version (of the current interpreter, should be from conda env)
        if not self.check_python_version():
            return # Exit if Python version is wrong
        
        # Check Conda environment
        if not self.check_conda_environment():
            print(f"{Fore.RED}Exiting setup. Please create or activate the '{self.conda_env_name}' Conda environment.{Style.RESET_ALL}")
            return # Exit if conda env is not found/activated
        
        # Install dependencies
        if not self.install_dependencies():
            print(f"{Fore.RED}Dependency installation failed. Exiting setup.{Style.RESET_ALL}")
            return # Exit if dependencies fail
        
        # Download models if requested
        if download_models:
            if not self.download_models():
                print(f"{Fore.YELLOW}Model download failed for some required models. Setup may be incomplete.{Style.RESET_ALL}")
                # Continue if possible, but warn user
        
        # Configure environment (.env file)
        if not self.configure_environment():
             # This is usually not critical, so just warn
             print(f"{Fore.YELLOW}Failed to configure .env file. Please check manually.{Style.RESET_ALL}")

        # Setup services if requested
        if setup_services:
            if not self.setup_fastwhisper_api():
                 print(f"{Fore.YELLOW}Failed to set up FastWhisperAPI. Transcription might not work locally.{Style.RESET_ALL}")
            if not self.setup_ollama():
                 print(f"{Fore.YELLOW}Failed to set up Ollama. Local LLM responses might not work.{Style.RESET_ALL}")

        # Test components if requested
        if test_components:
            print(f"\n{Fore.CYAN}=== Running Component Tests ==={Style.RESET_ALL}")
            mic_ok = self.test_microphone()
            tts_ok = self.test_kokoro_tts()
            
            if mic_ok: self.success_count += 1 
            else: self.error_count += 1
            if tts_ok: self.success_count += 1
            else: self.error_count += 1
            
            print(f"Microphone Test: {'Passed' if mic_ok else 'Failed'}")
            print(f"Kokoro TTS Test: {'Passed' if tts_ok else 'Failed'}")

        # Create run script
        self.create_run_script() # Attempt to create regardless of previous errors

        # Final summary
        print(f"\n{Fore.CYAN}=== Setup Summary ==={Style.RESET_ALL}")
        print(f"Successful steps: {self.success_count}")
        print(f"Errors encountered: {self.error_count}")
        
        if self.error_count == 0:
            print(f"{Fore.GREEN}✓ HowdyTTS setup completed successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ HowdyTTS setup completed with errors. Please review the logs.{Style.RESET_ALL}")

def main():
    """Main function to run the setup script."""
    parser = argparse.ArgumentParser(description="HowdyTTS Mac Setup Script")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading model files")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running component tests")
    parser.add_argument("--skip-services", action="store_true", help="Skip setting up FastWhisperAPI and Ollama")
    
    args = parser.parse_args()
    
    setup_manager = SetupManager()
    setup_manager.run(
        download_models=not args.skip_models,
        test_components=not args.skip_tests,
        setup_services=not args.skip_services
    )

if __name__ == "__main__":
    main()