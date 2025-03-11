#!/usr/bin/env python3
"""
Download script for Kokoro ONNX model
"""

import os
import sys
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import json

def download_file(url, dest_path, max_retries=3):
    """
    Download a file with progress tracking and retry logic
    
    Args:
        url: URL to download from
        dest_path: Path to save the file to
        max_retries: Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            # Check if the URL is accessible
            head_response = requests.head(url, timeout=10)
            if head_response.status_code != 200:
                print(f"Warning: URL returned status {head_response.status_code}: {url}")
                if attempt < max_retries - 1:
                    print(f"Retry attempt {attempt + 1}/{max_retries}...")
                    continue
                else:
                    return False
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 KB blocks
            
            print(f"Downloading: {url} -> {dest_path}")
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            # Verify the download
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                print(f"✓ Successfully downloaded {os.path.basename(dest_path)}")
                return True
            else:
                print(f"× File appears empty or corrupt: {dest_path}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                if attempt < max_retries - 1:
                    print(f"Retry attempt {attempt + 1}/{max_retries}...")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"Network error downloading {url}: {e}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            
        # Remove partial download if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
            
        if attempt < max_retries - 1:
            print(f"Retry attempt {attempt + 1}/{max_retries}...")
        else:
            print(f"× Failed to download after {max_retries} attempts: {url}")
            
    return False

def download_kokoro_onnx(output_dir=None, model_type="q8"):
    """
    Download the Kokoro ONNX model and voice files
    
    Args:
        output_dir: Directory to save model files (default: ~/.kokoro_onnx)
        model_type: Model precision type (fp32, fp16, q8, q4)
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.expanduser("~/.kokoro_onnx")
    
    # Create directories
    onnx_dir = os.path.join(output_dir, "onnx")
    voices_dir = os.path.join(output_dir, "voices")
    
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(voices_dir, exist_ok=True)
    
    print(f"Model files will be downloaded to: {output_dir}")
    
    # Get model filename based on type
    model_file_map = {
        "fp32": "model.onnx",
        "fp16": "model_fp16.onnx",
        "q8": "model_quantized.onnx",
        "q4": "model_q4.onnx"
    }
    
    if model_type not in model_file_map:
        print(f"Invalid model type: {model_type}")
        print(f"Valid types: {', '.join(model_file_map.keys())}")
        return False
    
    model_filename = model_file_map[model_type]
    
    # Base URL for the model files (from Hugging Face)
    base_url = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"
    
    # Alternative URLs to try if the main URL doesn't work
    alt_base_urls = [
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/raw/main",  # Raw file access
        "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main",   # Original repo without v1.0
        "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/raw/main"        # Original repo raw access
    ]
    
    # Define file paths to download
    file_paths = {
        "model": f"onnx/{model_filename}",
        "voices/af": "voices/af.bin",
        "voices/am": "voices/am.bin",
        "voices/ar": "voices/ar.bin",
        "voices/zh": "voices/zh.bin",
        "voices/en": "voices/en.bin",
        "voices/fr": "voices/fr.bin",
        "voices/de": "voices/de.bin",
        "voices/nl": "voices/nl.bin",
        "voices/ru": "voices/ru.bin",
        "voices/es": "voices/es.bin",
        "config": "config.json"
    }
    
    # Map file paths to local destinations
    dest_paths = {
        "model": os.path.join(onnx_dir, model_filename),
        "voices/af": os.path.join(voices_dir, "af.bin"),
        "voices/am": os.path.join(voices_dir, "am.bin"),
        "voices/ar": os.path.join(voices_dir, "ar.bin"),
        "voices/zh": os.path.join(voices_dir, "zh.bin"),
        "voices/en": os.path.join(voices_dir, "en.bin"),
        "voices/fr": os.path.join(voices_dir, "fr.bin"),
        "voices/de": os.path.join(voices_dir, "de.bin"),
        "voices/nl": os.path.join(voices_dir, "nl.bin"),
        "voices/ru": os.path.join(voices_dir, "ru.bin"),
        "voices/es": os.path.join(voices_dir, "es.bin"),
        "config": os.path.join(output_dir, "config.json")
    }
    
    # Create a symbolic link from 'am.bin' to 'am_michael.bin' for compatibility
    # with our code which expects 'am_michael' voice
    am_michael_link = os.path.join(voices_dir, "am_michael.bin")
    am_file = os.path.join(voices_dir, "am.bin")
    
    # Download each file, trying different base URLs if needed
    success_count = 0
    for file_key, file_path in file_paths.items():
        dest_path = dest_paths[file_key]
        
        # Try main URL first
        main_url = f"{base_url}/{file_path}"
        print(f"\nAttempting to download: {file_key}")
        
        if download_file(main_url, dest_path):
            success_count += 1
            continue
            
        # If main URL fails, try alternative URLs
        print(f"Main URL failed, trying alternative URLs for {file_key}...")
        downloaded = False
        
        for alt_url in alt_base_urls:
            alt_full_url = f"{alt_url}/{file_path}"
            print(f"Trying: {alt_full_url}")
            
            if download_file(alt_full_url, dest_path):
                success_count += 1
                downloaded = True
                break
                
        if not downloaded:
            print(f"⚠️ Failed to download {file_key} from all sources")
            
            # For the American voice, try to download directly as am_michael.bin
            if file_key == "voices/am":
                print("Trying to download am_michael.bin directly...")
                for try_url in [base_url, *alt_base_urls]:
                    direct_url = f"{try_url}/voices/am_michael.bin"
                    if download_file(direct_url, am_michael_link):
                        print(f"✓ Successfully downloaded am_michael.bin directly")
                        success_count += 1
                        downloaded = True
                        break
    
    # Create the symbolic link for am_michael if am.bin was downloaded
    if os.path.exists(am_file):
        print(f"Creating symbolic link: {am_michael_link} -> {am_file}")
        # If on Windows, might need to copy instead of link
        if os.name == 'nt':
            shutil.copy2(am_file, am_michael_link)
        else:
            # Remove existing link if it exists
            if os.path.exists(am_michael_link):
                os.remove(am_michael_link)
            # Create symbolic link
            os.symlink(am_file, am_michael_link)
    elif not os.path.exists(am_michael_link):
        # If neither am.bin nor am_michael.bin was downloaded, use any available voice file
        voice_files = [f for f in os.listdir(voices_dir) if f.endswith(".bin")]
        if voice_files:
            # Use the first available voice file as am_michael
            print(f"No am.bin or am_michael.bin found, using {voice_files[0]} as fallback")
            fallback_voice = os.path.join(voices_dir, voice_files[0])
            if os.name == 'nt':
                shutil.copy2(fallback_voice, am_michael_link)
            else:
                # Create symbolic link
                if os.path.exists(am_michael_link):
                    os.remove(am_michael_link)
                os.symlink(fallback_voice, am_michael_link)
        else:
            print("No voice files available. Voice synthesis may not work properly.")
    
    # Create a simple version of config.json if it doesn't exist
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "model": {
                "name": "Kokoro-82M-ONNX",
                "version": "1.0.0"
            },
            "audio": {
                "sampling_rate": 24000
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    # Count required files for minimum functionality
    required_files = [
        os.path.exists(dest_paths["model"]),  # Model file
        os.path.exists(config_path),          # Config file
        any([os.path.exists(am_file), os.path.exists(am_michael_link), 
             any([os.path.exists(os.path.join(voices_dir, f)) for f in os.listdir(voices_dir) if f.endswith(".bin")])])
    ]
    
    print(f"\nDownloaded {success_count} of {len(file_paths)} files")
    
    # Check if we have the minimum required files
    if all(required_files):
        print("\nBasic functionality files downloaded successfully!")
        if success_count == len(file_paths):
            print("All files were downloaded successfully!")
        else:
            print(f"Downloaded {success_count}/{len(file_paths)} files - some voice options may be missing")
            
        print(f"\nTo use the model:")
        print(f"1. Make sure you have the dependencies installed:")
        print(f"   pip install onnxruntime numpy scipy")
        print(f"2. Set TTS_MODEL = 'kokoro_onnx' in voice_assistant/config.py")
        print(f"3. Run the voice assistant: python run_voice_assistant.py")
        return True
    else:
        print("\nSome essential files could not be downloaded.")
        print("Requirements: model file, config file, and at least one voice file") 
        print("Please check your internet connection and try again.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kokoro ONNX model and voice files")
    parser.add_argument("--output", help="Output directory (default: ~/.kokoro_onnx)")
    parser.add_argument("--type", default="q8", choices=["fp32", "fp16", "q8", "q4"], 
                        help="Model precision type (default: q8)")
    parser.add_argument("--local", action="store_true",
                        help="Install to local project directory instead of user home directory")
    
    args = parser.parse_args()
    
    # Handle local installation option
    if args.local:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.join(script_dir, "voice_assistant", "kokoro_onnx", "model")
        print(f"Installing to local project directory: {project_dir}")
        output_dir = project_dir
    else:
        output_dir = args.output
    
    if download_kokoro_onnx(output_dir, args.type):
        sys.exit(0)
    else:
        sys.exit(1)