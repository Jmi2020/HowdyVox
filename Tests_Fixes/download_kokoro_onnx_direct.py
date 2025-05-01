#!/usr/bin/env python3
"""
Direct download script for Kokoro ONNX model
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import shutil
import sys

def download_file(url, dest_path):
    """Download a file with progress tracking"""
    try:
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
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Directory setup
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    voices_dir = models_dir / "voices"
    voices_dir.mkdir(exist_ok=True)
    
    # Direct links to model files (GitHub raw content or Hugging Face URLs)
    model_url = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx?download=true"
    am_michael_url = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/am_michael.bin?download=true"
    
    # Destination paths
    model_path = models_dir / "kokoro-v1.0.onnx"
    am_michael_path = voices_dir / "am_michael.bin"     
    
    # Download files
    print(f"Downloading Kokoro ONNX model to {model_path}")
    if download_file(model_url, model_path):
        print(f"✓ Successfully downloaded model to {model_path}")
    else:
        print(f"❌ Failed to download model")
        return 1
    
    print(f"Downloading am_michael voice to {am_michael_path}")
    if download_file(am_michael_url, am_michael_path):
        print(f"✓ Successfully downloaded am_michael voice")       
    else:
        print(f"❌ Failed to download am_michael voice")
        return 1
    
    print("\nSetup complete! Files downloaded successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
