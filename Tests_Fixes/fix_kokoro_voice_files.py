#!/usr/bin/env python3
"""
Fix script for Kokoro ONNX voice blending data type issues
"""

import os
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fix_voice_files():
    """
    Ensure voice files are in the correct format for the ONNX model
    """
    # Create directory if it doesn't exist
    voice_dir = Path("models/voices")
    voice_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if voice files exist
    voice_files = list(voice_dir.glob("*.bin"))
    
    if not voice_files:
        logging.error("No voice files found in models/voices. Please run the download script first.")
        return False
    
    success = True
    # Process each voice file
    for voice_file in voice_files:
        try:
            logging.info(f"Processing voice file: {voice_file}")
            
            # Read the current file
            with open(voice_file, 'rb') as f:
                data = f.read()
            
            # Check if file is unexpectedly small
            if len(data) < 100:
                logging.warning(f"Voice file {voice_file} is too small ({len(data)} bytes). May be corrupted.")
                continue
                
            # Create a backup
            backup_file = voice_file.with_suffix('.bin.bak')
            with open(backup_file, 'wb') as f:
                f.write(data)
            
            # Ensure the data is in float format
            # This is a simplified approach - in a real scenario, you'd need to 
            # know the exact format expected by the model
            try:
                # Try to load with numpy and save as float32
                arr = np.frombuffer(data, dtype=np.int32)
                float_arr = arr.astype(np.float32)
                
                # Save back to file
                with open(voice_file, 'wb') as f:
                    f.write(float_arr.tobytes())
                    
                logging.info(f"Successfully converted {voice_file} to float32 format")
            except Exception as e:
                logging.error(f"Error converting {voice_file}: {e}")
                # Restore from backup
                with open(backup_file, 'rb') as f_in:
                    with open(voice_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                success = False
        
        except Exception as e:
            logging.error(f"Error processing {voice_file}: {e}")
            success = False
    
    return success

if __name__ == "__main__":
    print("Running Kokoro voice file data type fixer...")
    if fix_voice_files():
        print("✓ Voice files processed successfully!")
    else:
        print("⚠️ Some errors occurred while processing voice files. Check the logs for details.")
