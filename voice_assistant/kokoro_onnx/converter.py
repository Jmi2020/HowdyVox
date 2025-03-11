import os
import torch
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

def convert_kokoro_to_onnx(
    source_model_path: str, 
    output_dir: Optional[str] = None,
    voice_name: str = "am_michael"
) -> str:
    """
    Convert a KokoroTTS PyTorch model to ONNX format.
    
    Args:
        source_model_path: Path to the source KokoroTTS model directory
        output_dir: Directory to save the ONNX model (default: ~/.kokoro_onnx/models/{voice_name})
        voice_name: Name of the voice model
        
    Returns:
        str: Path to the output directory containing the ONNX model
    """
    try:
        # Try to import kokoro, but don't fail if it's not installed
        import kokoro
        print(f"Found KokoroTTS version: {kokoro.__version__}")
    except ImportError:
        print("KokoroTTS not found. Please install it using: pip install kokoro")
        return ""
    
    # Set default output directory if not provided
    if output_dir is None:
        user_home = os.path.expanduser("~")
        output_dir = os.path.join(user_home, ".kokoro_onnx", "models", voice_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check if source model exists
    if not os.path.exists(source_model_path):
        print(f"Source model not found at {source_model_path}")
        return ""
    
    # Load the original KokoroTTS model
    print("Loading KokoroTTS model...")
    model = kokoro.load_tts_model(source_model_path)
    
    # Extract the model configuration
    config = _extract_model_config(model, source_model_path)
    
    # Save configuration to the output directory
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Extract and save symbols
    symbols_dict = _extract_symbols(model)
    with open(os.path.join(output_dir, "symbols.json"), "w") as f:
        json.dump(symbols_dict, f, indent=2)
    
    # Convert the model to ONNX
    print("Converting to ONNX format...")
    try:
        # Create dummy input tensor based on model's expected input shape
        # This is a placeholder - actual shapes depend on the specific model architecture
        dummy_input_length = 100  # Example length
        dummy_input = torch.zeros(1, dummy_input_length, dtype=torch.int64)
        
        # Set the model to evaluation mode
        model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            model,                               # PyTorch model
            dummy_input,                         # Example input
            os.path.join(output_dir, "model.onnx"),  # Output file
            export_params=True,                  # Store model weights in the model file
            opset_version=12,                    # ONNX opset version
            do_constant_folding=True,            # Optimize constant folding
            input_names=["input"],               # Model input names
            output_names=["output"],             # Model output names
            dynamic_axes={                       # Dynamic axes
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size", 2: "output_length"}
            }
        )
        
        print("ONNX conversion successful!")
        
        # Copy any additional necessary files
        _copy_additional_files(source_model_path, output_dir)
        
        return output_dir
        
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return ""

def _extract_model_config(model: Any, source_path: str) -> Dict:
    """Extract configuration from the model and source path."""
    config = {
        "audio": {
            "sampling_rate": getattr(model, "sample_rate", 22050),
            "max_wav_length": 10000  # Default value
        },
        "model": {
            "name": "kokoro_onnx_model",
            "version": "0.1.0",
            "source_model": os.path.basename(source_path)
        }
    }
    
    # Try to read config from source path
    source_config_path = os.path.join(source_path, "config.json")
    if os.path.exists(source_config_path):
        try:
            with open(source_config_path, "r") as f:
                source_config = json.load(f)
                
            # Update our config with values from source config
            if "audio" in source_config:
                config["audio"].update(source_config["audio"])
                
            # Keep track of the original model information
            if "model" in source_config:
                config["model"]["source_model_info"] = source_config["model"]
        except Exception as e:
            print(f"Warning: Could not read source config: {e}")
    
    return config

def _extract_symbols(model: Any) -> Dict:
    """Extract symbol dictionary from the model."""
    symbols = {}
    
    # Try to get symbols from the model
    try:
        # Different models might store symbols differently
        if hasattr(model, "symbols"):
            # If symbols are stored as a list
            if isinstance(model.symbols, list):
                symbols = {sym: i for i, sym in enumerate(model.symbols)}
            # If symbols are stored as a dictionary
            elif isinstance(model.symbols, dict):
                symbols = model.symbols
        elif hasattr(model, "symbol_to_id"):
            symbols = model.symbol_to_id
        else:
            # Default basic symbols
            symbols = {
                "pad": 0,
                "unk": 1,
                " ": 2,
                "!": 3,
                "\"": 4,
                "#": 5,
                # Add more symbols as needed
            }
            print("Warning: Could not extract symbols from model. Using default symbols.")
    except Exception as e:
        print(f"Warning: Error extracting symbols: {e}")
        symbols = {"pad": 0, "unk": 1}
    
    return symbols

def _copy_additional_files(source_path: str, output_dir: str):
    """Copy any additional necessary files from source to output directory."""
    # Files that might be needed by the model
    additional_files = ["symbols.json", "vocab.txt", "phonemizer_config.json"]
    
    for filename in additional_files:
        source_file = os.path.join(source_path, filename)
        if os.path.exists(source_file):
            shutil.copy(source_file, os.path.join(output_dir, filename))
            print(f"Copied {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert KokoroTTS model to ONNX format")
    parser.add_argument("--source", required=True, help="Path to source KokoroTTS model directory")
    parser.add_argument("--output", help="Output directory for ONNX model")
    parser.add_argument("--voice", default="am_michael", help="Voice name (default: am_michael)")
    
    args = parser.parse_args()
    
    convert_kokoro_to_onnx(args.source, args.output, args.voice)