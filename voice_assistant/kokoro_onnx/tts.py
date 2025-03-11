import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json
import re
import soundfile as sf
import scipy.io.wavfile as wavfile
from typing import List, Dict, Optional, Tuple, Union, Any

class KokoroOnnxTTS:
    """
    KokoroTTS implementation using ONNX runtime for better performance
    and platform compatibility, following the official Kokoro-ONNX approach.
    """
    
    DEFAULT_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
    SAMPLE_RATE = 24000  # Modern Kokoro models use 24kHz sample rate
    
    def __init__(self, model_path: Optional[str] = None, voice: str = "am_michael", dtype: str = "q8"):
        """
        Initialize the KokoroOnnxTTS with a specified model path and voice.
        
        Args:
            model_path: Path to the ONNX model directory. If None, will use default models.
            voice: Voice ID to use. Default is "am_michael" (cowboy voice).
            dtype: Model precision - "fp32", "fp16", "q8", "q4", or "q4f16"
        """
        self.voice_name = voice
        self.dtype = dtype
        
        # If model_path is not provided, use default model path
        if model_path is None:
            self.model_path = self._get_default_model_path()
        else:
            self.model_path = model_path
        
        # Determine model file based on dtype
        self.model_file = self._get_model_filename_for_dtype(dtype)
            
        # Load the model and voices
        self.session = self._load_model()
        self.voices = self._load_voices()
        
        # Load phoneme mapping for text processing
        self.phoneme_dict = self._load_phoneme_dict()
        
        print(f"KokoroOnnxTTS initialized with voice: {voice}, dtype: {dtype}")
    
    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_MODEL_ID, 
                      options: Dict[str, Any] = None) -> 'KokoroOnnxTTS':
        """
        Create a KokoroOnnxTTS instance from a pretrained model.
        
        Args:
            model_id: The model ID to load, defaults to "onnx-community/Kokoro-82M-v1.0-ONNX"
            options: Configuration options:
                     - dtype: Model precision ("fp32", "fp16", "q8", "q4", "q4f16")
                     - voice: Voice ID to use
                     
        Returns:
            KokoroOnnxTTS: A new instance
        """
        if options is None:
            options = {}
            
        dtype = options.get("dtype", "q8")
        voice = options.get("voice", "am_michael")
        
        # Check if model exists locally, if not try to download it
        user_home = os.path.expanduser("~")
        models_dir = os.path.join(user_home, ".kokoro_onnx")
        
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            print(f"Model not found locally. Downloading from {model_id}...")
            try:
                # Try to import the download function
                script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                import sys
                sys.path.append(script_dir)
                
                try:
                    from download_kokoro_onnx import download_kokoro_onnx
                    success = download_kokoro_onnx(output_dir=models_dir, model_type=dtype)
                    if not success:
                        print("Failed to download model automatically.")
                except ImportError:
                    print("Cannot import download_kokoro_onnx script.")
                    print(f"Please run the download script manually:")
                    print(f"python download_kokoro_onnx.py --type {dtype}")
            except Exception as e:
                print(f"Error attempting to download model: {e}")
        
        return cls(model_path=None, voice=voice, dtype=dtype)
        
    def _get_default_model_path(self) -> str:
        """Get the default model path."""
        user_home = os.path.expanduser("~")
        models_dir = os.path.join(user_home, ".kokoro_onnx")
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            print(f"Created model directory: {models_dir}")
            print("You'll need to download the ONNX model files.")
            
        return models_dir
    
    def _get_model_filename_for_dtype(self, dtype: str) -> str:
        """Get the appropriate model filename based on dtype."""
        dtype_map = {
            "fp32": "model.onnx",
            "fp16": "model_fp16.onnx",
            "q8": "model_quantized.onnx",  # q8 is the default quantized model
            "q8f16": "model_q8f16.onnx",
            "q4": "model_q4.onnx",
            "q4f16": "model_q4f16.onnx",
            "uint8": "model_uint8.onnx",
            "uint8f16": "model_uint8f16.onnx"
        }
        
        if dtype not in dtype_map:
            print(f"Warning: Unknown dtype '{dtype}', falling back to 'q8'")
            dtype = "q8"
            
        return dtype_map[dtype]
    
    def _load_model(self) -> ort.InferenceSession:
        """Load the ONNX model."""
        model_file = os.path.join(self.model_path, "onnx", self.model_file)
        
        if not os.path.exists(model_file):
            # Try without the "onnx" subdirectory
            model_file = os.path.join(self.model_path, self.model_file)
            
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Create ONNX runtime session with optimizations
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create inference session
        return ort.InferenceSession(model_file, session_options)
    
    def _load_voices(self) -> Dict[str, np.ndarray]:
        """Load voice style vectors."""
        # Try common voice directories
        possible_voice_dirs = [
            os.path.join(self.model_path, "voices"),
            self.model_path,
            os.path.join(os.path.dirname(self.model_path), "voices")
        ]
        
        voices = {}
        voice_files = []
        voices_dir = None
        
        # Find a directory with voice files
        for dir_path in possible_voice_dirs:
            if os.path.exists(dir_path):
                try:
                    files = [f for f in os.listdir(dir_path) 
                            if f.endswith('.bin') or f.endswith('.npy')]
                    if files:
                        voice_files = files
                        voices_dir = dir_path
                        break
                except Exception as e:
                    print(f"Error checking directory {dir_path}: {e}")
        
        if not voices_dir or not voice_files:
            print("Warning: No voice files found in any expected location.")
            # Create a dummy voice if no voices were found
            if self.voice_name and self.voice_name not in voices:
                print(f"Creating a dummy voice for '{self.voice_name}'")
                # Create a dummy embedding of random values
                voices[self.voice_name] = np.random.randn(512, 1, 256).astype(np.float32)
            return voices
            
        print(f"Found voice files in: {voices_dir}")
        for voice_file in voice_files:
            voice_name = os.path.splitext(voice_file)[0]
            voice_path = os.path.join(voices_dir, voice_file)
            
            try:
                if voice_file.endswith('.bin'):
                    # Try different shapes if loading fails
                    try:
                        # Most common shape for Kokoro embeddings
                        voice_data = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
                    except ValueError:
                        try:
                            # Try alternative shape
                            voice_data = np.fromfile(voice_path, dtype=np.float32).reshape(1, -1, 256)
                        except ValueError:
                            # Last resort: just load as flat array
                            voice_data = np.fromfile(voice_path, dtype=np.float32)
                            # Reshape to expected format (N, 1, 256)
                            voice_data = voice_data.reshape(-1, 1, 256)
                else:  # .npy file
                    voice_data = np.load(voice_path)
                    # Ensure proper shape
                    if len(voice_data.shape) == 1:
                        voice_data = voice_data.reshape(-1, 1, 256)
                
                voices[voice_name] = voice_data
                print(f"Loaded voice: {voice_name}, shape: {voice_data.shape}")
            except Exception as e:
                print(f"Failed to load voice {voice_name}: {e}")
        
        # Check for specifically requested voice
        if self.voice_name and self.voice_name not in voices:
            # Look for alternative name formats
            possible_names = [
                # Common variants - the voice name might be a language code
                f"{self.voice_name}_michael",  # am_michael
                self.voice_name.split('_')[0],  # am from am_michael
                'en',  # fallback to English
                'am'   # fallback to American
            ]
            
            for alt_name in possible_names:
                if alt_name in voices:
                    print(f"Voice '{self.voice_name}' not found, using '{alt_name}' instead")
                    voices[self.voice_name] = voices[alt_name].copy()
                    break
            
            # If no matching voice was found, create a dummy
            if self.voice_name not in voices and voices:
                # Use the first available voice and copy it
                first_voice = list(voices.keys())[0]
                print(f"Creating a voice for '{self.voice_name}' based on '{first_voice}'")
                voices[self.voice_name] = voices[first_voice].copy()
            elif self.voice_name not in voices:
                # No voices at all - create random
                print(f"Creating a dummy voice for '{self.voice_name}'")
                voices[self.voice_name] = np.random.randn(512, 1, 256).astype(np.float32)
                
        return voices
    
    def list_voices(self) -> List[str]:
        """List all available voices."""
        return list(self.voices.keys())
    
    def _load_phoneme_dict(self) -> Dict:
        """Load phoneme mapping."""
        # Try to find phoneme mapping in common locations
        config_paths = [
            os.path.join(self.model_path, "config.json"),
            os.path.join(self.model_path, "phonemes.json"),
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                        
                    # Check if the file contains phoneme mapping
                    if "phoneme_id_map" in config:
                        return config["phoneme_id_map"]
                    elif "phonemes" in config:
                        return config["phonemes"]
                except Exception as e:
                    print(f"Failed to load phoneme dict from {path}: {e}")
        
        print("Warning: No phoneme dictionary found, using defaults")
        # Create a basic default phoneme dictionary
        default_dict = {
            # Special tokens
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            
            # Lowercase letters (phoneme base)
            'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10,
            'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17,
            'o': 18, 'p': 19, 'q': 20, 'r': 21, 's': 22, 't': 23, 'u': 24,
            'v': 25, 'w': 26, 'x': 27, 'y': 28, 'z': 29,
            
            # Common punctuation and whitespace
            ' ': 30, '.': 31, ',': 32, '!': 33, '?': 34, ':': 35, ';': 36,
            '-': 37, '\'': 38, '"': 39,
            
            # Special phonetic symbols we're using
            'T': 40,  # th
            'S': 41,  # sh
            'C': 42,  # ch
            'N': 43,  # ng
        }
        return default_dict
        
    def _text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phonemes.
        
        Very simple implementation that just uses the raw characters.
        """
        # Simple character-by-character approach - no fancy phoneme conversion
        return list(text.lower())
        
    def _phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to token IDs using a very basic scheme."""
        ids = []
        
        # Unknown phoneme ID
        unk_id = 1
        
        # Use a very simple mapping based on ASCII values
        for phoneme in phonemes:
            # Map space to a special token ID
            if phoneme == ' ':
                ids.append(30)
            # Map letters (a-z) to IDs 2-27
            elif 'a' <= phoneme <= 'z':
                ids.append(ord(phoneme) - ord('a') + 2)
            # Map digits (0-9) to IDs 31-40
            elif '0' <= phoneme <= '9':
                ids.append(ord(phoneme) - ord('0') + 31)
            # Map punctuation to IDs 41+
            elif phoneme in '.,:;!?-_':
                ids.append(41 + '.,:;!?-_'.index(phoneme))
            # For anything else, use the unknown ID
            else:
                ids.append(unk_id)
            
        return ids
        
    def generate(self, text: str, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        Args:
            text: The text to convert to speech
            options: Optional parameters:
                     - voice: Override the voice to use
                     - speed: Speech speed factor (default: 1.0)
                     - output_file: If provided, save audio to this path
            
        Returns:
            Tuple[np.ndarray, int]: The generated audio and sample rate
        """
        if options is None:
            options = {}
            
        # Get voice from options or use default
        voice_name = options.get("voice", self.voice_name)
        speed = options.get("speed", 1.0)
        output_file = options.get("output_file")
        
        try:
            # Convert text to phonemes and then to IDs
            phonemes = self._text_to_phonemes(text)
            tokens = self._phonemes_to_ids(phonemes)
            
            # Ensure tokens aren't too long (context length is typically 512)
            if len(tokens) > 510:  # Save space for pad tokens
                print(f"Warning: Input too long ({len(tokens)} tokens), truncating to 510")
                tokens = tokens[:510]
            
            # Check if voice exists
            if voice_name not in self.voices:
                available_voices = list(self.voices.keys())
                print(f"Voice '{voice_name}' not found. Available voices: {available_voices}")
                if len(available_voices) > 0:
                    voice_name = available_voices[0]
                    print(f"Falling back to voice: {voice_name}")
                else:
                    raise ValueError("No voice data available")
            
            # Get the voice style vector - use the first one
            voice_data = self.voices[voice_name]
            ref_s = voice_data[0]
            
            # Add pad tokens at beginning and end
            padded_tokens = [[0, *tokens, 0]]
            
            # Prepare inputs
            model_inputs = {
                "input_ids": np.array(padded_tokens, dtype=np.int64),
                "style": ref_s,
                "speed": np.array([speed], dtype=np.float32)
            }
            
            # Run inference
            audio = self.session.run(None, model_inputs)[0]
            
            # The output is typically [1, T] where T is the number of samples
            waveform = audio[0]  # Get the first (and only) audio sample
            
            # Save to file if requested
            if output_file:
                wavfile.write(output_file, self.SAMPLE_RATE, waveform)
                print(f"Audio saved to {output_file}")
                
            return waveform, self.SAMPLE_RATE
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            # Return empty array on error
            return np.array([], dtype=np.float32), self.SAMPLE_RATE
    
    def save(self, audio: np.ndarray, filename: str) -> None:
        """Save audio to file."""
        wavfile.write(filename, self.SAMPLE_RATE, audio)
        print(f"Audio saved to {filename}")
    
    # Legacy methods for backward compatibility
    def tts(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """Legacy method for backward compatibility."""
        audio, _ = self.generate(text, {"output_file": output_path})
        return audio
    
    def generate_speech(self, text: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Legacy method for backward compatibility with the original KokoroTTS API."""
        return self.generate(text, {"output_file": output_path})