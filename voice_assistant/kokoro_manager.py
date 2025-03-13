# voice_assistant/kokoro_manager.py

import os
import logging
import warnings
from kokoro_onnx import Kokoro

class KokoroManager:
    """
    Singleton class to manage a persistent instance of the Kokoro TTS model.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, model_path="kokoro-v1.0.onnx", voices_path="voices-v1.0.bin", local_model_path=None):
        """
        Get or initialize the Kokoro TTS model instance.
        
        Args:
            model_path (str): Path to the Kokoro ONNX model file
            voices_path (str): Path to the Kokoro voices file
            local_model_path (str): Optional custom path for model files
            
        Returns:
            Kokoro: The initialized Kokoro TTS model
        """
        if cls._instance is None:
            logging.info("Initializing Kokoro TTS model (first-time setup)...")
            
            # If custom model paths are provided, use them
            if local_model_path:
                if os.path.isdir(local_model_path):
                    model_path = os.path.join(local_model_path, "kokoro-v1.0.onnx")
                    voices_path = os.path.join(local_model_path, "voices-v1.0.bin")
                else:
                    model_path = local_model_path
            
            # Initialize the Kokoro model and suppress warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    cls._instance = Kokoro(model_path, voices_path)
                    logging.info("Kokoro TTS model successfully initialized")
                except Exception as e:
                    logging.error(f"Failed to initialize Kokoro TTS model: {e}")
                    raise
        
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        Reset the Kokoro instance, forcing reinitialization on next use.
        Useful for troubleshooting or when model files change.
        """
        cls._instance = None
        logging.info("Kokoro TTS model instance reset")
