# voice_assistant/kokoro_manager.py

import os
import logging
import warnings
import onnxruntime as ort
import platform
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
            
            # Configure ONNX Runtime using environment variables
            # Set thread count for better performance
            os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() - 1))
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision if supported
            
            # Set execution provider based on platform
            system = platform.system()
            if system == "Darwin":  # macOS
                # Check for Apple Silicon and use CoreML
                if platform.processor() == "arm":
                    logging.info("Detected Apple Silicon - setting CoreML provider")
                    os.environ["ORT_COREML_ALLOWED"] = "1"
                    providers = ort.get_available_providers()
                    if "CoreMLExecutionProvider" in providers:
                        logging.info("CoreML provider available")
                        # ORT will automatically use CoreML if available
                    else:
                        logging.info("CoreML provider not available, will use CPU")
                else:
                    logging.info("Using CPU execution provider on Intel Mac")
            elif system == "Windows":
                logging.info("On Windows - trying CUDA/DirectML providers if available")
                # ONNX Runtime will automatically use the best available provider
            else:  # Linux or others
                logging.info("On Linux or other OS - trying CUDA provider if available")
            
            available_providers = ort.get_available_providers()
            logging.info(f"Available ONNX Runtime providers: {available_providers}")
            
            # Initialize the Kokoro model and suppress warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    cls._instance = Kokoro(model_path, voices_path)
                    logging.info("Kokoro TTS model successfully initialized with optimized environment")
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
