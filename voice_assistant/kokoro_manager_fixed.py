# voice_assistant/kokoro_manager.py

import os
import logging
import warnings
import platform
import sys
from dotenv import load_dotenv

# Load environment variables to check for optimizations
load_dotenv()

# Set up logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use onnxruntime-silicon on Apple Silicon Macs
is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

# Try different import strategies for ONNX Runtime
ort = None

if is_apple_silicon:
    try:
        # On Apple Silicon, try to use the optimized version first
        import onnxruntime_silicon as ort
        logging.info("Using onnxruntime-silicon for Apple Silicon optimizations")
    except ImportError:
        try:
            # Fall back to regular onnxruntime if silicon version isn't available
            import onnxruntime as ort
            logging.info("onnxruntime-silicon not found, falling back to standard onnxruntime")
        except ImportError:
            logging.error("No ONNX Runtime available. Please install onnxruntime or onnxruntime-silicon")
            sys.exit(1)
else:
    # On non-Apple Silicon platforms, use standard onnxruntime
    try:
        import onnxruntime as ort
        logging.info("Using standard onnxruntime on non-Apple Silicon platform")
    except ImportError:
        logging.error("ONNX Runtime not available. Please install onnxruntime")
        sys.exit(1)

# Verify that InferenceSession is available
if not hasattr(ort, 'InferenceSession'):
    logging.error("ONNX Runtime is missing InferenceSession. Your installation may be corrupted or incomplete.")
    logging.error("Please reinstall with: pip uninstall -y onnxruntime onnxruntime-silicon && pip install onnxruntime==1.17.0")
    sys.exit(1)

# At this point, we should have a working ONNX Runtime with InferenceSession
try:
    from kokoro_onnx import Kokoro
except ImportError:
    logging.error("kokoro_onnx is not installed. Please install with: pip install kokoro-onnx==0.4.8")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error importing Kokoro TTS engine: {e}")
    logging.error("This might indicate an incompatibility between the installed onnxruntime and kokoro-onnx.")
    logging.error("Please try reinstalling: pip uninstall -y kokoro-onnx && pip install kokoro-onnx==0.4.8")
    sys.exit(1)

class KokoroManager:
    """
    Singleton class to manage a persistent instance of the Kokoro TTS model.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, model_path=None, voices_path=None, local_model_path=None):
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
            
            # Use the models directory as default location
            if model_path is None:
                model_path = os.path.join("models", "kokoro-v1.0.onnx")
            
            if voices_path is None:
                voices_path = os.path.join("models", "voices-v1.0.bin")
            
            # If custom model paths are provided, use them
            if local_model_path:
                if os.path.isdir(local_model_path):
                    model_path = os.path.join(local_model_path, "kokoro-v1.0.onnx")
                    voices_path = os.path.join(local_model_path, "voices-v1.0.bin")
                else:
                    model_path = local_model_path
            
            # Verify model files exist
            if not os.path.exists(model_path):
                logging.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
            if not os.path.exists(voices_path):
                logging.error(f"Voices file not found at: {voices_path}")
                raise FileNotFoundError(f"Voices file not found at: {voices_path}")
            
            logging.info(f"Loading Kokoro model from: {model_path}")
            logging.info(f"Loading Kokoro voices from: {voices_path}")
            
            # Configure ONNX Runtime using environment variables
            # Set thread count for better performance
            os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() - 1))
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision if supported
            
            # Set execution provider based on platform
            system = platform.system()
            if system == "Darwin":  # macOS
                # Check for Apple Silicon and use CoreML
                if platform.machine() == "arm64":
                    logging.info("Detected Apple Silicon - setting CoreML provider")
                    os.environ["ORT_COREML_ALLOWED"] = "1"
                    
                    # Check if we're using the silicon-specific package
                    if 'onnxruntime_silicon' in sys.modules:
                        logging.info("Using onnxruntime-silicon with CoreML optimizations")
                    else:
                        # For standard onnxruntime
                        if hasattr(ort, 'get_available_providers'):
                            providers = ort.get_available_providers()
                            if "CoreMLExecutionProvider" in providers:
                                logging.info("CoreML provider available")
                                # ORT will automatically use CoreML if available
                            else:
                                logging.info("CoreML provider not available, will use CPU")
                        else:
                            logging.info("Cannot check for CoreML provider, assuming CPU only")
                else:
                    logging.info("Using CPU execution provider on Intel Mac")
            elif system == "Windows":
                logging.info("On Windows - trying CUDA/DirectML providers if available")
                # ONNX Runtime will automatically use the best available provider
            else:  # Linux or others
                logging.info("On Linux or other OS - trying CUDA provider if available")
            
            # Log available providers if the function exists
            if hasattr(ort, 'get_available_providers'):
                available_providers = ort.get_available_providers()
                logging.info(f"Available ONNX Runtime providers: {available_providers}")
            else:
                logging.info("Unable to query available providers (function not available)")
            
            # Initialize the Kokoro model and suppress warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    logging.info("Initializing Kokoro model...")
                    
                    # Initialize Kokoro with the model and voices paths
                    cls._instance = Kokoro(model_path, voices_path)
                    
                    # Log the successful initialization
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