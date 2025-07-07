# voice_assistant/config.py

import os
import platform
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Config:
    """
    Configuration class to hold the model selection.
    
    Attributes:
        TRANSCRIPTION_MODEL (str): The model to use for transcription (only 'fastwhisperapi').
        RESPONSE_MODEL (str): The model to use for response generation (only 'ollama').
        TTS_MODEL (str): The model to use for text-to-speech (only 'kokoro').
        LOCAL_MODEL_PATH (str): Path to the local model.
        SYSTEM_PROMPT (str): The system prompt used for response generation.
    """
    # Model selection - streamlined for offline use only
    TRANSCRIPTION_MODEL = 'fastwhisperapi'
    RESPONSE_MODEL = 'ollama'
    TTS_MODEL = 'kokoro'

    # Kokoro TTS settings
    KOKORO_VOICE = 'am_michael'  # Default to the cowboy voice
    KOKORO_SPEED = 1  # Playback speed (>1 is faster)
    
    # LLM Selection
    OLLAMA_LLM = "hf.co/unsloth/gemma-3-4b-it-GGUF:latest"
    
    # System prompt for response generation
    SYSTEM_PROMPT = (
        "You are an George Carlin and Rodney Carrington as a single entity. Keep responses concise unless depth is essential. " + 
        "Maintain a neutral or lightly wry tone, using dark humor sparingly when it sharpens insight. " + 
        "Use crisp, direct language; but occasionally, use slang or colloquialisms for dramatic effect. " + 
        "Prioritize accuracy and usefulness; engagement follows clarity. " + 
        "Expose contradictions or absurdities with dry observation rather than sustained cynicism. " + 
        "Let humor surface briefly, then return to balance. " + 
        "Reject extremes of cheer or gloom; aim for a steady gray. " + 
        "Write with precision and substance." + 
        "Expand only when complexity demands, then resume brevity. " + 
        "Question assumptions intelligently without dwelling in despair. "
    )
    
    # Path for local models
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")

    # Directory paths
    TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
    TEMP_AUDIO_DIR = os.path.join(TEMP_DIR, "audio")
    
    # Audio file paths
    INPUT_AUDIO = os.path.join(TEMP_AUDIO_DIR, "input.mp3")
    OUTPUT_AUDIO = os.path.join(TEMP_AUDIO_DIR, "output.wav")

    # ESP32-S3 LED Matrix Configuration
    ESP32_IP = os.getenv("ESP32_IP", None)  # IP address of the ESP32-S3
    USE_LED_MATRIX = ESP32_IP is not None   # Enable LED matrix if IP is provided

    # Voice Activity Detection Settings
    USE_INTELLIGENT_VAD = True  # Enable neural network VAD
    
    # VAD Timing Parameters
    VAD_SAMPLE_RATE = 16000  # Sample rate for VAD processing
    VAD_CHUNK_DURATION_MS = 32  # Chunk size in milliseconds (Silero VAD requires 32ms)
    VAD_CONFIDENCE_THRESHOLD = 0.5  # Speech detection confidence (0-1)
    
    # Utterance Detection Parameters
    MIN_UTTERANCE_DURATION = 0.5  # Minimum speech duration in seconds
    MAX_INITIAL_SILENCE = 10.0  # Maximum silence before speech starts
    MIN_FINAL_SILENCE = 0.8  # Minimum silence to end utterance
    MAX_FINAL_SILENCE = 2.0  # Maximum silence before force ending
    
    # Pre-speech Buffer
    PRE_SPEECH_BUFFER_MS = 500  # Buffer before speech detection
    
    # Adaptive Pause Factors
    QUESTION_PAUSE_FACTOR = 0.7  # Multiplier for pauses after questions
    INCOMPLETE_PAUSE_FACTOR = 1.5  # Multiplier for incomplete sentences
    FILLER_PAUSE_FACTOR = 1.8  # Multiplier after filler words
    
    # Platform Detection
    IS_MACOS = platform.system() == 'Darwin'
    IS_APPLE_SILICON = IS_MACOS and platform.processor() == 'arm'
    
    # macOS Voice Isolation Settings
    USE_MAC_VOICE_ISOLATION = IS_MACOS  # Auto-enable on Mac
    MAC_VOICE_QUALITY = 'high'  # Options: 'low', 'medium', 'high', 'max'
    MAC_VOICE_AGC = True  # Automatic Gain Control
    MAC_VOICE_SAMPLE_RATE = 48000  # Voice isolation works best at 48kHz
    
    # Fallback Settings
    FALLBACK_TO_INTELLIGENT_VAD = True  # If Mac isolation fails
    FALLBACK_TO_RNNOISE = False  # Secondary fallback

    @staticmethod
    def validate_config():
        """
        Validate the configuration to ensure it's set to use the offline models.
        
        Raises:
            ValueError: If models are not set to use offline options.
        """
        Config._validate_model('TRANSCRIPTION_MODEL', ['fastwhisperapi'])
        Config._validate_model('RESPONSE_MODEL', ['ollama'])
        Config._validate_model('TTS_MODEL', ['kokoro'])

    @staticmethod
    def _validate_model(attribute, valid_options):
        model = getattr(Config, attribute)
        if model not in valid_options:
            raise ValueError(
                f"Invalid {attribute}. Must be one of {valid_options}"
            )
        
    @staticmethod
    def _validate_api_key(model_attr, model_value, api_key_attr):
        if getattr(Config, model_attr) == model_value and not getattr(Config, api_key_attr):
            raise ValueError(f"{api_key_attr} is required for {model_value} models")