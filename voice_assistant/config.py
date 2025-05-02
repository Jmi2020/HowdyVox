# voice_assistant/config.py

import os
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