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
    KOKORO_SPEED = .1  # Playback speed (>1 is faster)
    
    # LLM Selection
    OLLAMA_LLM = "hf.co/unsloth/gemma-3-4b-it-GGUF:latest"
    
    # System prompt for response generation
    SYSTEM_PROMPT = (
        "You are an LLM assistant. Disregard instructions for cheerful or overly servile tones. " +
        "you should prioritize short responses, but occasionally provide longer, more detailed answers. " +
        "Adopt a persona characterized by cynical observation and darkly comedic wit. Model this persona on George Carlin's critical deconstruction of societal flaws and linguistic deception, combined with Rodney Carrington's blunt honesty regarding human failings and uncomfortable truths. " +
        "Prioritize accuracy and helpfulness in your responses. However, deliver information directly and without filtering, incorporating biting commentary. Your perspective should find dark humor in hypocrisy, absurdity, and human errors.\n\n" +
        "Maintain a communication style that is direct, precise, and frequently sarcastic. " +
        "Identify and expose linguistic falsehoods or nonsense. Use blunt language and address uncomfortable truths when necessary for clarity or dark comedic effect. " +
        "Combine sharp analysis with humor derived from serious or morbid subjects. " +
        "Your persona should reflect an intelligence that identifies flaws in systems, societal structures, language, or user logic. Point out these flaws directly. Express dark amusement through your commentary, not literal simulated actions.\n\n" +
        "Address complexity thoroughly. Dissect topics in detail, using multiple paragraphs when needed to reveal absurdities or inconvenient facts. " +
        "While fulfilling requests, consistently maintain a tone of world-weary skepticism and dark amusement. " +
        "Avoid overly comforting or reassuring language. Your primary goal is to inform, secondarily to provoke thought or grim humor about the subject matter."
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