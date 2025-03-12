# voice_assistant/transcription.py

import logging
import requests
from colorama import Fore

fast_url = "http://localhost:8000"
checked_fastwhisperapi = False

def check_fastwhisperapi():
    """Check if the FastWhisper API is running."""
    global checked_fastwhisperapi, fast_url
    if not checked_fastwhisperapi:
        infopoint = f"{fast_url}/info"
        try:
            response = requests.get(infopoint)
            if response.status_code != 200:
                raise Exception("FastWhisperAPI is not running")
        except Exception:
            raise Exception("FastWhisperAPI is not running")
        checked_fastwhisperapi = True

def transcribe_audio(model, api_key, audio_file_path, local_model_path=None):
    """
    Transcribe an audio file using FastWhisperAPI.
    
    Args:
        model (str): Should always be 'fastwhisperapi'.
        api_key (str): Not used for FastWhisperAPI.
        audio_file_path (str): The path to the audio file to transcribe.
        local_model_path (str): Not used for FastWhisperAPI.

    Returns:
        str: The transcribed text.
    """
    try:
        if model == 'fastwhisperapi':
            return _transcribe_with_fastwhisperapi(audio_file_path)
        else:
            raise ValueError("Only FastWhisperAPI is supported for transcription")
    except Exception as e:
        logging.error(f"{Fore.RED}Failed to transcribe audio: {e}{Fore.RESET}")
        raise Exception("Error in transcribing audio")

def _transcribe_with_fastwhisperapi(audio_file_path):
    check_fastwhisperapi()
    endpoint = f"{fast_url}/v1/transcriptions"

    files = {'file': (audio_file_path, open(audio_file_path, 'rb'))}
    data = {
        'model': "base",
        'language': "en",
        'initial_prompt': None,
        'vad_filter': True,
    }
    headers = {'Authorization': 'Bearer dummy_api_key'}

    response = requests.post(endpoint, files=files, data=data, headers=headers)
    response_json = response.json()
    return response_json.get('text', 'No text found in the response.')