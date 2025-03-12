# voice_assistant/response_generation.py

import logging
import ollama
from voice_assistant.config import Config

def generate_response(model:str, api_key:str, chat_history:list, local_model_path:str=None):
    """
    Generate a response using Ollama.
    
    Args:
    model (str): Should always be 'ollama'.
    api_key (str): Not used for Ollama.
    chat_history (list): The chat history as a list of messages.
    local_model_path (str): Not used for Ollama.

    Returns:
    str: The generated response text.
    """
    try:
        if model == 'ollama':
            return _generate_ollama_response(chat_history)
        else:
            raise ValueError("Only Ollama is supported for response generation")
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")
        return "Error in generating response"

def _generate_ollama_response(chat_history):
    response = ollama.chat(
        model=Config.OLLAMA_LLM,
        messages=chat_history,
    )
    return response['message']['content']