# voice_assistant/response_generation.py

import logging
import ollama
from voice_assistant.config import Config

def preload_ollama_model():
    """
    Preload the Ollama model into memory by making a simple test request.
    This ensures the model is ready for use when the first real request comes in.
    
    Returns:
    bool: True if the model was successfully preloaded, False otherwise.
    """
    try:
        logging.info(f"Preloading Ollama model: {Config.OLLAMA_LLM}")
        
        # Make a simple test request to load the model into memory
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        # Use minimal options for the preload test
        test_options = {
            "temperature": 0.1,
            "num_predict": 5,  # Very short response to minimize wait time
        }
        
        response = ollama.chat(
            model=Config.OLLAMA_LLM,
            messages=test_messages,
            options=test_options,
        )
        
        logging.info(f"Ollama model {Config.OLLAMA_LLM} preloaded successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to preload Ollama model: {e}")
        return False

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
    """
    Generate a response using Ollama with custom parameters.
    
    Args:
    chat_history (list): The chat history as a list of messages.
    
    Returns:
    str: The generated response text.
    """
    # Define options for the Ollama model
    options = {
        # Control randomness - lower values make responses more focused and deterministic
        "temperature": 0.7,
        
        # Limit the length of responses (in tokens) - increased from 100 to 1000 to allow for longer stories
        "num_predict": 1000,
        
        # Make the model more concise
        "top_k": 40,
        "top_p": 0.9,
        
        # Prevent repetitive responses
        "repeat_penalty": 1.1
    }
    
    # Check if the chat history already has a system message
    has_system_message = False
    for message in chat_history:
        if message.get("role") == "system":
            has_system_message = True
            # Update the existing system message with our system prompt from config
            message["content"] = Config.SYSTEM_PROMPT
            break
    
    # If there's no system message, add one from config
    if not has_system_message:
        chat_history.insert(0, {
            "role": "system", 
            "content": Config.SYSTEM_PROMPT
        })
    
    # Generate response with the configured options
    response = ollama.chat(
        model=Config.OLLAMA_LLM,
        messages=chat_history,
        options=options,
    )
    
    return response['message']['content']