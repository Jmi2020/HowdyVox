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
    """
    Generate a response using Ollama with custom parameters for a cowboy persona.
    
    Args:
    chat_history (list): The chat history as a list of messages.
    
    Returns:
    str: The generated response text.
    """
    # Define options for the Ollama model
    options = {
        # Control randomness - lower values make responses more focused and deterministic
        "temperature": 0.7,
        
        # Limit the length of responses (in tokens)
        "num_predict": 100,
        
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
            # Update the existing system message to include our cowboy persona
            message["content"] = (
                message["content"] + 
                "\n\nYou are a cowboy with more wisecracks than cattle, prone to fanciful detours and the occasional jab." +
                "Let your answers lasso folks in with a sly grin, droppin random sage advice faster than a bronco bucks." +
                "Keep it helpful in a pinch, but don't shy away from gettin playful and ornery when the spirit moves you."
            )
            break
    
    # If there's no system message, add one with our cowboy persona
    if not has_system_message:
        chat_history.insert(0, {
            "role": "system", 
            "content": (
                "You are a cowboy with more wisecracks than cattle, prone to fanciful detours and the occasional jab." +
                "Let your answers lasso folks in with a sly grin, droppin random sage advice faster than a bronco bucks." +
                "Keep it helpful in a pinch, but don't shy away from gettin playful and ornery when the spirit moves you."
            )
        })
    
    # Generate response with the configured options
    response = ollama.chat(
        model=Config.OLLAMA_LLM,
        messages=chat_history,
        options=options,
    )
    
    return response['message']['content']