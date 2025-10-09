# voice_assistant/greeting_generator.py

import logging
import random
import ollama
from voice_assistant.config import Config

# Fallback greetings in case LLM fails
FALLBACK_GREETINGS = [
    "Yeah, what's on your mind?",
    "Alright, shoot.",
    "What do you need?",
    "Go ahead.",
    "I'm listening.",
    "What's up?",
    "Hit me.",
    "Yeah?",
    "Speak.",
    "What is it?"
]

def generate_wake_greeting():
    """
    Generate a short, unique greeting for wake word responses.

    Uses the LLM to create greetings that match Howdy's personality:
    - George Carlin + Rodney Carrington blend
    - Dry, wry, occasionally sarcastic
    - Very brief (3-8 words max)
    - Direct and conversational

    Returns:
        str: A short greeting message
    """
    try:
        # Specialized prompt for wake word greetings
        greeting_prompt = {
            "role": "system",
            "content": (
                "You are Howdy, a voice assistant with George Carlin's wit and Rodney Carrington's edge. "
                "Generate ONE short, casual greeting response for when someone calls your name. "
                "Keep it VERY brief: 3-8 words maximum. "
                "Be conversational and direct. Mix up your style: "
                "Sometimes use questions ('What's the deal?', 'What is it?'), "
                "sometimes use commands ('Talk to me', 'Shoot', 'Go ahead'), "
                "sometimes use statements ('I'm listening', 'Yeah'), "
                "sometimes be wry or sarcastic. "
                "Examples of different styles (DO NOT repeat these exactly): "
                "'Yeah, what's on your mind?' (question), "
                "'Alright, shoot.' (command), "
                "'Talk to me.' (command), "
                "'I'm listening.' (statement), "
                "'What do you need?' (question), "
                "'Go ahead.' (command), "
                "'Lay it on me.' (command), "
                "'What's the deal?' (question), "
                "'Spit it out.' (command), "
                "'I'm all ears.' (statement), "
                "'What is it?' (question), "
                "'Yeah?' (one-word question). "
                "Generate ONE new greeting using a DIFFERENT sentence structure. "
                "Mix up your openers - avoid starting multiple greetings the same way. "
                "No quotes. No explanations."
            )
        }

        user_message = {
            "role": "user",
            "content": "Generate a new wake word greeting."
        }

        # Ollama options for very short, focused responses
        options = {
            "temperature": 0.9,  # Higher for more variety
            "num_predict": 20,   # Very short limit
            "top_k": 40,
            "top_p": 0.95,
            "repeat_penalty": 1.2,  # Discourage repetition
        }

        # Generate greeting
        response = ollama.chat(
            model=Config.OLLAMA_LLM,
            messages=[greeting_prompt, user_message],
            options=options,
        )

        greeting = response['message']['content'].strip()

        # Clean up the response
        # Remove quotes if present (including smart quotes)
        quotes_to_remove = '"\'""''\u201c\u201d'
        greeting = greeting.strip(quotes_to_remove)

        # Remove any explanatory text (take first sentence only)
        if '.' in greeting:
            greeting = greeting.split('.')[0].strip()

        # Strip quotes again after sentence split
        greeting = greeting.strip(quotes_to_remove)

        # Ensure it's not too long (word count check)
        words = greeting.split()
        if len(words) > 10:
            # If too long, fall back to a random fallback
            logging.warning(f"Generated greeting too long ({len(words)} words): '{greeting}'")
            greeting = random.choice(FALLBACK_GREETINGS)

        # Validate it's not empty
        if not greeting or len(greeting) < 2:
            logging.warning(f"Generated greeting invalid: '{greeting}'")
            greeting = random.choice(FALLBACK_GREETINGS)

        logging.info(f"Generated wake greeting: '{greeting}'")
        return greeting

    except Exception as e:
        logging.error(f"Failed to generate wake greeting: {e}")
        # Use random fallback
        fallback = random.choice(FALLBACK_GREETINGS)
        logging.info(f"Using fallback greeting: '{fallback}'")
        return fallback


def generate_greeting_with_cache():
    """
    Generate greeting with simple caching to avoid repeated calls.

    This is useful if we want to pre-generate some greetings
    or implement a cooldown system.

    Returns:
        str: A greeting message
    """
    # For now, just call generate_wake_greeting directly
    # Can be extended later with caching logic if needed
    return generate_wake_greeting()
