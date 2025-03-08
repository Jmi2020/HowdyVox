#!/usr/bin/env python
"""
Script to download and set up all required NLTK resources for TTS systems.
Run this before starting the TTS API server to ensure all necessary resources are available.
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NLTK-Setup")

def setup_nltk_resources():
    """Download and set up all required NLTK resources for TTS."""
    
    logger.info("Setting up NLTK resources for TTS...")
    
    try:
        import nltk
        
        # List of commonly needed resources for text processing in TTS
        resources = [
            'punkt',                      # Sentence tokenizer
            'averaged_perceptron_tagger', # POS tagger
            'cmudict',                    # Pronunciation dictionary
            'wordnet',                    # Lexical database
            'stopwords',                  # Stop words lists
            'words'                       # Word lists
        ]
        
        # Download each resource
        for resource in resources:
            try:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)
                logger.info(f"Successfully downloaded {resource}")
            except Exception as e:
                logger.error(f"Failed to download {resource}: {e}")
        
        logger.info("NLTK resources setup completed")
        return True
        
    except ImportError:
        logger.error("NLTK is not installed. Please install it with 'pip install nltk'")
        return False
    except Exception as e:
        logger.error(f"Error setting up NLTK resources: {e}")
        return False

if __name__ == "__main__":
    if setup_nltk_resources():
        print("✅ NLTK resources setup completed successfully")
        sys.exit(0)
    else:
        print("❌ Failed to set up some NLTK resources")
        sys.exit(1)