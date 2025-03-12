# voice_assistant/text_to_speech.py
import logging
import subprocess
import os
import soundfile as sf
import warnings
from kokoro_onnx import Kokoro
from voice_assistant.config import Config

# Custom filter to suppress specific warnings
class WordsCountMismatchFilter(logging.Filter):
    def filter(self, record):
        return "words count mismatch" not in record.getMessage()

def text_to_speech(model: str, api_key:str, text:str, output_file_path:str, local_model_path:str=None):
    """
    Convert text to speech using Kokoro ONNX model.
    
    Args:
    model (str): Should always be 'kokoro'.
    api_key (str): Not used for Kokoro.
    text (str): The text to convert to speech.
    output_file_path (str): The path to save the generated speech audio file.
    local_model_path (str): Optional custom voice model path.
    """
    
    # Apply the filter to suppress words count mismatch warnings
    for handler in logging.root.handlers:
        handler.addFilter(WordsCountMismatchFilter())
    
    try:
        # Delete existing output file if it exists
        if os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
                logging.info(f"Removed existing file: {output_file_path}")
            except Exception as e:
                logging.warning(f"Could not remove existing file {output_file_path}: {e}")
        
        if model == "kokoro":
            # Ensure the output path is accessible
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Get file extension
            _, file_ext = os.path.splitext(output_file_path)
            
            # Use the output path directly without conversion
            # If the requested output is MP3 but we want to skip conversion, use WAV instead
            if file_ext.lower() == '.mp3':
                # Change the extension to .wav
                wav_output_path = os.path.splitext(output_file_path)[0] + '.wav'
            else:
                wav_output_path = output_file_path
                
            # Use kokoro_onnx directly instead of subprocess
            voice_model = Config.KOKORO_VOICE
            logging.info(f"Generating speech using kokoro_onnx with voice: {voice_model}")
            
            try:
                # Initialize Kokoro model
                model_path = "kokoro-v1.0.onnx"
                voices_path = "voices-v1.0.bin"
                
                # If custom model paths are provided, use them
                if local_model_path:
                    if os.path.isdir(local_model_path):
                        model_path = os.path.join(local_model_path, "kokoro-v1.0.onnx")
                        voices_path = os.path.join(local_model_path, "voices-v1.0.bin")
                    else:
                        model_path = local_model_path
                
                # Initialize the Kokoro model and suppress warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    kokoro = Kokoro(model_path, voices_path)
                    
                    # Generate audio
                    samples, sample_rate = kokoro.create(
                        text, 
                        voice=voice_model, 
                        speed=1.0, 
                        lang="en-us"
                    )
                
                # Save the audio file
                sf.write(wav_output_path, samples, sample_rate)
                
                # If the output path was supposed to be an MP3 but we created a WAV,
                # return the WAV path instead, but don't convert it
                if wav_output_path != output_file_path:
                    logging.info(f"Using WAV file directly: {wav_output_path}")
                    # Update the output_file_path to point to the WAV file
                    # This will be returned to the calling function
                    output_file_path = wav_output_path
                    
                logging.info(f"Kokoro ONNX successfully generated audio file at {wav_output_path}")
            except Exception as e:
                logging.error(f"Kokoro ONNX processing failed: {str(e)}")
                raise
        else:
            raise ValueError("Only Kokoro is supported for text-to-speech")
        
        # Verify the output file exists and has content
        if not os.path.exists(output_file_path):
            raise FileNotFoundError(f"Output file was not created at {output_file_path}")
            
        file_size = os.path.getsize(output_file_path)
        if file_size < 100:  # Suspiciously small file
            logging.warning(f"Warning: Generated audio file is very small ({file_size} bytes)")
            
        return True
        
    except Exception as e:
        logging.error(f"Failed to convert text to speech: {e}")
        return False