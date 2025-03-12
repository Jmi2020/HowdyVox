# voice_assistant/text_to_speech.py
import logging
import subprocess
import os

from voice_assistant.config import Config

def text_to_speech(model: str, api_key:str, text:str, output_file_path:str, local_model_path:str=None):
    """
    Convert text to speech using Kokoro.
    
    Args:
    model (str): Should always be 'kokoro'.
    api_key (str): Not used for Kokoro.
    text (str): The text to convert to speech.
    output_file_path (str): The path to save the generated speech audio file.
    local_model_path (str): Optional custom voice model path.
    """
    
    try:
        # Delete existing output file if it exists
        if os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
                logging.info(f"Removed existing file: {output_file_path}")
            except Exception as e:
                logging.warning(f"Could not remove existing file {output_file_path}: {e}")
        
        if model == "kokoro":
            # Define the kokoro command
            voice_model = Config.KOKORO_VOICE  # Use the voice from config
            if local_model_path:
                voice_model = local_model_path
                
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
                
            # Run kokoro as a subprocess
            cmd = ["kokoro", "-m", voice_model, "-t", text, "-o", wav_output_path]
            logging.info(f"Running kokoro command: {' '.join(cmd)}")
            
            try:
                process = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    check=True
                )
                
                # Check if the file was created
                if not os.path.exists(wav_output_path):
                    raise FileNotFoundError(f"Kokoro failed to generate audio file at {wav_output_path}")
                
                # If the output path was supposed to be an MP3 but we created a WAV,
                # return the WAV path instead, but don't convert it
                if wav_output_path != output_file_path:
                    logging.info(f"Using WAV file directly: {wav_output_path}")
                    # Update the output_file_path to point to the WAV file
                    # This will be returned to the calling function
                    output_file_path = wav_output_path
                    
                logging.info(f"Kokoro TTS successfully generated audio file at {wav_output_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Kokoro command failed: {e.stderr}")
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