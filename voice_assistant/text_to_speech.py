# voice_assistant/text_to_speech.py
import logging
import subprocess
import os
import soundfile as sf
import warnings
import nltk
import re
import queue
import threading
from kokoro_onnx import Kokoro
from voice_assistant.config import Config

# Initialize NLTK if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Custom filter to suppress specific warnings
class WordsCountMismatchFilter(logging.Filter):
    def filter(self, record):
        return "words count mismatch" not in record.getMessage()

def text_to_speech(model: str, api_key:str, text:str, output_file_path:str, local_model_path:str=None, stream=True):
    """
    Convert text to speech using Kokoro ONNX model with text chunking for long responses.
    
    Args:
    model (str): Should always be 'kokoro'.
    api_key (str): Not used for Kokoro.
    text (str): The text to convert to speech.
    output_file_path (str): The path to save the generated speech audio file.
    local_model_path (str): Optional custom voice model path.
    stream (bool): If True, enables streaming mode to yield chunks as they're generated.
    
    Returns:
    tuple: (bool, list or queue) where bool indicates success and second value is either:
           - A list of chunk file paths if stream=False
           - A queue that will receive chunk paths as they're generated if stream=True
    """
    
    # Apply the filter to suppress words count mismatch warnings
    for handler in logging.root.handlers:
        handler.addFilter(WordsCountMismatchFilter())
    
    # If streaming is enabled, use a queue to manage chunks
    chunk_queue = queue.Queue() if stream else None
    
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
            file_base, file_ext = os.path.splitext(output_file_path)
            
            # Use WAV format
            if file_ext.lower() == '.mp3':
                # Change the extension to .wav
                output_format = '.wav'
            else:
                output_format = file_ext
            
            # Split text into manageable chunks to avoid phoneme limit
            chunks = split_text_into_chunks(text)
            logging.info(f"Split text into {len(chunks)} chunks")
            
            if stream:
                # Start a thread to generate chunks in the background
                generator_thread = threading.Thread(
                    target=_generate_audio_chunks_worker,
                    args=(chunks, file_base, output_format, model, local_model_path, chunk_queue)
                )
                generator_thread.daemon = True
                generator_thread.start()
                
                # Return the chunk queue for the main thread to consume from
                return True, chunk_queue
            else:
                # Traditional approach - generate all chunks and return the list
                chunk_files = _generate_audio_chunks(chunks, file_base, output_format, model, local_model_path)
                
                if chunk_files:
                    return True, chunk_files
                else:
                    logging.error("Failed to generate any audio chunks")
                    return False, []
        else:
            raise ValueError("Only Kokoro is supported for text-to-speech")
        
    except Exception as e:
        logging.error(f"Failed to convert text to speech: {e}")
        if stream:
            # Signal that generation has completed with an error
            chunk_queue.put(None)
        return False, [] if not stream else chunk_queue

def _generate_audio_chunks_worker(chunks, file_base, output_format, model, local_model_path, chunk_queue):
    """
    Worker function that generates audio chunks and puts them in a queue.
    
    Args:
    chunks (list): List of text chunks to process.
    file_base (str): Base filename for output files.
    output_format (str): File extension to use.
    model (str): Model name.
    local_model_path (str): Custom model path.
    chunk_queue (queue.Queue): Queue to put generated chunk paths into.
    """
    try:
        # Initialize the Kokoro model
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
        
        # Voice model to use
        voice_model = Config.KOKORO_VOICE
        
        # Generate audio for each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Skip empty chunks
                continue
                
            chunk_file = f"{file_base}_chunk_{i}{output_format}"
            
            try:
                # Generate audio for this chunk
                samples, sample_rate = kokoro.create(
                    chunk, 
                    voice=voice_model, 
                    speed=1.0, 
                    lang="en-us"
                )
                
                # Save the audio file for this chunk
                sf.write(chunk_file, samples, sample_rate)
                logging.info(f"Generated chunk {i+1}/{len(chunks)}: {chunk_file}")
                
                # Add the chunk to the queue for immediate playback
                chunk_queue.put(chunk_file)
                
            except Exception as e:
                logging.error(f"Error generating audio for chunk {i+1}: {str(e)}")
                logging.error(f"Problematic text: {chunk}")
                # Try with an even smaller chunk
                subchunks = re.split(r'[.!?]+', chunk)
                for j, subchunk in enumerate(subchunks):
                    if not subchunk.strip():
                        continue
                    subchunk = subchunk.strip() + "."
                    subchunk_file = f"{file_base}_chunk_{i}_{j}{output_format}"
                    try:
                        samples, sample_rate = kokoro.create(
                            subchunk, 
                            voice=voice_model, 
                            speed=1.0, 
                            lang="en-us"
                        )
                        sf.write(subchunk_file, samples, sample_rate)
                        chunk_queue.put(subchunk_file)
                        logging.info(f"Generated subchunk {j+1}: {subchunk_file}")
                    except Exception as e2:
                        logging.error(f"Failed on subchunk too: {str(e2)}")
        
        # Signal that all chunks have been generated
        chunk_queue.put(None)  # None signals end of generation
            
    except Exception as e:
        logging.error(f"Worker thread error: {str(e)}")
        # Signal error
        chunk_queue.put(None)

def _generate_audio_chunks(chunks, file_base, output_format, model, local_model_path):
    """
    Generate audio for all chunks sequentially.
    
    Args:
    chunks (list): List of text chunks to process.
    file_base (str): Base filename for output files.
    output_format (str): File extension to use.
    model (str): Model name.
    local_model_path (str): Custom model path.
    
    Returns:
    list: List of generated audio file paths.
    """
    # Initialize the Kokoro model
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
    
    # Voice model to use
    voice_model = Config.KOKORO_VOICE
    logging.info(f"Generating speech using kokoro_onnx with voice: {voice_model}")
    
    # Generate audio for each chunk
    chunk_files = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():  # Skip empty chunks
            continue
            
        chunk_file = f"{file_base}_chunk_{i}{output_format}"
        
        try:
            # Generate audio for this chunk
            samples, sample_rate = kokoro.create(
                chunk, 
                voice=voice_model, 
                speed=1.0, 
                lang="en-us"
            )
            
            # Save the audio file for this chunk
            sf.write(chunk_file, samples, sample_rate)
            chunk_files.append(chunk_file)
            logging.info(f"Generated chunk {i+1}/{len(chunks)}: {chunk_file}")
            
        except Exception as e:
            logging.error(f"Error generating audio for chunk {i+1}: {str(e)}")
            logging.error(f"Problematic text: {chunk}")
            # Try with an even smaller chunk
            subchunks = re.split(r'[.!?]+', chunk)
            for j, subchunk in enumerate(subchunks):
                if not subchunk.strip():
                    continue
                subchunk = subchunk.strip() + "."
                subchunk_file = f"{file_base}_chunk_{i}_{j}{output_format}"
                try:
                    samples, sample_rate = kokoro.create(
                        subchunk, 
                        voice=voice_model, 
                        speed=1.0, 
                        lang="en-us"
                    )
                    sf.write(subchunk_file, samples, sample_rate)
                    chunk_files.append(subchunk_file)
                    logging.info(f"Generated subchunk {j+1}: {subchunk_file}")
                except Exception as e2:
                    logging.error(f"Failed on subchunk too: {str(e2)}")
    
    return chunk_files

def split_text_into_chunks(text, max_chars=250):
    """
    Split text into smaller chunks to avoid phoneme limit issues.
    
    Args:
    text (str): The text to split.
    max_chars (int): Maximum characters per chunk.
    
    Returns:
    list: List of text chunks.
    """
    # First try to split by sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If any chunk is still too long, split it further
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            # Split by punctuation first
            subchunks = re.split(r'[,;:]', chunk)
            subchunk_text = ""
            
            for subchunk in subchunks:
                if len(subchunk_text) + len(subchunk) > max_chars and subchunk_text:
                    final_chunks.append(subchunk_text.strip())
                    subchunk_text = subchunk
                else:
                    if subchunk_text:
                        subchunk_text += ", " + subchunk.strip()
                    else:
                        subchunk_text = subchunk.strip()
            
            if subchunk_text:
                final_chunks.append(subchunk_text)
        else:
            final_chunks.append(chunk)
    
    return final_chunks