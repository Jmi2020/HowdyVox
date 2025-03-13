# voice_assistant/text_to_speech.py
import logging
import os
import soundfile as sf
import warnings
import nltk
import re
import threading
import queue
from voice_assistant.config import Config
from voice_assistant.kokoro_manager import KokoroManager

# Initialize NLTK if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Custom filter to suppress specific warnings
class WordsCountMismatchFilter(logging.Filter):
    def filter(self, record):
        return "words count mismatch" not in record.getMessage()

# Global queue for passing chunks between threads
chunk_queue = queue.Queue()

# Flag to signal generation is complete
generation_complete = threading.Event()

def text_to_speech(model: str, api_key:str, text:str, output_file_path:str, local_model_path:str=None):
    """
    Convert text to speech using persistent Kokoro ONNX model instance.
    Returns ONLY the first chunk immediately for fast response time, and puts the rest in a queue.
    
    Args:
    model (str): Should always be 'kokoro'.
    api_key (str): Not used for Kokoro.
    text (str): The text to convert to speech.
    output_file_path (str): The path to save the generated speech audio file.
    local_model_path (str): Optional custom voice model path.
    
    Returns:
    tuple: (bool, str) where bool indicates success and str is the path to the first chunk file
    """
    # Reset the generation complete flag
    generation_complete.clear()
    
    # Apply the filter to suppress specific warnings
    for handler in logging.root.handlers:
        handler.addFilter(WordsCountMismatchFilter())
    
    # Clear the queue of any pending chunks
    while not chunk_queue.empty():
        try:
            chunk_queue.get_nowait()
        except queue.Empty:
            break
    
    try:
        if model == "kokoro":
            # Create audio directory if it doesn't exist
            os.makedirs("temp/audio", exist_ok=True)
            
            # Get file extension
            file_base = "temp/audio/output"
            file_ext = os.path.splitext(output_file_path)[1]
            output_format = '.wav'  # Always use WAV for better compatibility
            
            # Split text into manageable chunks
            chunks = split_text_into_chunks(text, max_chars=200)
            logging.info(f"Split text into {len(chunks)} chunks")
            
            # Get the persistent Kokoro model instance
            kokoro = KokoroManager.get_instance(local_model_path=local_model_path)
            
            # Voice model to use
            voice_model = Config.KOKORO_VOICE
            
            # If no chunks, return failure
            if not chunks or not chunks[0].strip():
                return False, None
            
            # Generate ONLY the first chunk in the main thread
            first_chunk = chunks[0]
            first_chunk_file = f"{file_base}_chunk_0{output_format}"
            
            try:
                # Generate audio for first chunk
                samples, sample_rate = kokoro.create(
                    first_chunk, 
                    voice=voice_model, 
                    speed=1.0, 
                    lang="en-us"
                )
                
                # Save the audio file
                sf.write(first_chunk_file, samples, sample_rate)
                logging.info(f"Generated chunk 1/{len(chunks)}: {first_chunk_file}")
                
                # Start a background thread to generate the rest of the chunks
                def generate_remaining_chunks():
                    try:
                        # Process each remaining chunk
                        for i, chunk in enumerate(chunks[1:], start=1):
                            if not chunk.strip():
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
                                
                                # Save the audio file
                                sf.write(chunk_file, samples, sample_rate)
                                
                                # Add this chunk to the queue
                                chunk_queue.put(chunk_file)
                                logging.info(f"Generated chunk {i+1}/{len(chunks)}: {chunk_file}")
                                
                            except Exception as e:
                                logging.error(f"Error generating chunk {i+1}: {str(e)}")
                    except Exception as e:
                        logging.error(f"Error in background generation: {str(e)}")
                    finally:
                        # Signal that generation is complete
                        generation_complete.set()
                
                # Start the background thread if there are more chunks
                if len(chunks) > 1:
                    thread = threading.Thread(target=generate_remaining_chunks)
                    thread.daemon = True
                    thread.start()
                else:
                    # If only one chunk, signal completion
                    generation_complete.set()
                
                # Return success and the path to the first chunk
                return True, first_chunk_file
            
            except Exception as e:
                logging.error(f"Error generating first chunk: {str(e)}")
                generation_complete.set()
                return False, None
                
        else:
            raise ValueError("Only Kokoro is supported for text-to-speech")
            
    except Exception as e:
        logging.error(f"Failed to convert text to speech: {e}")
        generation_complete.set()
        return False, None

def get_next_chunk():
    """
    Get the next chunk from the queue.
    
    Returns:
    str or None: Path to the next chunk file, or None if no more chunks
    """
    try:
        if not chunk_queue.empty():
            return chunk_queue.get_nowait()
        elif generation_complete.is_set():
            return None
        else:
            # Wait for a chunk to be added to the queue, with a timeout
            try:
                return chunk_queue.get(timeout=0.5)
            except queue.Empty:
                return None
    except:
        return None

def split_text_into_chunks(text, max_chars=200):
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