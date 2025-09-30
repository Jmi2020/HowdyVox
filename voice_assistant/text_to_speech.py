# voice_assistant/text_to_speech.py
import logging
import os
import warnings
import nltk
import re
import threading
import queue
import time
from voice_assistant.config import Config
from voice_assistant.kokoro_manager import KokoroManager

# Adaptive timing thresholds for chunking and buffering
ADAPTIVE_TIMING_THRESHOLDS = [
    {
        "length": 100,
        "max_chars": 150,
        "initial_delay": 0.05,
        "chunk_buffer_delay": 0.02
    },
    {
        "length": 300,
        "max_chars": 180,
        "initial_delay": 0.08,
        "chunk_buffer_delay": 0.05
    },
    {
        "length": 800,
        "max_chars": 200,
        "initial_delay": 0.12,
        "chunk_buffer_delay": 0.08
    },
    {
        "length": float("inf"),
        "max_chars": 220,
        "initial_delay": 0.15,
        "chunk_buffer_delay": 0.1
    }
]

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

def clean_text_for_tts(text):
    """
    Clean text before sending to TTS by removing formatting characters like asterisks.

    Args:
        text (str): The input text to clean

    Returns:
        str: The cleaned text suitable for TTS
    """
    # Remove markdown emphasis patterns (both *word* and **word**)
    cleaned_text = re.sub(r'\*+([^*]+)\*+', r'\1', text)

    # Also handle standalone asterisks that might remain
    cleaned_text = cleaned_text.replace('*', '')

    # Remove other potential markdown formatting
    cleaned_text = cleaned_text.replace('_', '')  # Remove underscores used for emphasis
    cleaned_text = cleaned_text.replace('`', '')  # Remove backticks used for code

    # Clean up any potential double spaces created by removing characters
    cleaned_text = ' '.join(cleaned_text.split())

    # Log original vs cleaned text if there was a change
    if cleaned_text != text:
        logging.info(f"Cleaned text for TTS - Removed formatting characters")
        logging.debug(f"Original: '{text}' -> Cleaned: '{cleaned_text}'")

    return cleaned_text


def encode_pcm_to_opus(samples, sample_rate, output_file):
    """
    Encode PCM audio samples to Opus format and save to file.

    Args:
        samples: NumPy array of PCM samples (float32, -1.0 to 1.0)
        sample_rate: Sample rate (should be 16000 for Kokoro)
        output_file: Path to save .opus file

    Returns:
        bool: True if successful, False if Opus encoding failed
    """
    try:
        import opuslib
        import numpy as np

        # Opus encoder configuration
        CHANNELS = 1
        BITRATE = 24000  # 24 kbps for speech

        # Calculate frame size based on actual sample rate (20ms worth of samples)
        FRAME_SIZE = int(sample_rate * 0.02)  # 20ms frame: 320 @ 16kHz, 480 @ 24kHz

        # Validate sample rate (Opus supports 8k, 12k, 16k, 24k, 48k)
        valid_rates = [8000, 12000, 16000, 24000, 48000]
        if sample_rate not in valid_rates:
            logging.warning(f"Sample rate {sample_rate} not optimal for Opus, closest supported rates: {valid_rates}")

        # Create Opus encoder
        encoder = opuslib.Encoder(sample_rate, CHANNELS, opuslib.APPLICATION_VOIP)
        encoder.bitrate = BITRATE

        # Convert float samples to 16-bit PCM bytes
        pcm_int16 = (samples * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        # Encode to Opus frames
        opus_frames = []
        offset = 0
        frame_size_bytes = FRAME_SIZE * 2  # 2 bytes per 16-bit sample

        while offset < len(pcm_bytes):
            # Extract one frame
            frame_end = min(offset + frame_size_bytes, len(pcm_bytes))
            pcm_frame = pcm_bytes[offset:frame_end]

            # Pad last frame if needed
            if len(pcm_frame) < frame_size_bytes:
                pcm_frame += b'\x00' * (frame_size_bytes - len(pcm_frame))

            # Encode frame
            opus_frame = encoder.encode(pcm_frame, FRAME_SIZE)
            opus_frames.append(opus_frame)
            offset = frame_end

        # Write Opus frames to file
        with open(output_file, 'wb') as f:
            for frame in opus_frames:
                # Write frame length as 2-byte header (for easier decoding later)
                f.write(len(frame).to_bytes(2, 'little'))
                f.write(frame)

        original_size = len(pcm_bytes)
        opus_size = sum(len(frame) for frame in opus_frames)
        compression_ratio = opus_size / original_size * 100

        logging.debug(f"Encoded Opus: {original_size} bytes PCM → {opus_size} bytes Opus ({compression_ratio:.1f}%)")
        return True

    except ImportError:
        logging.debug("opuslib not available, skipping Opus encoding")
        return False
    except Exception as e:
        logging.error(f"Failed to encode Opus: {e}")
        logging.error(f"  Sample rate: {sample_rate}, Channels: {CHANNELS}, Frame size: {FRAME_SIZE}")
        logging.error(f"  PCM data: {len(pcm_bytes) if 'pcm_bytes' in locals() else 'N/A'} bytes")
        logging.error(f"  Sample data type: {type(samples)}, shape: {samples.shape if hasattr(samples, 'shape') else 'N/A'}")
        return False


def text_to_speech(model: str, api_key:str, text:str, output_file_path:str, local_model_path:str=None):
    """
    Convert text to speech using persistent Kokoro ONNX model instance with adaptive buffering.
    Uses intelligent chunk sizing and pre-buffering to prevent stuttering on longer texts.
    
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
            
            # Clean the text to remove asterisks and other formatting
            cleaned_text = clean_text_for_tts(text)
            
            # Enhanced adaptive chunk sizing with improved timing using configurable thresholds
            text_length = len(cleaned_text)
            for threshold in ADAPTIVE_TIMING_THRESHOLDS:
                if text_length < threshold["length"]:
                    max_chars = threshold["max_chars"]
                    initial_delay = threshold["initial_delay"]
                    chunk_buffer_delay = threshold["chunk_buffer_delay"]
                    break

            # Split text into adaptively-sized chunks with enhanced buffering info
            chunks = split_text_into_chunks(cleaned_text, max_chars=max_chars)
            logging.info(f"Split text into {len(chunks)} chunks (adaptive size: {max_chars} chars, text length: {text_length})")
            logging.info(f"Using timing strategy: initial_delay={initial_delay:.3f}s, buffer_delay={chunk_buffer_delay:.3f}s")
            
            # Get the persistent Kokoro model instance
            kokoro = KokoroManager.get_instance(local_model_path=local_model_path)
            
            # Voice model to use
            voice_model = Config.KOKORO_VOICE
            
            # If no chunks, return failure
            if not chunks or not chunks[0].strip():
                return False, None
            
            # Generate ONLY the first chunk in the main thread with timing
            first_chunk = chunks[0]
            first_chunk_file = f"{file_base}_chunk_0.opus"  # Generate Opus directly, skip WAV

            try:
                # Track timing for first chunk generation
                first_chunk_start = time.time()

                # Generate audio for first chunk
                samples, sample_rate = kokoro.create(
                    first_chunk,
                    voice=voice_model,
                    speed=Config.KOKORO_SPEED,  # Use config value instead of hardcoded 1.0
                    lang="en-us"
                )

                # Encode directly to Opus (skip WAV generation)
                opus_encoded = encode_pcm_to_opus(samples, sample_rate, first_chunk_file)

                if not opus_encoded:
                    logging.error("Failed to encode Opus for first chunk - opuslib may not be installed")
                    generation_complete.set()
                    return False, None

                first_chunk_time = time.time() - first_chunk_start
                logging.info(f"Generated chunk 1/{len(chunks)}: {first_chunk_file} (took {first_chunk_time:.3f}s)")
                
                # Enhanced background thread with adaptive pre-buffering
                def generate_remaining_chunks():
                    try:
                        # Enhanced buffering strategy for longer texts
                        if len(chunks) > 3:
                            logging.info(f"Using enhanced buffering strategy for {len(chunks)} chunks")
                            # For very long texts, add extra stabilization between chunks
                            inter_chunk_stabilization = chunk_buffer_delay
                        else:
                            inter_chunk_stabilization = 0.02  # Minimal delay for shorter texts
                        
                        # Process each remaining chunk with enhanced timing
                        for i, chunk in enumerate(chunks[1:], start=1):
                            if not chunk.strip():
                                continue

                            chunk_file = f"{file_base}_chunk_{i}.opus"  # Generate Opus directly, skip WAV

                            try:
                                chunk_start_time = time.time()

                                # Apply inter-chunk stabilization delay for better audio quality
                                if i > 1 and inter_chunk_stabilization > 0:
                                    time.sleep(inter_chunk_stabilization)

                                # Generate audio for this chunk
                                samples, sample_rate = kokoro.create(
                                    chunk,
                                    voice=voice_model,
                                    speed=Config.KOKORO_SPEED,
                                    lang="en-us"
                                )

                                # Encode directly to Opus (skip WAV generation)
                                opus_encoded = encode_pcm_to_opus(samples, sample_rate, chunk_file)

                                if not opus_encoded:
                                    logging.error(f"Failed to encode Opus for chunk {i+1} - skipping")
                                    continue

                                chunk_generation_time = time.time() - chunk_start_time

                                # Add this chunk to the queue
                                chunk_queue.put(chunk_file)
                                logging.info(f"Generated chunk {i+1}/{len(chunks)}: {chunk_file} (took {chunk_generation_time:.3f}s)")
                                
                            except Exception as e:
                                logging.error(f"Error generating chunk {i+1}: {str(e)}")
                    except Exception as e:
                        logging.error(f"Error in background generation: {str(e)}")
                    finally:
                        # Signal that generation is complete
                        generation_complete.set()
                        logging.info("Background chunk generation completed")
                
                # Start the background thread if there are more chunks with enhanced startup
                if len(chunks) > 1:
                    thread = threading.Thread(target=generate_remaining_chunks)
                    thread.daemon = True
                    thread.start()
                    
                    # Enhanced adaptive head start with improved timing for different text lengths
                    if len(chunks) > 5:
                        # Very long texts: maximum stabilization time
                        head_start_delay = initial_delay + 0.05
                    elif len(chunks) > 3:
                        # Long texts: use the adaptive delay
                        head_start_delay = initial_delay
                    else:
                        # Shorter texts: reduced delay to maintain responsiveness
                        head_start_delay = max(0.08, initial_delay - 0.02)
                    
                    time.sleep(head_start_delay)
                    logging.info(f"Background chunk generation started with {head_start_delay:.3f}s head start for {len(chunks)} chunks")
                else:
                    # If only one chunk, signal completion immediately
                    generation_complete.set()
                    logging.info("Single chunk generation - no background processing needed")
                
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
    Get the next chunk from the queue with enhanced timeout handling and monitoring.
    
    Returns:
    str or None: Path to the next chunk file, or None if no more chunks
    """
    try:
        if not chunk_queue.empty():
            chunk = chunk_queue.get_nowait()
            logging.debug(f"Retrieved chunk from queue (queue size now: {chunk_queue.qsize()})")
            return chunk
        elif generation_complete.is_set():
            logging.debug("No more chunks - generation complete")
            return None
        else:
            # Enhanced adaptive timeout based on queue state and generation progress
            base_timeout = 0.6  # Slightly increased base timeout
            
            # If queue is empty but generation isn't complete, implement progressive timeout
            if chunk_queue.empty() and not generation_complete.is_set():
                # First try a short wait
                try:
                    chunk = chunk_queue.get(timeout=base_timeout)
                    logging.debug("Retrieved chunk after standard wait")
                    return chunk
                except queue.Empty:
                    # If still empty, try an extended timeout for complex processing
                    extended_timeout = 2.5  # Slightly longer for very complex chunks
                    try:
                        chunk = chunk_queue.get(timeout=extended_timeout)
                        logging.info("Retrieved chunk after extended wait - possible complex generation")
                        return chunk
                    except queue.Empty:
                        logging.warning("Extended timeout reached - generation may be experiencing issues")
                        return None
            else:
                try:
                    chunk = chunk_queue.get(timeout=base_timeout)
                    logging.debug("Retrieved chunk with standard timeout")
                    return chunk
                except queue.Empty:
                    logging.debug("No chunk available within timeout")
                    return None
    except Exception as e:
        logging.error(f"Error getting next chunk: {e}")
        return None


def get_chunk_generation_stats():
    """
    Get comprehensive statistics about chunk generation for debugging and monitoring.
    
    Returns:
    dict: Detailed statistics about current generation state
    """
    return {
        'queue_size': chunk_queue.qsize(),
        'generation_complete': generation_complete.is_set(),
        'queue_empty': chunk_queue.empty(),
        'timestamp': time.time(),
        'status': 'complete' if generation_complete.is_set() else ('generating' if not chunk_queue.empty() else 'waiting')
    }

def split_text_into_chunks(text, max_chars=150):
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