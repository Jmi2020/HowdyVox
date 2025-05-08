# voice_assistant/utils.py

import os
import logging
import io
import gc
from threading import Lock

def delete_file(file_path):
    """
    Delete a file from the filesystem.
    
    Args:
    file_path (str): The path to the file to delete.
    """
    try:
        os.remove(file_path)
        logging.info(f"Deleted file: {file_path}")
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
    except PermissionError:
        logging.error(f"Permission denied when trying to delete file: {file_path}")
    except OSError as e:
        logging.error(f"Error deleting file {file_path}: {e}")

# Audio Buffer Pool for Memory Optimization
_audio_buffer_pool = []
_audio_buffer_pool_lock = Lock()
MAX_POOL_SIZE = 10

def get_audio_buffer():
    """
    Get a reusable BytesIO buffer from the pool.
    This reduces memory allocations when processing audio.
    
    Returns:
        BytesIO: A buffer object for audio data
    """
    global _audio_buffer_pool
    
    with _audio_buffer_pool_lock:
        if _audio_buffer_pool:
            return _audio_buffer_pool.pop()
    
    # If no buffer available in pool, create a new one
    return io.BytesIO()

def release_audio_buffer(buffer):
    """
    Return a buffer to the pool for reuse.
    
    Args:
        buffer (BytesIO): The buffer to return to the pool
    """
    global _audio_buffer_pool
    
    # Reset the buffer
    buffer.seek(0)
    buffer.truncate(0)
    
    # Return to pool if not full
    with _audio_buffer_pool_lock:
        if len(_audio_buffer_pool) < MAX_POOL_SIZE:
            _audio_buffer_pool.append(buffer)

def clear_audio_buffer_pool():
    """Clear all buffers from the pool (for cleanup)"""
    global _audio_buffer_pool
    
    with _audio_buffer_pool_lock:
        _audio_buffer_pool.clear()

def targeted_gc():
    """
    Perform targeted garbage collection focused on audio objects.
    This helps reduce memory leaks from audio processing resources.
    
    Returns:
        int: Number of audio objects cleaned up
    """
    # First, collect only objects that are no longer referenced
    gc.collect(0)
    
    # Look for specific problematic object types to clean up
    audio_related_types = ['PyAudio', 'AudioSegment', 'Wave_read', 'Wave_write',
                           'Stream', 'InferenceSession']
    count = 0
    
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        if obj_type in audio_related_types:
            # Force deallocation of audio objects if possible
            try:
                if hasattr(obj, 'close') and callable(obj.close):
                    obj.close()
                if hasattr(obj, '__del__') and callable(obj.__del__):
                    obj.__del__()
                count += 1
            except Exception as e:
                logging.debug(f"Error cleaning up {obj_type} object: {e}")
    
    # Final collection to clean up newly unreferenced objects
    gc.collect()
    logging.debug(f"Targeted GC cleaned up {count} audio-related objects")
    return count
