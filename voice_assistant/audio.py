# voice_assistant/audio.py

import speech_recognition as sr
import pygame
import time
import logging
import pydub
from io import BytesIO
from pydub import AudioSegment
from functools import lru_cache
import pyaudio
import wave
import os
import subprocess
from voice_assistant.utils import get_audio_buffer, release_audio_buffer
from voice_assistant.enhanced_audio import record_audio_enhanced
from voice_assistant.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@lru_cache(maxsize=None)
def get_recognizer():
    """
    Return a cached speech recognizer instance
    """
    return sr.Recognizer()

def detect_leading_silence(sound, silence_threshold=-50, chunk_size=10):
    """
    Detect leading silence in an audio segment.
    Returns the duration of leading silence in milliseconds.
    
    Parameters:
        sound: pydub.AudioSegment
        silence_threshold: threshold in dB below reference (default: -50)
        chunk_size: size of chunks to analyze in ms (default: 10)
    """
    trim_ms = 0
    assert chunk_size > 0
    while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms


def record_audio(file_path, timeout=10, phrase_time_limit=None, retries=3, energy_threshold=1200, 
                 pause_threshold=0.8, phrase_threshold=0.3, dynamic_energy_threshold=False, 
                 calibration_duration=1.5, is_wake_word_response=False):
    """
    Record audio from the microphone and save it as an MP3 file.
    
    This function now uses intelligent VAD when enabled in configuration.
    
    Args:
        file_path: Path to save the MP3 file
        timeout: Maximum time to wait for phrase to start
        phrase_time_limit: Maximum time to allow for a phrase
        retries: Number of times to retry recording on failure
        energy_threshold: Minimum audio energy to consider for recording
        pause_threshold: Seconds of silence to consider the end of a phrase
        phrase_threshold: Minimum duration of speaking to consider a phrase
        dynamic_energy_threshold: Whether to adjust energy threshold dynamically
        calibration_duration: Seconds to calibrate microphone for ambient noise
        is_wake_word_response: Whether this is recording after wake word detection (forces trim)
    """
    # Make sure temp audio directory exists
    os.makedirs(Config.TEMP_AUDIO_DIR, exist_ok=True)
    
    # Check if intelligent VAD is enabled
    if Config.USE_INTELLIGENT_VAD:
        logging.info("Using intelligent VAD for recording")
        
        # Use the enhanced recorder
        for attempt in range(retries):
            success = record_audio_enhanced(
                file_path=file_path,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
                is_wake_word_response=is_wake_word_response
            )
            
            if success:
                return True
            
            logging.warning(f"Recording attempt {attempt + 1} failed, retrying...")
        
        logging.error("All recording attempts failed")
        return False
    
    else:
        # Fall back to original implementation
        logging.info("Using legacy energy-based VAD")
    
    recognizer = get_recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold
    recognizer.phrase_threshold = phrase_threshold
    recognizer.dynamic_energy_threshold = dynamic_energy_threshold
    
    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                logging.info(f"Calibrating for ambient noise (threshold: {recognizer.energy_threshold})...")
                # Longer calibration to better detect ambient noise levels
                recognizer.adjust_for_ambient_noise(source, duration=calibration_duration)
                
                # Log the post-calibration energy threshold
                logging.info(f"Post-calibration energy threshold: {recognizer.energy_threshold}")
                
                # If dynamic threshold is disabled, ensure minimum threshold is maintained
                if not dynamic_energy_threshold and recognizer.energy_threshold < energy_threshold:
                    recognizer.energy_threshold = energy_threshold
                    logging.info(f"Enforcing minimum energy threshold: {recognizer.energy_threshold}")
                
                # Wait a short moment to ensure any activation sounds have completely stopped
                time.sleep(0.5)
                
                logging.info("Recording started")
                # Listen for the first phrase and extract it into audio data
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logging.info("Recording complete")

                # Get a buffer from the pool for processing
                buffer = get_audio_buffer()
                
                try:
                    # Convert the recorded audio data to an MP3 file
                    wav_data = audio_data.get_wav_data()
                    buffer.write(wav_data)
                    buffer.seek(0)
                    
                    audio_segment = pydub.AudioSegment.from_wav(buffer)
                    
                    # Try to remove any potential activation sound remnants by trimming initial sounds
                    # This helps avoid hearing the tail end of system prompts in recordings
                    try:
                        # Find where actual speech starts using our custom detector function
                        # Using a very aggressive threshold (-40dB) to ensure activation sounds are removed
                        start_trim = detect_leading_silence(audio_segment, silence_threshold=-40)
                        
                        if is_wake_word_response:
                            # For wake word response, always trim the first 500ms of audio to remove activation sound reliably
                            # But don't trim more than 3 seconds total to avoid cutting off actual speech
                            forced_trim = 500  # Always trim at least 500ms
                            if start_trim > forced_trim:
                                # If we detected substantial leading noise, trim it
                                # But don't trim more than 3 seconds
                                start_trim = min(3000, start_trim)
                                # A small buffer (100ms) to make sure we don't cut off actual speech
                                trim_point = max(forced_trim, start_trim - 100)
                                audio_segment = audio_segment[trim_point:]
                                logging.info(f"Trimmed {trim_point}ms from beginning of recording to remove activation sound")
                            else:
                                # If we didn't detect much silence, still trim the forced amount
                                audio_segment = audio_segment[forced_trim:]
                                logging.info(f"Forced trim of {forced_trim}ms from beginning of recording")
                        else:
                            # For regular conversation turns, only trim if we detect substantial silence
                            if start_trim > 300:
                                # A reasonable buffer (100ms) to make sure we don't cut off speech
                                trim_point = max(0, start_trim - 100)
                                audio_segment = audio_segment[trim_point:]
                                logging.info(f"Trimmed {trim_point}ms of silence from beginning of recording")
                            else:
                                logging.info("No significant leading silence detected, keeping full recording")
                    except Exception as trim_error:
                        # If trimming fails, just use the original audio
                        logging.warning(f"Failed to trim audio: {trim_error}")
                    
                    # Export to MP3 using optimized parameters
                    mp3_data = audio_segment.export(file_path, format="mp3", bitrate="128k", parameters=["-ar", "22050", "-ac", "1"])
                    
                    # Release the buffer back to the pool
                    release_audio_buffer(buffer)
                except Exception as e:
                    # Release the buffer even if an error occurs
                    release_audio_buffer(buffer)
                    raise e
                return True  # Successfully recorded audio
        except sr.WaitTimeoutError:
            logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            logging.error(f"Failed to record audio: {e}")
            # Don't raise on final attempt, just log and continue to next attempt or return False
        
    logging.error("Recording failed after all retries")
    return False  # Failed to record audio

def play_audio(file_path):
    """
    Play a single audio file using pygame.
    
    Args:
    file_path (str): The path to the audio file to play.
    """
    if not os.path.exists(file_path):
        logging.error(f"No file '{file_path}' found in working directory '{os.getcwd()}'.")
        return
    
    # Import here to avoid circular import
    from voice_assistant.config import Config
    
    # Make sure temp audio directory exists
    os.makedirs(Config.TEMP_AUDIO_DIR, exist_ok=True)
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.wav':
            # Play WAV files using the wave and pyaudio modules
            wf = wave.open(file_path, 'rb')
            p = pyaudio.PyAudio()
            
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
        elif file_extension == '.mp3':
            # For MP3 files, try different approaches depending on platform
            try:
                # Get a buffer from the pool for processing
                buffer = get_audio_buffer()
                
                try:
                    # First, try using pydub which is more reliable
                    sound = AudioSegment.from_file(file_path, format="mp3")
                    # Export to a temporary WAV file in the temp directory
                    temp_wav = os.path.join(Config.TEMP_AUDIO_DIR, "temp_output.wav")
                    sound.export(temp_wav, format="wav")
                    # Play the WAV file
                    play_audio(temp_wav)
                    # Clean up temp file
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                finally:
                    # Release the buffer back to the pool
                    release_audio_buffer(buffer)
            except Exception as e:
                logging.warning(f"Pydub playback failed: {str(e)}. Trying system player...")
                
                # If pydub fails, try system commands
                if os.name == 'posix':  # macOS, Linux
                    try:
                        # Try with afplay on macOS
                        if os.uname().sysname == 'Darwin':
                            subprocess.run(['afplay', file_path], check=True)
                        else:
                            # Try with mplayer on Linux
                            subprocess.run(['mplayer', file_path], check=True)
                    except Exception as e2:
                        logging.error(f"System audio player failed: {str(e2)}")
                else:  # Windows
                    try:
                        # On Windows, use the default media player
                        os.startfile(file_path)
                    except Exception as e2:
                        logging.error(f"System audio player failed: {str(e2)}")
        else:
            logging.error(f"Unsupported audio file format: {file_extension}")
    except Exception as e:
        logging.error(f"Failed to play audio: {str(e)}")

def play_audio_chunks(chunk_files):
    """
    Play multiple audio files in sequence, waiting for each to complete.
    
    Args:
    chunk_files (list): List of paths to audio chunk files to play in sequence.
    
    Returns:
    bool: True if playback was successful, False otherwise.
    """
    logging.info(f"Playing {len(chunk_files)} audio chunks sequentially")
    
    # Import here to avoid circular import
    from voice_assistant.config import Config
    
    # Make sure temp audio directory exists
    os.makedirs(Config.TEMP_AUDIO_DIR, exist_ok=True)
    
    try:
        for i, file_path in enumerate(chunk_files):
            logging.info(f"Playing chunk {i+1}/{len(chunk_files)}: {file_path}")
            
            if not os.path.exists(file_path):
                logging.error(f"Chunk file not found: {file_path}")
                continue
                
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.wav':
                # Play WAV files using the wave and pyaudio modules
                wf = wave.open(file_path, 'rb')
                p = pyaudio.PyAudio()
                
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)
                
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                wf.close()
            elif file_extension == '.mp3':
                # Get a buffer from the pool for WAV conversion
                buffer = get_audio_buffer()
                temp_wav = os.path.join(Config.TEMP_AUDIO_DIR, f"temp_{i}.wav")
                
                try:
                    # Convert MP3 to WAV then play
                    sound = AudioSegment.from_file(file_path, format="mp3")
                    sound.export(temp_wav, format="wav")
                    
                    # Play the temp WAV file
                    wf = wave.open(temp_wav, 'rb')
                    p = pyaudio.PyAudio()
                    
                    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)
                    
                    chunk_size = 1024
                    
                    # Use the buffer to read chunks
                    data = wf.readframes(chunk_size)
                    
                    while data:
                        stream.write(data)
                        data = wf.readframes(chunk_size)
                    
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    wf.close()
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                finally:
                    # Release the buffer back to the pool
                    release_audio_buffer(buffer)
            else:
                logging.error(f"Unsupported audio format: {file_extension}")
                
            # Small delay between chunks for natural pauses
            time.sleep(0.25)
            
        return True
    except Exception as e:
        logging.error(f"Error in sequential playback: {str(e)}")
        return False