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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@lru_cache(maxsize=None)
def get_recognizer():
    """
    Return a cached speech recognizer instance
    """
    return sr.Recognizer()

def record_audio(file_path, timeout=10, phrase_time_limit=None, retries=3, energy_threshold=2000, 
                 pause_threshold=1, phrase_threshold=0.1, dynamic_energy_threshold=True, 
                 calibration_duration=1):
    """
    Record audio from the microphone and save it as an MP3 file.
    """
    # Import here to avoid circular import
    from voice_assistant.config import Config
    
    # Make sure temp audio directory exists
    os.makedirs(Config.TEMP_AUDIO_DIR, exist_ok=True)
    
    recognizer = get_recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold
    recognizer.phrase_threshold = phrase_threshold
    recognizer.dynamic_energy_threshold = dynamic_energy_threshold
    
    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                logging.info("Calibrating for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=calibration_duration)
                logging.info("Recording started")
                # Listen for the first phrase and extract it into audio data
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logging.info("Recording complete")

                # Convert the recorded audio data to an MP3 file
                wav_data = audio_data.get_wav_data()
                audio_segment = pydub.AudioSegment.from_wav(BytesIO(wav_data))
                mp3_data = audio_segment.export(file_path, format="mp3", bitrate="128k", parameters=["-ar", "22050", "-ac", "1"])
                return
        except sr.WaitTimeoutError:
            logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            logging.error(f"Failed to record audio: {e}")
            if attempt == retries -1:
                raise
        
    logging.error("Recording failed after all retries")

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
                # Convert MP3 to WAV then play
                sound = AudioSegment.from_file(file_path, format="mp3")
                temp_wav = os.path.join(Config.TEMP_AUDIO_DIR, f"temp_{i}.wav")
                sound.export(temp_wav, format="wav")
                
                # Play the temp WAV file
                wf = wave.open(temp_wav, 'rb')
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
                
                # Clean up temp file
                try:
                    os.remove(temp_wav)
                except:
                    pass
            else:
                logging.error(f"Unsupported audio format: {file_extension}")
                
            # Small delay between chunks for natural pauses
            time.sleep(0.25)
            
        return True
    except Exception as e:
        logging.error(f"Error in sequential playback: {str(e)}")
        return False