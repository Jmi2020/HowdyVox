import os
import struct
import logging
import time
import pyaudio
import pvporcupine
import gc  # Garbage collection
from colorama import Fore, init
from dotenv import load_dotenv
import speech_recognition as sr
import threading
import queue

# Initialize colorama
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# Global variables for better resource management
DETECTOR_REGISTRY = []  # Keep track of all detector instances for proper cleanup
MAX_DETECTORS = 5  # Maximum number of detectors to keep in memory

def cleanup_all_detectors():
    """Force cleanup of all detector instances"""
    for detector in list(DETECTOR_REGISTRY):
        try:
            detector.cleanup(force=True)
        except:
            pass
    DETECTOR_REGISTRY.clear()
    gc.collect()

class WakeWordDetector:
    """
    Wake word detection using Porcupine by Picovoice.
    Listens for the custom wake word "Hey howdy" using a provided model file.
    """
    
    def __init__(self, wake_word_callback, sensitivity=0.5):
        """
        Initialize the wake word detector.
        
        Args:
            wake_word_callback: Function to call when wake word is detected
            sensitivity: Detection sensitivity (0-1), higher is more sensitive
        """
        self.wake_word_callback = wake_word_callback
        self.sensitivity = sensitivity
        self.porcupine = None
        self.audio = None
        self.is_running = False
        self.audio_stream = None
        self.stop_event = threading.Event()
        self.detection_thread = None
        self.detection_queue = queue.Queue()
        
        # Register this instance for cleanup
        DETECTOR_REGISTRY.append(self)
        
        # Limit the number of instances to prevent memory issues
        while len(DETECTOR_REGISTRY) > MAX_DETECTORS:
            old_detector = DETECTOR_REGISTRY.pop(0)
            try:
                if old_detector != self:
                    old_detector.cleanup(force=True)
            except:
                pass
        
        # Get access key from environment variable
        access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        if not access_key:
            logging.error(f"{Fore.RED}PORCUPINE_ACCESS_KEY not found in environment variables{Fore.RESET}")
            raise ValueError("PORCUPINE_ACCESS_KEY environment variable not set")
        
        # Path to custom wake word model file
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "models", "Hey-Howdy_en_mac_v3_0_0.ppn")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logging.error(f"{Fore.RED}Custom wake word model not found at: {model_path}{Fore.RESET}")
            # Try looking in the current directory
            current_dir_model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           "Hey-Howdy_en_mac_v3_0_0.ppn")
            if os.path.exists(current_dir_model):
                logging.info(f"{Fore.GREEN}Found model in current directory: {current_dir_model}{Fore.RESET}")
                model_path = current_dir_model
            else:
                raise FileNotFoundError(f"Custom wake word model not found")
        else:
            logging.info(f"{Fore.GREEN}Found custom wake word model at: {model_path}{Fore.RESET}")
        
        # Try to create a Porcupine instance with the custom wake word model
        try:
            logging.info(f"{Fore.GREEN}Creating Porcupine with custom 'Hey Howdy' wake word model{Fore.RESET}")
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[model_path],
                sensitivities=[sensitivity]
            )
            logging.info(f"{Fore.GREEN}Porcupine initialized successfully with custom wake word{Fore.RESET}")
        except Exception as e:
            logging.error(f"{Fore.RED}Error initializing Porcupine with custom model: {e}{Fore.RESET}")
            # Try with built-in keywords as fallback
            try:
                logging.info(f"{Fore.YELLOW}Trying with built-in keywords instead...{Fore.RESET}")
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=["alexa", "computer", "hey google"],
                    sensitivities=[sensitivity] * 3
                )
                logging.info(f"{Fore.YELLOW}Using built-in wake words. Say 'Hey Google', 'Alexa', or 'Computer' to activate.{Fore.RESET}")
            except Exception as e2:
                logging.error(f"{Fore.RED}Error initializing Porcupine with built-in keywords: {e2}{Fore.RESET}")
                raise ValueError(f"Could not initialize Porcupine with either custom model or built-in keywords: {e2}")
            
        # Initialize PyAudio
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            logging.error(f"{Fore.RED}Error initializing PyAudio: {e}{Fore.RESET}")
            raise
        
    def _detection_worker(self):
        """Worker thread that processes audio in the background"""
        try:
            # Create audio stream
            self.audio_stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            logging.info(f"{Fore.GREEN}Wake word detector started. Listening...{Fore.RESET}")
            
            # Main detection loop
            while not self.stop_event.is_set():
                try:
                    # Read audio frame
                    pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                    # Convert to the format Porcupine expects
                    pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                    
                    # Process with Porcupine
                    keyword_index = self.porcupine.process(pcm)
                    
                    # If wake word detected (keyword_index >= 0)
                    if keyword_index >= 0:
                        logging.info(f"{Fore.CYAN}Wake word detected!{Fore.RESET}")
                        self.detection_queue.put(True)
                        # Brief pause to prevent multiple detections
                        time.sleep(0.5)
                except Exception as e:
                    if not self.stop_event.is_set():
                        logging.error(f"Error processing audio frame: {e}")
                    time.sleep(0.1)  # Short sleep to avoid tight loop on error
        
        except Exception as e:
            logging.error(f"Error in detection worker: {e}")
        finally:
            self.cleanup_audio_stream()
    
    def cleanup_audio_stream(self):
        """Clean up just the audio stream"""
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
                logging.info("Audio stream closed")
            except Exception as e:
                logging.error(f"Error closing audio stream: {e}")
        
    def start(self):
        """Start listening for the wake word in a separate thread."""
        self.is_running = True
        self.stop_event.clear()
        
        # Start detection in a separate thread
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start callback thread to handle detections
        def callback_worker():
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Wait for detection with timeout
                    detected = self.detection_queue.get(timeout=0.5)
                    if detected and self.wake_word_callback:
                        try:
                            self.wake_word_callback()
                        except Exception as e:
                            logging.error(f"Error in wake word callback: {e}")
                except queue.Empty:
                    # Timeout, just continue
                    pass
                except Exception as e:
                    logging.error(f"Error in callback worker: {e}")
                    time.sleep(0.1)  # Avoid tight loop on error
        
        # Start callback thread
        self.callback_thread = threading.Thread(target=callback_worker)
        self.callback_thread.daemon = True
        self.callback_thread.start()
    
    def stop(self):
        """Stop the wake word detector."""
        logging.info("Stopping wake word detector...")
        self.is_running = False
        self.stop_event.set()
        
        # Allow threads to terminate gracefully
        time.sleep(0.2)
        
        # Clean up resources
        self.cleanup(force=False)
    
    def cleanup(self, force=False):
        """Clean up resources."""
        self.cleanup_audio_stream()
        
        # Clean up Porcupine
        if self.porcupine is not None:
            try:
                self.porcupine.delete()
                self.porcupine = None
                logging.info("Porcupine instance deleted")
            except Exception as e:
                logging.error(f"Error deleting Porcupine instance: {e}")
        
        # Clean up PyAudio only if forcing or no other detectors are using it
        if force or not any(d.is_running for d in DETECTOR_REGISTRY if d != self):
            if self.audio is not None:
                try:
                    self.audio.terminate()
                    self.audio = None
                    logging.info("PyAudio terminated")
                except Exception as e:
                    logging.error(f"Error terminating PyAudio: {e}")
        
        # Manually trigger garbage collection to reclaim resources
        gc.collect()
        
        # Remove from registry
        if self in DETECTOR_REGISTRY:
            DETECTOR_REGISTRY.remove(self)
        
        logging.info("Wake word detector resources cleaned up")
        
    def __del__(self):
        """Clean up resources on deletion."""
        self.cleanup(force=True)


# Alternative implementation using SpeechRecognition for wake word detection
# This can be used if Porcupine is not available or not preferred
class SpeechRecognitionWakeWord:
    """
    Wake word detection using SpeechRecognition library.
    Listens for the wake phrase "Hey howdy" and triggers a callback when detected.
    """
    
    def __init__(self, wake_word_callback, wake_phrase="hey howdy"):
        """
        Initialize the wake word detector.
        
        Args:
            wake_word_callback: Function to call when wake word is detected
            wake_phrase: The wake phrase to listen for (default: "hey howdy")
        """
        self.wake_word_callback = wake_word_callback
        self.wake_phrase = wake_phrase.lower()
        self.recognizer = sr.Recognizer()
        self.is_running = False
        self.stop_event = threading.Event()
        self.detection_thread = None
        
        # Set a lower energy threshold to pick up more speech
        self.recognizer.energy_threshold = 500
        self.recognizer.dynamic_energy_threshold = True
        
    def _detection_worker(self):
        """Worker thread for wake word detection"""
        try:
            while not self.stop_event.is_set():
                try:
                    with sr.Microphone() as source:
                        # Adjust for ambient noise
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        
                        # Listen for audio with timeout
                        try:
                            audio = self.recognizer.listen(
                                source, 
                                timeout=5.0,  # Add timeout to allow checking stop_event
                                phrase_time_limit=3  # Only listen for short phrases (wake words)
                            )
                        except sr.WaitTimeoutError:
                            # No speech detected, continue
                            continue
                        
                        if self.stop_event.is_set():
                            break
                        
                        try:
                            # Try to recognize the audio (use Google's API)
                            text = self.recognizer.recognize_google(audio).lower()
                            logging.debug(f"Heard: {text}")
                            
                            # Check if the wake phrase is in the recognized text
                            if self.wake_phrase in text:
                                logging.info(f"{Fore.CYAN}Wake word detected: '{text}'{Fore.RESET}")
                                # Call the callback function
                                self.wake_word_callback()
                        
                        except sr.UnknownValueError:
                            # Speech was unintelligible
                            pass
                        except sr.RequestError as e:
                            # API was unreachable or unresponsive
                            logging.error(f"API error: {e}")
                
                except Exception as e:
                    logging.error(f"Error in SpeechRecognition detection: {e}")
                    # Avoid tight loop on error
                    time.sleep(0.5)
        
        except Exception as e:
            logging.error(f"Fatal error in SpeechRecognition worker: {e}")
    
    def start(self):
        """Start listening for the wake word."""
        self.is_running = True
        self.stop_event.clear()
        
        logging.info(f"{Fore.GREEN}Wake word detector started. Listening for '{self.wake_phrase}'...{Fore.RESET}")
        
        # Start detection in a separate thread
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop(self):
        """Stop the wake word detector."""
        logging.info("Stopping SpeechRecognition wake word detector")
        self.is_running = False
        self.stop_event.set()
        
        # Wait for thread to terminate
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Manually trigger garbage collection
        gc.collect()
        
    def cleanup(self, force=True):
        """Clean up resources (compatibility with WakeWordDetector)"""
        self.stop()