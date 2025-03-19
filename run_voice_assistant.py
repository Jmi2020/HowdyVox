import logging
import time
import os
import threading
import signal
import sys
import gc  # For garbage collection
from colorama import Fore, init
from voice_assistant.audio import record_audio, play_audio
from voice_assistant.transcription import transcribe_audio
from voice_assistant.response_generation import generate_response
from voice_assistant.text_to_speech import text_to_speech, get_next_chunk, generation_complete
from voice_assistant.utils import delete_file
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key
from voice_assistant.kokoro_manager import KokoroManager
from voice_assistant.led_matrix_controller import LEDMatrixController

# Import the fixed wake word implementation
from voice_assistant.wake_word import WakeWordDetector, SpeechRecognitionWakeWord, cleanup_all_detectors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize colorama
init(autoreset=True)

# Global variables
wake_word_detected = threading.Event()
stop_signal = threading.Event()
wake_word_detector = None
conversation_active = threading.Event()  # Flag to track if we're in an active conversation
restart_count = 0  # Track how many times we've restarted to avoid infinite loops
led_matrix = None  # LED Matrix controller instance

def check_end_conversation(text):
    """
    Use simple pattern matching to determine if the user wants to end the conversation.
    
    Args:
        text (str): The user's input text
        
    Returns:
        bool: True if the user wants to end the conversation, False otherwise
    """
    # Only check for very explicit exit phrases, nothing ambiguous
    clear_exit_phrases = [
        "that's all", "that's it", "goodbye", "bye", "exit", 
        "end conversation", "stop listening", "thanks that's all", 
        "no more questions", "we're done", "i'm done", "that'll be all",
        "that will be all", "no more", "that's enough"
    ]
                          
    # Only end if there's an exact match with an exit phrase
    for phrase in clear_exit_phrases:
        if phrase in text.lower():
            logging.info(f"Exit phrase detected: '{phrase}' in '{text}'")
            return True
    
    # Questions and commands clearly indicate continuing the conversation
    if any(indicator in text.lower() for indicator in ["?", "tell me", "what", "how", "why", "can you", "could you"]):
        return False
        
    # By default, keep the conversation going unless explicitly ended
    return False

def handle_wake_word():
    """Callback function when wake word is detected"""
    logging.info("Wake word detected, activating conversation mode")
    # Set the events to trigger conversation mode
    wake_word_detected.set()
    conversation_active.set()
    
    # Update LED matrix to show "Listening"
    if led_matrix:
        led_matrix.set_listening()
    
    # Play a sound to indicate wake word detected
    try:
        play_audio("voice_samples/activate.wav")
    except:
        logging.info("Activation sound not found, continuing without it")

def safe_start_wake_word_detection():
    """Safely start a new wake word detection with error handling"""
    global wake_word_detector, restart_count
    
    try:
        # Force cleanup of any existing detector
        if wake_word_detector is not None:
            try:
                wake_word_detector.stop()
                # Let resources be properly released
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error stopping existing wake word detector: {e}")
        
        # Create a new detector
        print(Fore.CYAN + "Starting wake word detection..." + Fore.RESET)
        try:
            wake_word_detector = WakeWordDetector(wake_word_callback=handle_wake_word)
            print(Fore.GREEN + "Using Porcupine for wake word detection" + Fore.RESET)
        except Exception as e:
            print(Fore.YELLOW + f"Porcupine not available ({e}), falling back to SpeechRecognition" + Fore.RESET)
            wake_word_detector = SpeechRecognitionWakeWord(wake_word_callback=handle_wake_word)
            print(Fore.GREEN + "Using SpeechRecognition for wake word detection" + Fore.RESET)
        
        # Start the detector
        wake_word_detector.start()
        print(Fore.CYAN + "Wake word detection active - Say 'Hey Howdy' to activate..." + Fore.RESET)
        
        # Reset restart count on successful start
        restart_count = 0
        
        # Update LED matrix to "waiting" state when wake word detection is active
        if led_matrix:
            led_matrix.set_waiting()
        
        return True
    except Exception as e:
        # Handle start failure
        logging.error(f"Failed to start wake word detector: {e}")
        restart_count += 1
        
        # If we've tried restarting too many times, give up
        if restart_count > 3:
            logging.error("Too many failed restart attempts, giving up")
            return False
        
        # Run garbage collection to free resources
        gc.collect()
        time.sleep(1.0)  # Wait a bit before trying again
        
        return False

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit"""
    print(f"\n{Fore.YELLOW}Shutting down...{Fore.RESET}")
    stop_signal.set()
    
    # Set matrix to waiting state before exiting
    if led_matrix:
        led_matrix.set_waiting()
    
    # Cleanup all wake word detectors
    cleanup_all_detectors()
    
    sys.exit(0)

def main():
    """
    Main function to run the offline voice assistant with wake word detection
    and continuous conversation support.
    """
    global led_matrix
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create necessary directories
    os.makedirs("temp/audio", exist_ok=True)
    os.makedirs("voice_samples", exist_ok=True)
    
    # Initialize LED Matrix controller if ESP32 IP is provided
    if hasattr(Config, 'USE_LED_MATRIX') and Config.USE_LED_MATRIX and Config.ESP32_IP:
        logging.info(f"{Fore.CYAN}Initializing ESP32 LED Matrix controller with IP: {Config.ESP32_IP}{Fore.RESET}")
        led_matrix = LEDMatrixController(Config.ESP32_IP)
        # Set initial state to waiting for wake word
        led_matrix.set_waiting()
    
    # Check if we have a notification sound, create a simple one if not
    activate_sound = "voice_samples/activate.wav"
    if not os.path.exists(activate_sound):
        try:
            # Try to generate a simple "activated" sound using Kokoro
            print(Fore.YELLOW + "Creating activation sound..." + Fore.RESET)
            kokoro = KokoroManager.get_instance(local_model_path=Config.LOCAL_MODEL_PATH)
            samples, sample_rate = kokoro.create("Howdy, I'm listening!", 
                                                voice=Config.KOKORO_VOICE, 
                                                speed=1.0, 
                                                lang="en-us")
            import soundfile as sf
            sf.write(activate_sound, samples, sample_rate)
        except Exception as e:
            print(Fore.RED + f"Couldn't create activation sound: {e}" + Fore.RESET)
    
    # Preload Kokoro model
    print(Fore.YELLOW + "Initializing Kokoro TTS model..." + Fore.RESET)
    try:
        KokoroManager.get_instance(local_model_path=Config.LOCAL_MODEL_PATH)
        print(Fore.GREEN + "Kokoro TTS model loaded successfully!" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + f"Warning: Failed to preload Kokoro TTS model: {e}" + Fore.RESET)
        print(Fore.YELLOW + "Will attempt to load on first use" + Fore.RESET)
    
    # Initialize chat history
    chat_history = [
        {"role": "system", "content": """ You are a helpful Assistant called Howdy. 
         You are friendly and fun and you will help the users with their requests.
         Your answers are short and concise. """}
    ]
    
    # Flag to track if we're currently playing audio
    playback_complete_event = threading.Event()
    playback_complete_event.set()  # Initially set to True since no playback is happening

    # Start the wake word detection
    if not safe_start_wake_word_detection():
        print(Fore.RED + "Failed to start wake word detection. Exiting..." + Fore.RESET)
        return
    
    while not stop_signal.is_set():
        try:
            # Wait for wake word to be detected or a conversation to be active
            if not wake_word_detected.is_set() and not conversation_active.is_set():
                # Check if wake word detection is working
                if wake_word_detector is None:
                    logging.warning("Wake word detector not initialized, restarting...")
                    if not safe_start_wake_word_detection():
                        # If we can't restart, wait a bit and try again
                        time.sleep(5)
                        continue
                
                time.sleep(0.1)  # Short sleep to prevent busy waiting
                continue
            
            # Clear the wake word detected flag for next time
            wake_word_detected.clear()
            
            # Make sure previous playback is complete before recording
            if not playback_complete_event.is_set():
                logging.info("Waiting for previous audio playback to complete...")
                playback_complete_event.wait()
            
            # Record audio from the microphone and save it
            if conversation_active.is_set():
                # In active conversation, give a visual indicator we're listening
                print(Fore.GREEN + "Listening..." + Fore.RESET)
                # Update LED matrix to "Listening" (LED state is already set in handle_wake_word)
                # The second set_listening call was here - removed to fix duplicate updates
            
            record_audio(Config.INPUT_AUDIO)

            # Get the API key for transcription (will be None for FastWhisperAPI)
            transcription_api_key = get_transcription_api_key()
            
            # Transcribe the audio file using FastWhisperAPI
            user_input = transcribe_audio(Config.TRANSCRIPTION_MODEL, transcription_api_key, Config.INPUT_AUDIO, Config.LOCAL_MODEL_PATH)

            # Check if the transcription is empty
            if not user_input:
                logging.info("No transcription was returned.")
                if conversation_active.is_set():
                    # If in active conversation, just continue listening
                    print(Fore.YELLOW + "I didn't catch that. Could you say it again?" + Fore.RESET)
                    # Get a quick TTS response
                    success, first_chunk_file = text_to_speech(
                        Config.TTS_MODEL, 
                        get_tts_api_key(), 
                        "I didn't catch that. Could you say it again?", 
                        "temp_feedback.wav", 
                        Config.LOCAL_MODEL_PATH
                    )
                    if success:
                        play_audio(first_chunk_file)
                        delete_file(first_chunk_file)
                    continue
                else:
                    # If waiting for wake word, go back to waiting
                    continue
                
            logging.info(Fore.GREEN + "You said: " + user_input + Fore.RESET)

            # Check if the user wants to exit the program (complete shutdown)
            if "goodbye" in user_input.lower() or "arrivederci" in user_input.lower():
                print(Fore.YELLOW + "Goodbye, partner! Happy trails!" + Fore.RESET)
                # Set matrix to waiting state before exiting
                if led_matrix:
                    led_matrix.set_waiting()
                stop_signal.set()
                break
                
            # Check if we should end the current conversation
            if conversation_active.is_set() and check_end_conversation(user_input):
                # End the conversation and go back to wake word mode
                print(Fore.YELLOW + "Ending conversation. Say 'Hey Howdy' when you need me again!" + Fore.RESET)
                
                # Update LED matrix to "Later Space Cowboy"
                if led_matrix:
                    led_matrix.set_ending()
                    # The LED matrix will auto-switch back to waiting after 10 seconds
                
                # Get a quick TTS response
                success, first_chunk_file = text_to_speech(
                    Config.TTS_MODEL, 
                    get_tts_api_key(), 
                    "Alright then, holler at me when you need me again!", 
                    "temp_end.wav", 
                    Config.LOCAL_MODEL_PATH
                )
                if success:
                    play_audio(first_chunk_file)
                    delete_file(first_chunk_file)
                
                # End the conversation
                conversation_active.clear()
                
                # Force recreation of wake word detector to avoid memory issues
                try:
                    logging.info("Restarting wake word detection after conversation...")
                    # First stop the current detector
                    if wake_word_detector:
                        wake_word_detector.stop()
                    
                    # Run garbage collection explicitly to free resources
                    gc.collect()
                    
                    # Wait for resources to be completely released
                    time.sleep(2.0)
                    
                    # Create a fresh detector
                    safe_start_wake_word_detection()
                except Exception as e:
                    logging.error(f"Error recreating wake word detector: {e}")
                
                continue

            # Append the user's input to the chat history
            chat_history.append({"role": "user", "content": user_input})

            # Get the API key for response generation (will be None for Ollama)
            response_api_key = get_response_api_key()

            # Update LED matrix to "Thinking"
            if led_matrix:
                led_matrix.set_thinking()
                
            # Generate a response using Ollama
            response_text = generate_response(Config.RESPONSE_MODEL, response_api_key, chat_history, Config.LOCAL_MODEL_PATH)
            logging.info(Fore.CYAN + "Response: " + response_text + Fore.RESET)

            # Append the assistant's response to the chat history
            chat_history.append({"role": "assistant", "content": response_text})

            # For Kokoro, always use WAV output
            output_file = 'output.wav'

            # Get the API key for TTS (will be None for Kokoro)
            tts_api_key = get_tts_api_key()

            # Signal that we're starting audio processing
            playback_complete_event.clear()
            
            # Update LED matrix to "speaking" mode with the response text
            if led_matrix:
                led_matrix.set_speaking(response_text)
            
            # Get just the first chunk
            success, first_chunk_file = text_to_speech(
                Config.TTS_MODEL, 
                tts_api_key, 
                response_text, 
                output_file, 
                Config.LOCAL_MODEL_PATH
            )
            
            # List to track files for cleanup
            files_to_cleanup = []
            
            if success and first_chunk_file:
                # Add first chunk to cleanup list
                files_to_cleanup.append(first_chunk_file)
                
                # Define a thread to handle playback of all chunks
                def play_all_chunks():
                    try:
                        # Play the first chunk immediately
                        logging.info(f"Playing first chunk (immediately)")
                        play_audio(first_chunk_file)
                        
                        # Continue playing chunks as they become available
                        chunk_index = 1
                        while True:
                            # Check if there are more chunks to play
                            next_chunk = get_next_chunk()
                            
                            # If no more chunks and generation is complete, we're done
                            if next_chunk is None and generation_complete.is_set():
                                break
                            
                            # If we got a chunk, play it
                            if next_chunk:
                                files_to_cleanup.append(next_chunk)
                                logging.info(f"Playing next chunk ({chunk_index+1})")
                                play_audio(next_chunk)
                                chunk_index += 1
                            else:
                                # Short wait to check again for new chunks
                                time.sleep(0.1)
                    
                    except Exception as e:
                        logging.error(f"Error in playback thread: {str(e)}")
                    finally:
                        # Clean up all chunk files
                        for chunk_file in files_to_cleanup:
                            try:
                                delete_file(chunk_file)
                            except Exception as e:
                                logging.warning(f"Could not delete chunk file {chunk_file}: {e}")
                        
                        # Signal that playback is complete
                        playback_complete_event.set()
                        
                        # If conversation is active, provide visual cue that we're ready for next input
                        if conversation_active.is_set():
                            print(Fore.GREEN + "Ready for your next question..." + Fore.RESET)
                            # Update LED matrix back to "Listening" mode
                            if led_matrix:
                                led_matrix.set_listening()
                
                # Start playback thread
                playback_thread = threading.Thread(target=play_all_chunks)
                playback_thread.daemon = True
                playback_thread.start()
            else:
                logging.error("Failed to generate speech")
                playback_complete_event.set()
                # Reset LED matrix to "Listening" in case of failure
                if led_matrix and conversation_active.is_set():
                    led_matrix.set_listening()

        except Exception as e:
            logging.error(Fore.RED + f"An error occurred: {e}" + Fore.RESET)
            import traceback
            traceback.print_exc()
            delete_file(Config.INPUT_AUDIO)
            playback_complete_event.set()
            time.sleep(1)
    
    # Cleanup
    if led_matrix:
        # Set matrix to waiting state before exiting
        led_matrix.set_waiting()
    cleanup_all_detectors()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Shutting down due to keyboard interrupt...{Fore.RESET}")
        if led_matrix:
            led_matrix.set_waiting()
        cleanup_all_detectors()
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}{Fore.RESET}")
        import traceback
        traceback.print_exc()
        if led_matrix:
            led_matrix.set_waiting()
        cleanup_all_detectors()