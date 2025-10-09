import logging
import time
import os
import threading
import signal
import sys
import gc  # For garbage collection
import re
import argparse
from colorama import Fore, init
from voice_assistant.audio import record_audio, play_audio
from voice_assistant.transcription import transcribe_audio
from voice_assistant.response_generation import generate_response, preload_ollama_model
from voice_assistant.text_to_speech import text_to_speech, get_next_chunk, get_chunk_generation_stats, generation_complete
from voice_assistant.utils import delete_file, targeted_gc
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key
from voice_assistant.kokoro_manager import KokoroManager
from voice_assistant.led_matrix_controller import LEDMatrixController
# Import voice initializer to ensure blended voices are set up
from voice_assistant.voice_initializer import initialize_success as voice_initialized

# Import the fixed wake word implementation
from voice_assistant.wake_word import WakeWordDetector, SpeechRecognitionWakeWord, cleanup_all_detectors

# Import wireless audio support
from voice_assistant.network_audio_source import NetworkAudioSource
from voice_assistant.wireless_device_manager import WirelessDeviceManager
from voice_assistant.audio_source_manager import AudioSourceManager, AudioSourceType, get_audio_manager, set_audio_manager, cleanup_audio_manager
from voice_assistant.hotkey_manager import get_hotkey_manager, start_hotkeys, stop_hotkeys

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
activation_sound_playing = threading.Event()  # Flag to track if activation sound is playing
is_first_turn_after_wake = threading.Event()  # Flag to track if this is the first turn after wake word
network_audio_source = None  # Wireless audio source instance
audio_source_manager = None  # Audio source manager instance


def _is_wireless_audio_active() -> bool:
    """Return True when the ESP32-P4 wireless pipeline is connected and selected."""
    if not audio_source_manager:
        return False

    try:
        network_source = getattr(audio_source_manager, "_network_source", None)
        if not network_source:
            return False

        if audio_source_manager.get_current_source() != AudioSourceType.WIRELESS:
            return False

        if hasattr(network_source, "is_esp32p4_connected"):
            return network_source.is_esp32p4_connected()
    except Exception as exc:
        logging.debug(f"Wireless audio check failed: {exc}")

    # If we cannot definitively confirm, treat as inactive so we fall back gracefully.
    return False

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
    # No need for global declaration as it's already declared at module level
    
    logging.info("Wake word detected, activating conversation mode")
    # Set the events to trigger conversation mode
    wake_word_detected.set()
    conversation_active.set()
    
    # Set the flag that this is the first turn after wake word
    is_first_turn_after_wake.set()
    
    # Play activation sound first, before updating LED matrix
    try:
        # First, check if the activation sound exists
        if os.path.exists("voice_samples/activate.wav"):
            # Signal that activation sound is playing
            activation_sound_playing.set()
            
            # Play the activation sound in a separate thread so we don't block
            threading.Thread(target=lambda: play_and_clear_flag("voice_samples/activate.wav"), daemon=True).start()
            
            # Brief pause to ensure the playback thread starts
            time.sleep(0.1)
            
            logging.info("Activation sound started, preparing to listen")
    except Exception as e:
        logging.info(f"Activation sound error: {e}, continuing without it")
        # Clear the flag in case of error
        activation_sound_playing.clear()
        
    # Update LED matrix to show "Listening" right away
    if led_matrix:
        led_matrix.set_listening()
        
# Helper function to play audio and clear flag when done
def play_and_clear_flag(audio_path):
    try:
        play_audio(audio_path)
    finally:
        # Always clear the flag when audio playback completes
        activation_sound_playing.clear()

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
        
        # NEW: Check if we should use wireless audio for wake word detection
        use_wireless = (audio_source_manager and 
                       audio_source_manager.get_current_source() == AudioSourceType.WIRELESS)
        
        # Create a new detector
        print(Fore.CYAN + "Starting wake word detection..." + Fore.RESET)
        if use_wireless:
            print(Fore.YELLOW + "Using wireless audio for wake word detection" + Fore.RESET)
        
        try:
            wake_word_detector = WakeWordDetector(
                wake_word_callback=handle_wake_word,
                use_wireless_audio=use_wireless  # NEW parameter
            )
            print(Fore.GREEN + "Using Porcupine for wake word detection" + Fore.RESET)
        except Exception as e:
            print(Fore.YELLOW + f"Porcupine not available ({e}), falling back to SpeechRecognition" + Fore.RESET)
            wake_word_detector = SpeechRecognitionWakeWord(
                wake_word_callback=handle_wake_word,
                use_wireless_audio=use_wireless  # NEW parameter
            )
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
        
        # Run targeted garbage collection to free audio resources
        cleaned_count = targeted_gc()
        logging.info(f"Cleaned up {cleaned_count} audio-related objects")
        time.sleep(0.5)  # Short pause to allow resource cleanup
        
        return False

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit"""
    print(f"\n{Fore.YELLOW}Shutting down...{Fore.RESET}")
    stop_signal.set()
    
    # Set matrix to waiting state before exiting
    if led_matrix:
        led_matrix.set_waiting()
    
    # Cleanup audio source manager
    if audio_source_manager:
        print(f"{Fore.CYAN}Cleaning up audio source manager...{Fore.RESET}")
        audio_source_manager.cleanup()
    
    # Cleanup global audio manager
    cleanup_audio_manager()
    
    # Stop hotkeys
    stop_hotkeys()
    
    # Cleanup all wake word detectors
    cleanup_all_detectors()
    
    # Perform targeted garbage collection to clean up audio resources
    cleaned_count = targeted_gc()
    print(f"{Fore.CYAN}Cleaned up {cleaned_count} audio resources{Fore.RESET}")
    
    sys.exit(0)

def main():
    """
    Main function to run the offline voice assistant with wake word detection
    and continuous conversation support.
    """
    global led_matrix, network_audio_source, audio_source_manager
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HowdyTTS Voice Assistant')
    parser.add_argument('--wireless', action='store_true', 
                       help='Use wireless ESP32P4 devices instead of local microphone')
    parser.add_argument('--room', type=str, 
                       help='Target specific room for wireless audio (e.g., "Living Room")')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available wireless devices and exit')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect audio source (wireless first, local fallback)')
    
    args = parser.parse_args()
    
    # Determine initial audio source
    if args.list_devices:
        # Quick device listing
        temp_manager = AudioSourceManager()
        devices = temp_manager.get_available_devices()
        if devices:
            print(f"{Fore.GREEN}Available wireless devices:{Fore.RESET}")
            for idx, (_, name, ip) in enumerate(devices):
                print(f"  {idx}: {name} - {ip}")
        else:
            print(f"{Fore.YELLOW}No wireless devices found{Fore.RESET}")
        temp_manager.cleanup()
        return
    
    # Initialize audio source manager
    if args.wireless or args.room:
        initial_source = AudioSourceType.WIRELESS
    elif args.auto:
        initial_source = AudioSourceType.LOCAL  # Will auto-select in manager
    else:
        initial_source = AudioSourceType.LOCAL
    
    print(f"{Fore.CYAN}Initializing audio source manager...{Fore.RESET}")
    audio_source_manager = AudioSourceManager(initial_source, target_room=args.room)
    set_audio_manager(audio_source_manager)
    
    # Set up source change callback
    def on_source_changed(source_type: AudioSourceType, success: bool):
        status = "✓" if success else "✗"
        color = Fore.GREEN if success else Fore.RED
        print(f"{color}[Audio] {status} Switched to {source_type.value} microphone{Fore.RESET}")
    
    audio_source_manager.set_source_changed_callback(on_source_changed)
    
    # Auto-select or set initial source
    if args.auto:
        selected_source = audio_source_manager.auto_select_source()
        print(f"{Fore.GREEN}Auto-selected audio source: {selected_source.value}{Fore.RESET}")
    elif args.wireless or args.room:
        print(f"{Fore.CYAN}=" * 60 + Fore.RESET)
        print(f"{Fore.YELLOW}🎯 Wireless Mode - Waiting for ESP32-P4 HowdyScreen...{Fore.RESET}")
        print(f"{Fore.CYAN}=" * 60 + Fore.RESET)
        print(f"{Fore.CYAN}📡 Discovery server listening on UDP port 8001{Fore.RESET}")
        print(f"{Fore.CYAN}⏳ Waiting for ESP32-P4 to send 'HOWDYTTS_DISCOVERY' packet...{Fore.RESET}")
        print()
        
        # In wireless mode, wait indefinitely for ESP32-P4 connection
        device_found = False
        start_time = time.time()
        
        # Initialize wireless mode
        if audio_source_manager.switch_to_wireless(args.room):
            print(f"{Fore.CYAN}Wireless mode initialized successfully!{Fore.RESET}")
            print(f"{Fore.YELLOW}Waiting indefinitely for ESP32-P4 devices...{Fore.RESET}")
            
            # Keep checking for devices indefinitely
            while not device_found and not stop_signal.is_set():
                info = audio_source_manager.get_source_info()
                if info.get('wireless_devices', 0) > 0:
                    device_found = True
                    print(f"\n{Fore.GREEN}✅ ESP32-P4 device connected!{Fore.RESET}")
                    print(f"{Fore.GREEN}Using wireless audio source{Fore.RESET}")
                    if args.room:
                        print(f"{Fore.CYAN}Target room: {args.room}{Fore.RESET}")
                    break
                
                # Show waiting animation
                elapsed = int(time.time() - start_time)
                dots = "." * (elapsed % 4)
                print(f"\r{Fore.YELLOW}Waiting for ESP32-P4 connection{dots:4} ({elapsed}s) [Ctrl+C to exit]{Fore.RESET}", end="", flush=True)
                time.sleep(1)
                
            if stop_signal.is_set():
                print(f"\n{Fore.YELLOW}Exiting...{Fore.RESET}")
                return
        else:
            print(f"{Fore.RED}Failed to initialize wireless mode{Fore.RESET}")
            print(f"{Fore.RED}Cannot proceed in wireless mode. Please check:{Fore.RESET}")
            print(f"  1. Network audio modules are installed")
            print(f"  2. No permission errors for network access")
            print(f"  3. System firewall allows UDP port 8001")
            print(f"\n{Fore.RED}Exiting...{Fore.RESET}")
            return
        
        print()  # Add blank line after connection status
    
    # Show current audio source info
    info = audio_source_manager.get_source_info()
    print(f"{Fore.CYAN}Audio source: {info['current_source']}{Fore.RESET}")
    if info.get('wireless_devices', 0) > 0:
        print(f"{Fore.CYAN}Wireless devices: {info['wireless_devices']}{Fore.RESET}")
        
        # Get device details if available
        if hasattr(audio_source_manager, '_network_source') and audio_source_manager._network_source:
            devices = audio_source_manager._network_source.get_available_devices()
            if devices:
                print(f"{Fore.CYAN}Connected ESP32-P4 devices:{Fore.RESET}")
                for idx, device_name, device_ip in devices:
                    print(f"  • {device_name}: {device_ip}")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create necessary directories
    os.makedirs("temp/audio", exist_ok=True)
    os.makedirs("voice_samples", exist_ok=True)
    
    # In wireless mode with connected device, show connection banner
    if (args.wireless or args.room) and info.get('wireless_devices', 0) > 0:
        print(f"\n{Fore.GREEN}{'=' * 60}{Fore.RESET}")
        print(f"{Fore.GREEN}🎯 ESP32-P4 HowdyScreen Connected Successfully!{Fore.RESET}")
        print(f"{Fore.GREEN}{'=' * 60}{Fore.RESET}\n")
        print(f"{Fore.CYAN}Now loading voice assistant components...{Fore.RESET}\n")
    
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
            samples, sample_rate = kokoro.create("Hey there Partner, hows it hanging?", 
                                                voice=Config.KOKORO_VOICE, 
                                                speed=0.8, 
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
    
    # Preload Ollama model
    print(Fore.YELLOW + "Initializing Ollama LLM model..." + Fore.RESET)
    try:
        if preload_ollama_model():
            print(Fore.GREEN + "Ollama LLM model loaded successfully!" + Fore.RESET)
        else:
            print(Fore.RED + f"Warning: Failed to preload Ollama model: {Config.OLLAMA_LLM}" + Fore.RESET)
            print(Fore.YELLOW + "Will attempt to load on first use" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + f"Warning: Failed to preload Ollama model: {e}" + Fore.RESET)
        print(Fore.YELLOW + "Will attempt to load on first use" + Fore.RESET)
    
    # Initialize chat history
    chat_history = [
        {"role": "system", "content": """ You are a helpful Assistant called Howdy. 
         You are friendly and fun and you will help the users with their requests.
         Your answers are short and concise. """}
    ]
    
    # Start hotkey manager for runtime audio source switching
    if start_hotkeys():
        print(f"{Fore.GREEN}Runtime hotkeys enabled:{Fore.RESET}")
        print(f"  {Fore.CYAN}Ctrl+Alt+L{Fore.RESET} - Switch to local microphone")
        print(f"  {Fore.CYAN}Ctrl+Alt+W{Fore.RESET} - Switch to wireless microphone")
        print(f"  {Fore.CYAN}Ctrl+Alt+T{Fore.RESET} - Toggle audio source")
        print(f"  {Fore.CYAN}Ctrl+Alt+I{Fore.RESET} - Show audio source info")
        print(f"  {Fore.CYAN}Ctrl+Alt+D{Fore.RESET} - List wireless devices")
    else:
        print(f"{Fore.YELLOW}Runtime hotkeys disabled (keyboard module not available){Fore.RESET}")
    
    # Flag to track if we're currently playing audio
    playback_complete_event = threading.Event()
    playback_complete_event.set()  # Initially set to True since no playback is happening
    
    # Ensure activation sound flag is initially cleared
    activation_sound_playing.clear()  # Initially clear since no activation sound is playing
    
    # Make sure first turn flag is initially cleared
    is_first_turn_after_wake.clear()  # Initially clear

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
                        logging.warning("Wake word detector not initialized, restarting...")
                    if not safe_start_wake_word_detection():
                        # If we can't restart, wait a bit and try again
                        time.sleep(5)
                        continue
                
                time.sleep(0.1)  # Short sleep to prevent busy waiting
                continue
            
            # First, ensure that any previous playback is complete
            if not playback_complete_event.is_set():
                logging.info("Waiting for audio playback to complete before continuing...")
                playback_complete_event.wait()
                logging.info("Audio playback complete, continuing")
            
            # Check if a new conversation is starting (wake word just detected)
            if wake_word_detected.is_set():
                # Clear the wake word detected flag for next time
                wake_word_detected.clear()
                
                # Check if activation sound is playing
                if activation_sound_playing.is_set():
                    logging.info("Activation sound is playing in background")
                    time.sleep(0.1)
                else:
                    logging.info("No activation sound playing")
                
                logging.info("Starting recording now...")
            
            # Record audio from the microphone and save it
            if conversation_active.is_set():
                # In active conversation, give a visual indicator we're listening
                print(Fore.GREEN + "Listening..." + Fore.RESET)
                # Update LED matrix to "Listening" (LED state is already set in handle_wake_word)
                # The second set_listening call was here - removed to fix duplicate updates
            
            # Check if we need to apply wake word filtering
            apply_wake_word_filter = is_first_turn_after_wake.is_set()
            
            # Try to refresh wireless selection before recording
            if audio_source_manager and audio_source_manager.get_current_source() != AudioSourceType.WIRELESS:
                audio_source_manager.switch_to_wireless(audio_source_manager.target_room)

            if _is_wireless_audio_active():
                logging.info("📡 Using ESP32-P4 wireless audio recorder")
                network_source = getattr(audio_source_manager, "_network_source", None)
                if network_source:
                    recording_success = network_source.record_audio(
                        Config.INPUT_AUDIO,
                        is_wake_word_response=apply_wake_word_filter
                    )
                else:
                    recording_success = audio_source_manager.record_audio(
                        Config.INPUT_AUDIO,
                        is_wake_word_response=apply_wake_word_filter
                    )
            else:
                logging.info("🎙️ Wireless audio inactive, falling back to local microphone")
                recording_success = record_audio(
                    Config.INPUT_AUDIO,
                    is_wake_word_response=apply_wake_word_filter
                )
            
            # Clear the first turn flag after recording
            is_first_turn_after_wake.clear()

            # Check if recording was successful
            if not recording_success:
                logging.warning("Failed to record audio, returning to wake word detection")
                # Clear wake word detection flag and conversation state
                wake_word_detected.clear()
                conversation_active.clear()
                # Reset LED matrix to waiting
                if led_matrix:
                    led_matrix.set_waiting()
                print(Fore.YELLOW + "Sorry, I'm having trouble hearing you. Say 'Hey Howdy' to try again!" + Fore.RESET)
                
                # Aggressive cleanup to prevent memory leaks and segfaults
                logging.info("Performing aggressive cleanup after recording failure...")
                cleanup_all_detectors()
                import gc
                gc.collect()
                time.sleep(0.5)  # Brief pause for cleanup
                
                # Force restart wake word detection
                if not safe_start_wake_word_detection():
                    logging.error("Failed to restart wake word detection after recording failure")
                    time.sleep(2)
                    if not safe_start_wake_word_detection():
                        logging.error("Failed to restart wake word detection on second attempt")
                        break  # Exit main loop if we can't restart wake word detection
                
                continue  # Go back to wake word detection

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
                        # Brief delay to prevent TTS stuttering on immediate feedback
                        time.sleep(0.1)
                        play_audio(first_chunk_file)
                        delete_file(first_chunk_file)
                    continue
                else:
                    # If waiting for wake word, go back to waiting
                    continue
                
            logging.info(Fore.GREEN + "You said: " + user_input + Fore.RESET)

            # Check if the user wants to exit the program (complete shutdown)
            if "shut down howdy program" in user_input.lower() or "shut down the howdy program" in user_input.lower() or "code phrase exit" in user_input.lower():
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
                    # Brief delay to prevent TTS stuttering on conversation end
                    time.sleep(0.1)
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
                    
                    # Run targeted garbage collection to free audio resources
                    cleaned_count = targeted_gc()
                    logging.info(f"Cleaned up {cleaned_count} audio-related objects")
                    
                    # Wait for resources to be completely released
                    time.sleep(0.5)  # Short pause to allow resource cleanup
                    
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
            
            # Determine adaptive delays based on response complexity
            response_length = len(response_text)
            logging.info(f"Response length: {response_length} characters")
            
            if response_length < 100:
                playback_delay = 0.1  # Short responses
            elif response_length < 500:
                playback_delay = 0.2  # Medium responses
            else:
                playback_delay = 0.3  # Long responses - more stabilization time
            
            # Get just the first chunk
            try:
                success, first_chunk_file = text_to_speech(
                    Config.TTS_MODEL, 
                    tts_api_key, 
                    response_text, 
                    output_file, 
                    Config.LOCAL_MODEL_PATH
                )
            except Exception as tts_error:
                logging.error(f"TTS generation failed: {tts_error}")
                success = False
                first_chunk_file = None
            
            # List to track files for cleanup
            files_to_cleanup = []
            
            if success and first_chunk_file:
                # Add first chunk to cleanup list
                files_to_cleanup.append(first_chunk_file)
                
                # Define a thread to handle playback of all chunks with enhanced monitoring and timing
                def play_all_chunks():
                    try:
                        # Enhanced adaptive delay with detailed logging
                        logging.info(f"Using {playback_delay:.3f}s stabilization delay for {response_length} character response")
                        time.sleep(playback_delay)
                        
                        # Play the first chunk after the stabilization delay with timing
                        first_chunk_start = time.time()
                        logging.info(f"Starting playback of first chunk after {playback_delay:.3f}s stabilization")
                        
                        wireless_active = _is_wireless_audio_active()

                        # Play audio locally only when not actively using the wireless path
                        if not wireless_active:
                            logging.info("🎧 Wireless speaker unavailable – playing TTS locally")
                            play_audio(first_chunk_file)

                        # Send to ESP32-P4 devices if using wireless audio
                        if wireless_active and hasattr(audio_source_manager, '_network_source'):
                            try:
                                success = audio_source_manager._network_source.send_tts_audio_to_devices(
                                    first_chunk_file, 
                                    response_text[:50] + ('...' if len(response_text) > 50 else '')
                                )
                                if success:
                                    logging.info(f"📡 Sent TTS chunk to ESP32-P4 devices")
                                else:
                                    logging.warning(f"⚠️ Failed to send TTS chunk to ESP32-P4 devices")
                            except Exception as e:
                                logging.error(f"❌ Error sending TTS to ESP32-P4: {e}")
                        
                        first_chunk_duration = time.time() - first_chunk_start
                        logging.info(f"First chunk playback completed in {first_chunk_duration:.3f}s")
                        
                        # Continue playing chunks with enhanced monitoring and gap analysis
                        chunk_index = 1
                        last_chunk_time = time.time()
                        total_gaps = []
                        
                        while True:
                            wireless_active = _is_wireless_audio_active()
                            # Get comprehensive generation stats for debugging
                            stats = get_chunk_generation_stats()
                            
                            # Check if there are more chunks to play
                            next_chunk = get_next_chunk()
                            
                            # If no more chunks and generation is complete, we're done
                            if next_chunk is None and generation_complete.is_set():
                                avg_gap = sum(total_gaps) / len(total_gaps) if total_gaps else 0
                                logging.info(f"All chunks played successfully. Average inter-chunk gap: {avg_gap:.3f}s")
                                break
                            
                            # If we got a chunk, play it with enhanced monitoring
                            if next_chunk:
                                current_time = time.time()
                                inter_chunk_time = current_time - last_chunk_time
                                total_gaps.append(inter_chunk_time)
                                
                                files_to_cleanup.append(next_chunk)
                                
                                # Enhanced gap monitoring with more detailed analysis
                                if inter_chunk_time > 2.5:
                                    logging.warning(f"Long gap detected: {inter_chunk_time:.3f}s before chunk {chunk_index+1}")
                                    # Brief stabilization for very long gaps
                                    time.sleep(0.05)
                                elif inter_chunk_time > 1.5:
                                    logging.info(f"Moderate gap: {inter_chunk_time:.3f}s before chunk {chunk_index+1}")
                                
                                chunk_play_start = time.time()
                                logging.info(f"Playing chunk {chunk_index+1} (gap: {inter_chunk_time:.3f}s, queue_size: {stats['queue_size']})")
                                
                                # Play audio locally only when not using wireless output
                                if not wireless_active:
                                    logging.debug("🎧 Playing follow-up TTS chunk locally")
                                    play_audio(next_chunk)

                                # Send to ESP32-P4 devices if using wireless audio
                                if wireless_active and hasattr(audio_source_manager, '_network_source'):
                                    try:
                                        audio_source_manager._network_source.send_tts_audio_to_devices(
                                            next_chunk, 
                                            f"chunk_{chunk_index+1}"
                                        )
                                    except Exception as e:
                                        logging.debug(f"Error sending TTS chunk {chunk_index+1} to ESP32-P4: {e}")
                                
                                chunk_play_time = time.time() - chunk_play_start
                                logging.debug(f"Chunk {chunk_index+1} playback took {chunk_play_time:.3f}s")
                                
                                chunk_index += 1
                                last_chunk_time = current_time
                            else:
                                # Enhanced adaptive wait time based on generation state
                                if stats['generation_complete']:
                                    logging.debug("Generation complete, no more chunks expected")
                                    break
                                elif stats['queue_empty']:
                                    # If generation is still active but queue is empty, wait longer
                                    logging.debug(f"Queue empty, generation status: {stats['status']}, waiting...")
                                    time.sleep(0.25)  # Slightly longer wait
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
            
            # Reset states to return to wake word detection
            wake_word_detected.clear()
            conversation_active.clear()
            if led_matrix:
                led_matrix.set_waiting()
            
            # Explicitly restart wake word detection after error
            cleanup_all_detectors()  # Force cleanup first
            time.sleep(2)  # Give more time for cleanup
            if not safe_start_wake_word_detection():
                logging.error("Failed to restart wake word detection after error")
                time.sleep(3)  # Wait longer before continuing
            
            time.sleep(1)
    
    # Cleanup
    if led_matrix:
        # Set matrix to waiting state before exiting
        led_matrix.set_waiting()
    
    # Clean up detectors and audio resources
    cleanup_all_detectors()
    
    # Perform targeted garbage collection
    cleaned_count = targeted_gc()
    print(f"{Fore.CYAN}Cleaned up {cleaned_count} audio resources{Fore.RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Shutting down due to keyboard interrupt...{Fore.RESET}")
        if led_matrix:
            led_matrix.set_waiting()
        cleanup_all_detectors()
        
        # Perform targeted garbage collection to clean up audio resources
        cleaned_count = targeted_gc()
        print(f"{Fore.CYAN}Cleaned up {cleaned_count} audio resources{Fore.RESET}")
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}{Fore.RESET}")
        import traceback
        traceback.print_exc()
        if led_matrix:
            led_matrix.set_waiting()
        cleanup_all_detectors()
        
        # Still try to clean up audio resources
        try:
            targeted_gc()
        except:
            pass
