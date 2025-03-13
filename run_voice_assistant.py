# run_voice_assistant.py

import logging
import time
import os
import threading
from colorama import Fore, init
from voice_assistant.audio import record_audio, play_audio
from voice_assistant.transcription import transcribe_audio
from voice_assistant.response_generation import generate_response
from voice_assistant.text_to_speech import text_to_speech, get_next_chunk, generation_complete
from voice_assistant.utils import delete_file
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key
from voice_assistant.kokoro_manager import KokoroManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize colorama
init(autoreset=True)

def main():
    """
    Main function to run the offline voice assistant using FastWhisperAPI, Ollama, and Kokoro.
    """
    # Preload Kokoro model
    print(Fore.YELLOW + "Initializing Kokoro TTS model..." + Fore.RESET)
    try:
        KokoroManager.get_instance(local_model_path=Config.LOCAL_MODEL_PATH)
        print(Fore.GREEN + "Kokoro TTS model loaded successfully!" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + f"Warning: Failed to preload Kokoro TTS model: {e}" + Fore.RESET)
        print(Fore.YELLOW + "Will attempt to load on first use" + Fore.RESET)
    
    chat_history = [
        {"role": "system", "content": """ You are a helpful Assistant called Howdy. 
         You are friendly and fun and you will help the users with their requests.
         Your answers are short and concise. """}
    ]
    
    # Flag to track if we're currently playing audio
    playback_complete_event = threading.Event()
    playback_complete_event.set()  # Initially set to True since no playback is happening

    print(Fore.CYAN + "Howdy TTS is ready! Say something..." + Fore.RESET)
    
    while True:
        try:
            # Make sure previous playback is complete before recording
            if not playback_complete_event.is_set():
                logging.info("Waiting for previous audio playback to complete...")
                playback_complete_event.wait()
            
            # Record audio from the microphone and save it
            record_audio(Config.INPUT_AUDIO)

            # Get the API key for transcription (will be None for FastWhisperAPI)
            transcription_api_key = get_transcription_api_key()
            
            # Transcribe the audio file using FastWhisperAPI
            user_input = transcribe_audio(Config.TRANSCRIPTION_MODEL, transcription_api_key, Config.INPUT_AUDIO, Config.LOCAL_MODEL_PATH)

            # Check if the transcription is empty and restart the recording if it is
            if not user_input:
                logging.info("No transcription was returned. Starting recording again.")
                continue
                
            logging.info(Fore.GREEN + "You said: " + user_input + Fore.RESET)

            # Check if the user wants to exit the program
            if "goodbye" in user_input.lower() or "arrivederci" in user_input.lower():
                print(Fore.YELLOW + "Goodbye, partner! Happy trails!" + Fore.RESET)
                break

            # Append the user's input to the chat history
            chat_history.append({"role": "user", "content": user_input})

            # Get the API key for response generation (will be None for Ollama)
            response_api_key = get_response_api_key()

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
                
                # Start playback thread
                playback_thread = threading.Thread(target=play_all_chunks)
                playback_thread.daemon = True
                playback_thread.start()
            else:
                logging.error("Failed to generate speech")
                playback_complete_event.set()

        except Exception as e:
            logging.error(Fore.RED + f"An error occurred: {e}" + Fore.RESET)
            delete_file(Config.INPUT_AUDIO)
            playback_complete_event.set()
            time.sleep(1)

if __name__ == "__main__":
    main()