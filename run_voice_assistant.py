# run_voice_assistant.py

import logging
import time
import os
import threading
from colorama import Fore, init
from voice_assistant.audio import record_audio, play_audio, play_audio_chunks
from voice_assistant.transcription import transcribe_audio
from voice_assistant.response_generation import generate_response
from voice_assistant.text_to_speech import text_to_speech
from voice_assistant.utils import delete_file
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize colorama
init(autoreset=True)

def main():
    """
    Main function to run the offline voice assistant using FastWhisperAPI, Ollama, and Kokoro.
    """
    chat_history = [
        {"role": "system", "content": """ You are a helpful Assistant called Howdy. 
         You are friendly and fun and you will help the users with their requests.
         Your answers are short and concise. """}
    ]
    
    # Track files to clean up
    chunk_files_to_cleanup = []
    playback_complete_event = threading.Event()
    # Initially set to True since there's no previous playback when starting
    playback_complete_event.set()

    while True:
        try:
            # Make sure any previous playback has completed
            if not playback_complete_event.is_set():
                logging.info("Waiting for previous audio playback to complete...")
                playback_complete_event.wait()
                
            # Reset the event for the next cycle
            playback_complete_event.clear()
            
            # Record audio from the microphone and save it
            record_audio(Config.INPUT_AUDIO)

            # Get the API key for transcription (will be None for FastWhisperAPI)
            transcription_api_key = get_transcription_api_key()
            
            # Transcribe the audio file using FastWhisperAPI
            user_input = transcribe_audio(Config.TRANSCRIPTION_MODEL, transcription_api_key, Config.INPUT_AUDIO, Config.LOCAL_MODEL_PATH)

            # Check if the transcription is empty and restart the recording if it is
            if not user_input:
                logging.info("No transcription was returned. Starting recording again.")
                playback_complete_event.set()  # No audio to play, so mark as complete
                continue
                
            logging.info(Fore.GREEN + "You said: " + user_input + Fore.RESET)

            # Check if the user wants to exit the program
            if "goodbye" in user_input.lower() or "arrivederci" in user_input.lower():
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

            # For Kokoro, always use WAV output from the config
            output_file = Config.OUTPUT_AUDIO

            # Get the API key for TTS (will be None for Kokoro)
            tts_api_key = get_tts_api_key()

            # Convert the response text to speech using Kokoro with streaming
            success, chunk_queue = text_to_speech(
                Config.TTS_MODEL, 
                tts_api_key, 
                response_text, 
                output_file, 
                Config.LOCAL_MODEL_PATH,
                stream=True  # Enable streaming mode
            )
            
            if success:
                # Start a thread to play chunks as they become available
                def play_streamed_chunks():
                    all_chunks = []
                    
                    try:
                        # Keep playing chunks as they arrive in the queue
                        while True:
                            chunk_file = chunk_queue.get()
                            
                            # None signals the end of chunks
                            if chunk_file is None:
                                break
                                
                            all_chunks.append(chunk_file)
                            
                            # Play this chunk
                            logging.info(f"Playing chunk: {chunk_file}")
                            play_audio(chunk_file)
                            
                            # Small delay for natural pause between chunks
                            time.sleep(0.2)
                    except Exception as e:
                        logging.error(f"Error in streaming playback: {e}")
                    finally:
                        # Add all chunks to the cleanup list
                        chunk_files_to_cleanup.extend(all_chunks)
                        
                        # Signal that playback is complete
                        playback_complete_event.set()
                        
                        # Clean up chunk files
                        for chunk_file in all_chunks:
                            try:
                                delete_file(chunk_file)
                            except Exception as e:
                                logging.warning(f"Could not delete chunk file {chunk_file}: {e}")
                
                # Start the playback thread
                playback_thread = threading.Thread(target=play_streamed_chunks)
                playback_thread.daemon = True
                playback_thread.start()
                
                # Continue with the next cycle. The next recording will wait 
                # for playback_complete_event to be set
            else:
                logging.error("Failed to generate speech, skipping playback")
                # Signal that there's no playback happening
                playback_complete_event.set()
            
            # Clean up audio files - uncomment if you want to delete after use
            # delete_file(Config.INPUT_AUDIO)

        except Exception as e:
            logging.error(Fore.RED + f"An error occurred: {e}" + Fore.RESET)
            delete_file(Config.INPUT_AUDIO)
            if 'output_file' in locals():
                delete_file(output_file)
            
            # Signal that any playback has completed (due to error)
            playback_complete_event.set()
            
            time.sleep(1)

if __name__ == "__main__":
    main()