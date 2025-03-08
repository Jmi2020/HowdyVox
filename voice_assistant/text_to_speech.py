# voice_assistant/text_to_speech.py
import logging
import json
import pyaudio
import elevenlabs
import soundfile as sf
import subprocess
import os
import shutil

from openai import OpenAI
from deepgram import DeepgramClient, SpeakOptions
from elevenlabs.client import ElevenLabs
from cartesia import Cartesia
from voice_assistant.config import Config
from voice_assistant.local_tts_generation import generate_audio_file_melotts

def text_to_speech(model: str, api_key:str, text:str, output_file_path:str, local_model_path:str=None):
    """
    Convert text to speech using the specified model.
    
    Args:
    model (str): The model to use for TTS ('openai', 'deepgram', 'elevenlabs', 'local', 'kokoro').
    api_key (str): The API key for the TTS service.
    text (str): The text to convert to speech.
    output_file_path (str): The path to save the generated speech audio file.
    local_model_path (str): The path to the local model (if applicable).
    """
    
    try:
        # Delete existing output file if it exists
        if os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
                logging.info(f"Removed existing file: {output_file_path}")
            except Exception as e:
                logging.warning(f"Could not remove existing file {output_file_path}: {e}")
        
        if model == 'openai':
            client = OpenAI(api_key=api_key)
            speech_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )

            speech_response.stream_to_file(output_file_path)

        elif model == 'deepgram':
            client = DeepgramClient(api_key=api_key)
            options = SpeakOptions(
                model="aura-arcas-en", #"aura-luna-en", # https://developers.deepgram.com/docs/tts-models
                encoding="linear16",
                container="wav"
            )
            SPEAK_OPTIONS = {"text": text}
            response = client.speak.v("1").save(output_file_path, SPEAK_OPTIONS, options)
        
        elif model == 'elevenlabs':
            client = ElevenLabs(api_key=api_key)
            audio = client.generate(
                text=text, 
                voice="Paul J.", 
                output_format="mp3_22050_32", 
                model="eleven_turbo_v2"
            )
            elevenlabs.save(audio, output_file_path)
        
        elif model == "cartesia":
            client = Cartesia(api_key=api_key)
            # voice_name = "Barbershop Man"
            voice_id = "f114a467-c40a-4db8-964d-aaba89cd08fa"#"a0e99841-438c-4a64-b679-ae501e7d6091"
            voice = client.voices.get(id=voice_id)

            # You can check out our models at https://docs.cartesia.ai/getting-started/available-models
            model_id = "sonic-english"

            # You can find the supported `output_format`s at https://docs.cartesia.ai/api-reference/endpoints/stream-speech-server-sent-events
            output_format = {
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            }

            p = pyaudio.PyAudio()
            rate = 44100

            stream = None

            # Generate and stream audio
            for output in client.tts.sse(
                model_id=model_id,
                transcript=text,
                voice_embedding=voice["embedding"],
                stream=True,
                output_format=output_format,
            ):
                buffer = output["audio"]

                if stream is None:
                    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

                # Write the audio data to the stream
                stream.write(buffer)
            
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

        elif model == "kokoro":
            # Define the kokoro command
            voice_model = Config.KOKORO_VOICE  # Use the voice from config
            if local_model_path:
                voice_model = local_model_path
                
            # Ensure the output path is accessible
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Get file extension
            _, file_ext = os.path.splitext(output_file_path)
            
            # Create a temporary file with .wav extension if the output is MP3
            temp_output_path = output_file_path
            if file_ext.lower() == '.mp3':
                temp_output_path = output_file_path + ".temp.wav"
                
            # Run kokoro as a subprocess
            cmd = ["kokoro", "-m", voice_model, "-t", text, "-o", temp_output_path]
            logging.info(f"Running kokoro command: {' '.join(cmd)}")
            
            try:
                process = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    check=True
                )
                
                # Check if the file was created
                if not os.path.exists(temp_output_path):
                    raise FileNotFoundError(f"Kokoro failed to generate audio file at {temp_output_path}")
                
                # If we need to convert from WAV to MP3
                if temp_output_path != output_file_path:
                    try:
                        from pydub import AudioSegment
                        sound = AudioSegment.from_wav(temp_output_path)
                        sound.export(output_file_path, format="mp3", bitrate="192k")
                        logging.info(f"Converted WAV to MP3: {output_file_path}")
                        
                        # Remove the temporary WAV file
                        os.remove(temp_output_path)
                    except Exception as conv_err:
                        logging.error(f"Error converting WAV to MP3: {conv_err}")
                        # If conversion fails, copy the WAV file as a fallback
                        shutil.copy2(temp_output_path, output_file_path)
                        logging.warning(f"Copied WAV file to output path as fallback")
                    
                # Verify the final output file exists
                if not os.path.exists(output_file_path):
                    raise FileNotFoundError(f"Failed to create final audio file at {output_file_path}")
                    
                logging.info(f"Kokoro TTS successfully generated audio file at {output_file_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Kokoro command failed: {e.stderr}")
                raise
            
        elif model == "melotts": # this is a local model
            try:
                result = generate_audio_file_melotts(text=text, filename=output_file_path)
                
                # Check if the API returned a different path than requested
                if result and isinstance(result, dict) and 'file_path' in result:
                    generated_path = result['file_path']
                    
                    # If the generated path is different from the requested one, copy the file
                    if generated_path != output_file_path and os.path.exists(generated_path):
                        logging.info(f"Copying audio from {generated_path} to {output_file_path}")
                        shutil.copy2(generated_path, output_file_path)
            except Exception as e:
                logging.error(f"MeloTTS failed, error: {str(e)}")
                # Fall back to Kokoro if MeloTTS fails
                logging.info("Falling back to Kokoro TTS")
                return text_to_speech("kokoro", api_key, text, output_file_path, local_model_path)
        
        elif model == 'local':
            with open(output_file_path, "wb") as f:
                f.write(b"Local TTS audio data")
        
        else:
            raise ValueError("Unsupported TTS model")
        
        # Verify the output file exists and has content
        if not os.path.exists(output_file_path):
            raise FileNotFoundError(f"Output file was not created at {output_file_path}")
            
        file_size = os.path.getsize(output_file_path)
        if file_size < 100:  # Suspiciously small file
            logging.warning(f"Warning: Generated audio file is very small ({file_size} bytes)")
            
        return True
        
    except Exception as e:
        logging.error(f"Failed to convert text to speech: {e}")
        return False