# voice_assistant/mac_voice_isolation.py

import os
import sys
import logging
import numpy as np
import struct
import threading
import queue
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

# PyObjC imports
try:
    from Foundation import NSObject, NSError
    from AVFoundation import (
        AVAudioEngine, AVAudioFormat, AVAudioPCMBuffer,
        AVAudioNode, AVAudioConnectionPoint,
        AVAudioSession, AVAudioSessionCategoryRecord,
        AVAudioSessionModeVoiceChat, AVAudioPCMFormatFloat32
    )
    import objc

    # Try to load AudioToolbox framework functions
    # AudioUnitSetProperty and related functions are C functions from AudioToolbox
    # They need to be loaded via objc.loadBundle or ctypes
    AUDIO_TOOLBOX_AVAILABLE = False
    AudioUnitSetProperty = None
    kAudioUnitScope_Global = None

    try:
        # Load AudioToolbox framework bundle
        bundle_dict = {}
        objc.loadBundle(
            'AudioToolbox',
            bundle_dict,
            bundle_path='/System/Library/Frameworks/AudioToolbox.framework'
        )

        # Check if key functions are available
        if 'AudioUnitSetProperty' in bundle_dict:
            AudioUnitSetProperty = bundle_dict['AudioUnitSetProperty']
            kAudioUnitScope_Global = bundle_dict.get('kAudioUnitScope_Global', 0)
            AUDIO_TOOLBOX_AVAILABLE = True
            logging.info("AudioToolbox framework functions loaded successfully")
        else:
            logging.warning("AudioToolbox framework loaded but functions not found")
    except Exception as e:
        logging.warning(f"AudioToolbox framework functions not available: {e}")

    PYOBJC_AVAILABLE = True
except ImportError as e:
    logging.error(f"PyObjC not available: {e}")
    PYOBJC_AVAILABLE = False
    AUDIO_TOOLBOX_AVAILABLE = False

# Define audio unit property constants
kAudioUnitProperty_BypassEffect = 0x0000000c
kAudioUnitProperty_SetRenderCallback = 0x00000017
kAUVoiceIOProperty_BypassVoiceProcessing = 0x00000842
kAUVoiceIOProperty_MuteOutput = 0x00000843
kAUVoiceIOProperty_VoiceProcessingEnableAGC = 0x00000844
kAUVoiceIOProperty_VoiceProcessingQuality = 0x00000845

# Voice processing quality levels
kAUVoiceIOProperty_VoiceProcessingQuality_Max = 0x7f
kAUVoiceIOProperty_VoiceProcessingQuality_High = 0x60
kAUVoiceIOProperty_VoiceProcessingQuality_Medium = 0x40
kAUVoiceIOProperty_VoiceProcessingQuality_Low = 0x20

@dataclass
class VoiceIsolationConfig:
    """Configuration for Voice Isolation."""
    sample_rate: int = 48000
    channels: int = 1
    quality: int = kAUVoiceIOProperty_VoiceProcessingQuality_High
    enable_agc: bool = True
    enable_noise_suppression: bool = True
    buffer_size: int = 512


# Only define the real class if PyObjC is available
if PYOBJC_AVAILABLE:
    class MacVoiceIsolation:
        """
        macOS Native Voice Isolation using AVAudioEngine.

        This class provides real-time voice isolation using Apple's
        built-in Neural Engine accelerated voice processing.
        """

        def __init__(self, config: Optional[VoiceIsolationConfig] = None):
            """
            Initialize Voice Isolation.

            Args:
                config: Configuration settings
            """
            self.config = config or VoiceIsolationConfig()

            # Audio engine and nodes
            self.engine = None
            self.input_node = None
            self.voice_processing_enabled = False

            # Audio format
            self.audio_format = None

            # Processing state
            self.is_running = False
            self.process_callback = None

            # Buffers
            self.input_buffer = queue.Queue(maxsize=100)
            self.output_buffer = queue.Queue(maxsize=100)

            # Initialize the audio engine
            self._setup_audio_engine()

            logging.info(f"MacVoiceIsolation initialized (sample_rate={self.config.sample_rate})")
    
        def _setup_audio_engine(self):
            """Set up AVAudioEngine with voice processing."""
            try:
                # Create audio engine
                self.engine = AVAudioEngine.alloc().init()
            
                # Get input node
                self.input_node = self.engine.inputNode()
            
                # Create audio format
                self.audio_format = AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
                    AVAudioPCMFormatFloat32,
                    self.config.sample_rate,
                    self.config.channels,
                    True
                )
            
                # Configure input node for voice processing
                self._configure_voice_processing()
            
                # Install tap to capture processed audio
                self._install_audio_tap()
            
            except Exception as e:
                logging.error(f"Failed to setup audio engine: {e}")
                raise
    
        def _configure_voice_processing(self):
            """Configure voice processing on the input node."""
            try:
                # Get the audio unit from input node
                audio_unit = self.input_node.audioUnit()
            
                if not audio_unit:
                    raise RuntimeError("Failed to get audio unit from input node")
            
                # Enable voice processing mode
                # This automatically enables Neural Engine acceleration
                success = self._set_audio_unit_property(
                    audio_unit,
                    kAUVoiceIOProperty_BypassVoiceProcessing,
                    0  # 0 = Don't bypass (enable processing)
                )
            
                if not success:
                    logging.warning("Failed to enable voice processing")
            
                # Set voice processing quality
                success = self._set_audio_unit_property(
                    audio_unit,
                    kAUVoiceIOProperty_VoiceProcessingQuality,
                    self.config.quality
                )
            
                if success:
                    quality_name = self._get_quality_name(self.config.quality)
                    logging.info(f"Voice processing quality set to: {quality_name}")
            
                # Enable automatic gain control if requested
                if self.config.enable_agc:
                    success = self._set_audio_unit_property(
                        audio_unit,
                        kAUVoiceIOProperty_VoiceProcessingEnableAGC,
                        1  # Enable AGC
                    )
                
                    if success:
                        logging.info("Automatic Gain Control enabled")
            
                self.voice_processing_enabled = True
            
            except Exception as e:
                logging.error(f"Failed to configure voice processing: {e}")
                self.voice_processing_enabled = False
    
        def _set_audio_unit_property(self, audio_unit, property_id: int,
                                    value: int) -> bool:
            """
            Set an audio unit property.

            Args:
                audio_unit: The audio unit
                property_id: Property ID to set
                value: Value to set

            Returns:
                bool: Success status
            """
            if not AUDIO_TOOLBOX_AVAILABLE or AudioUnitSetProperty is None:
                logging.debug(f"AudioToolbox not available, skipping property {property_id}")
                return False

            try:
                # Create value buffer
                value_data = struct.pack('I', value)

                # Set the property using AudioToolbox
                scope = kAudioUnitScope_Global if kAudioUnitScope_Global is not None else 0
                result = AudioUnitSetProperty(
                    audio_unit,
                    property_id,
                    scope,
                    0,  # Element
                    value_data,
                    len(value_data)
                )

                return result == 0  # noErr

            except Exception as e:
                logging.error(f"Failed to set audio unit property {property_id}: {e}")
                return False
    
        def _get_quality_name(self, quality: int) -> str:
            """Get human-readable quality name."""
            quality_map = {
                kAUVoiceIOProperty_VoiceProcessingQuality_Max: "Maximum",
                kAUVoiceIOProperty_VoiceProcessingQuality_High: "High",
                kAUVoiceIOProperty_VoiceProcessingQuality_Medium: "Medium",
                kAUVoiceIOProperty_VoiceProcessingQuality_Low: "Low"
            }
            return quality_map.get(quality, f"Custom ({quality})")
    
        def _install_audio_tap(self):
            """Install tap on input node to capture processed audio."""
        
            def tap_block(buffer, time):
                """
                Callback for audio tap.
            
                This receives voice-isolated audio from the Neural Engine.
                """
                if not self.is_running:
                    return
            
                # Convert AVAudioPCMBuffer to numpy array
                frame_count = buffer.frameLength()
                channels = buffer.format().channelCount()
            
                # Get float samples
                samples = buffer.floatChannelData()[0]
            
                # Create numpy array from samples
                audio_data = np.zeros(frame_count, dtype=np.float32)
                for i in range(frame_count):
                    audio_data[i] = samples[i]
            
                # Add to output buffer
                try:
                    self.output_buffer.put_nowait(audio_data)
                except queue.Full:
                    # Drop oldest if buffer is full
                    try:
                        self.output_buffer.get_nowait()
                        self.output_buffer.put_nowait(audio_data)
                    except:
                        pass
            
                # Call process callback if set
                if self.process_callback:
                    self.process_callback(audio_data)
        
            # Install the tap
            self.input_node.installTapOnBus_bufferSize_format_block_(
                0,  # Bus 0
                self.config.buffer_size,
                self.audio_format,
                tap_block
            )
        
            logging.info("Audio tap installed successfully")
    
        def start(self):
            """Start the voice isolation engine."""
            if self.is_running:
                logging.warning("Voice isolation already running")
                return
        
            try:
                # Prepare audio session for recording
                session = AVAudioSession.sharedInstance()
                error = None
            
                # Set category to record
                success = session.setCategory_mode_options_error_(
                    AVAudioSessionCategoryRecord,
                    AVAudioSessionModeVoiceChat,
                    0,
                    error
                )
            
                if not success:
                    logging.error(f"Failed to set audio session category: {error}")
            
                # Activate session
                success = session.setActive_error_(True, error)
            
                if not success:
                    logging.error(f"Failed to activate audio session: {error}")
            
                # Start the engine
                self.engine.startAndReturnError_(error)
            
                if error:
                    raise RuntimeError(f"Failed to start audio engine: {error}")
            
                self.is_running = True
                logging.info("Voice isolation started")
            
            except Exception as e:
                logging.error(f"Failed to start voice isolation: {e}")
                raise
    
        def stop(self):
            """Stop the voice isolation engine."""
            if not self.is_running:
                return
        
            try:
                self.is_running = False
            
                # Stop the engine
                self.engine.stop()
            
                # Remove the tap
                self.input_node.removeTapOnBus_(0)
            
                # Clear buffers
                while not self.output_buffer.empty():
                    self.output_buffer.get_nowait()
            
                logging.info("Voice isolation stopped")
            
            except Exception as e:
                logging.error(f"Error stopping voice isolation: {e}")
    
        def read_processed_audio(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
            """
            Read processed audio from the output buffer.
        
            Args:
                timeout: Maximum time to wait for audio
            
            Returns:
                Processed audio array or None if timeout
            """
            try:
                return self.output_buffer.get(timeout=timeout)
            except queue.Empty:
                return None
    
        def set_process_callback(self, callback: Callable[[np.ndarray], None]):
            """
            Set a callback to be called with each processed audio chunk.
        
            Args:
                callback: Function to call with audio data
            """
            self.process_callback = callback
    
        def get_statistics(self) -> dict:
            """Get processing statistics."""
            return {
                'is_running': self.is_running,
                'voice_processing_enabled': self.voice_processing_enabled,
                'buffer_size': self.output_buffer.qsize(),
                'sample_rate': self.config.sample_rate,
                'quality': self._get_quality_name(self.config.quality)
            }


    # If PyObjC is not available, create a disabled stub class
if not PYOBJC_AVAILABLE:
    logging.warning("macOS Voice Isolation not available - PyObjC framework missing")

    class MacVoiceIsolation:
        """Disabled stub for MacVoiceIsolation when PyObjC is not available."""

        def __init__(self, config: Optional[VoiceIsolationConfig] = None):
            """Initialize disabled voice isolation."""
            raise RuntimeError(
                "macOS Voice Isolation requires PyObjC framework. "
                "Install with: pip install pyobjc-framework-AVFoundation pyobjc-framework-CoreAudio"
            )