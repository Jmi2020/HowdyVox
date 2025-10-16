#!/usr/bin/env python3
"""
Ultra-light audio analyzer for HowdyVox TTS
Processes PCM audio to extract expressive features for face animation

Features extracted:
- RMS envelope (volume/energy) → eye size scaling
- Zero-Crossing Rate (ZCR) (sibilance/brightness) → horizontal eye squeeze
- Peak detection (onset/emphasis) → head nod

Uses stdlib audioop (C-accelerated) for minimal CPU overhead
"""

from __future__ import annotations
import audioop
import socket
import time
from array import array
import logging

logging.basicConfig(level=logging.INFO)


class ReactiveMeter:
    """
    Real-time audio analyzer for TTS expressiveness.

    Processes PCM audio chunks and sends UDP control messages to face renderer.
    """

    def __init__(
        self,
        samplerate=24000,
        sample_width=2,
        channels=1,
        udp_host="127.0.0.1",
        udp_port=31337,
        win_ms=20,
        update_hz=12,
    ):
        """
        Initialize the reactive meter.

        Args:
            samplerate: Audio sample rate (Hz)
            sample_width: Bytes per sample (2 for 16-bit)
            channels: Number of audio channels
            udp_host: Hostname/IP for face renderer
            udp_port: UDP port for face renderer
            win_ms: Analysis window size (ms)
            update_hz: Update rate for face (messages per second)
        """
        self.sr = samplerate
        self.sw = sample_width
        self.ch = channels
        self.host = udp_host
        self.port = udp_port
        self.win = int(self.sr * win_ms / 1000)
        self.buf = bytearray()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Envelope tracking with AGC (Automatic Gain Control)
        self.env = 0.0
        self.noise_floor = 200.0  # Tune based on your TTS
        self.crest = 2000.0  # Running maximum (adaptive)
        self.last_send = 0.0
        self.send_interval = 1.0 / update_hz
        self.last_peak_t = 0.0  # ms

        # State tracking
        self.active = False
        self.last_features = {"env": 0.0, "zcr": 0.0, "peak": False}

        logging.info(f"ReactiveMeter initialized: {samplerate}Hz, {channels}ch, "
                    f"→ {udp_host}:{udp_port} @ {update_hz}Hz")

    def _mono16(self, data: bytes) -> bytes:
        """Convert to mono if needed"""
        if self.ch == 1:
            return data
        return audioop.tomono(data, self.sw, 0.5, 0.5)

    def _rms(self, mono16: bytes) -> float:
        """Calculate RMS (Root Mean Square) - volume/energy"""
        return float(audioop.rms(mono16, self.sw))  # 0..32767

    def _zcr(self, mono16: bytes) -> float:
        """
        Calculate Zero-Crossing Rate - sibilance/brightness proxy.

        High ZCR = lots of zero crossings = sibilant sounds (s, sh, ch, f)
        Low ZCR = few zero crossings = vowels (a, e, i, o, u)
        """
        a = array("h")
        a.frombytes(mono16)
        n = len(a)
        if n < 2:
            return 0.0

        crossings = 0
        prev = a[0]
        for i in range(1, n):
            cur = a[i]
            if (prev < 0 and cur > 0) or (prev > 0 and cur < 0):
                crossings += 1
            prev = cur

        return min(1.0, crossings / (n - 1))

    def _smooth_env(self, x: float) -> float:
        """
        Smooth envelope with attack/decay and AGC.

        AGC (Automatic Gain Control) keeps output in 0-1 range
        even as speech volume varies.
        """
        # Update crest (running maximum)
        if x > self.crest:
            self.crest = 0.9 * self.crest + 0.1 * x  # Fast rise
        else:
            self.crest = 0.999 * self.crest + 0.001 * x  # Slow decay

        # Calculate normalized level
        ref = max(self.noise_floor, self.crest * 0.6)
        lvl = max(0.0, min(1.0, (x - self.noise_floor) / (ref - self.noise_floor)))

        # Attack/Decay smoothing (fast rise, slower fall)
        attack_a, decay_a = 0.35, 0.10
        if lvl > self.env:
            self.env = (1 - attack_a) * self.env + attack_a * lvl
        else:
            self.env = (1 - decay_a) * self.env + decay_a * lvl

        return self.env

    def _peak(self, lvl: float) -> bool:
        """
        Detect peaks (emphasis/onset) with refractory period.

        Triggers brief head nod animation.
        """
        now = time.time() * 1000.0
        # Detect peaks above threshold with 180ms refractory
        if lvl > 0.55 and (now - self.last_peak_t) > 180.0:
            self.last_peak_t = now
            return True
        return False

    def process(self, pcm_chunk: bytes):
        """
        Feed raw PCM bytes (16-bit signed) from TTS.

        Call this for each audio chunk you're about to play.
        """
        self.buf.extend(pcm_chunk)
        bytes_per_frame = self.sw * self.ch
        need = self.win * bytes_per_frame

        while len(self.buf) >= need:
            frame = self.buf[:need]
            del self.buf[:need]

            # Analyze frame
            mono = self._mono16(frame)
            rms = self._rms(mono)
            env = self._smooth_env(rms)
            zcr = self._zcr(mono)

            # Store features
            self.last_features = {
                "env": env,
                "zcr": zcr,
                "peak": self._peak(env),
            }

    def tick(self):
        """
        Send face update message.

        Call this ~10-15 times per second (matches update_hz).
        """
        now = time.time()
        if now - self.last_send < self.send_interval:
            return

        self.last_send = now

        env = self.last_features.get("env", 0.0)
        zcr = self.last_features.get("zcr", 0.0)
        peak = 1 if self.last_features.get("peak", False) else 0

        # Determine state based on envelope
        if env > 0.07:
            state = "speaking"
        else:
            state = "idle"

        # Format message
        msg = f"{state}:{env:.3f};zcr={zcr:.3f};peak={peak}"

        # Send UDP message
        try:
            self.sock.sendto(msg.encode("utf-8"), (self.host, self.port))
        except OSError as e:
            logging.debug(f"UDP send error: {e}")

    def start(self):
        """Mark meter as active"""
        self.active = True
        self.send_state("speaking")  # Initial state

    def stop(self):
        """Mark meter as inactive and send idle state"""
        self.active = False
        self.send_state("idle")

    def send_state(self, state: str):
        """Send a simple state message without audio features"""
        try:
            self.sock.sendto(state.encode("utf-8"), (self.host, self.port))
        except OSError as e:
            logging.debug(f"UDP send error: {e}")


# Example usage
if __name__ == "__main__":
    import numpy as np

    print("🔊 Testing ReactiveMeter")
    print("=" * 50)

    # Create meter
    meter = ReactiveMeter(
        samplerate=24000,
        sample_width=2,
        channels=1,
        udp_host="127.0.0.1",
        udp_port=31337,
        update_hz=12,
    )

    # Generate test audio: sine wave with varying amplitude
    print("Generating test audio (5 seconds)...")
    duration = 5.0
    chunk_size = 2048
    t = 0

    meter.start()

    while t < duration:
        # Generate chunk
        samples = []
        for i in range(chunk_size):
            # Sine wave at 440 Hz with amplitude modulation
            phase = 2 * np.pi * 440 * (t + i / 24000)
            amplitude = 0.3 + 0.7 * abs(np.sin(t * 2))  # Varying volume
            sample = int(16000 * amplitude * np.sin(phase))
            samples.append(sample)

        # Convert to bytes
        pcm_chunk = array("h", samples).tobytes()

        # Process
        meter.process(pcm_chunk)
        meter.tick()

        t += chunk_size / 24000
        time.sleep(chunk_size / 24000)  # Simulate real-time

    meter.stop()
    print("Test complete!")
