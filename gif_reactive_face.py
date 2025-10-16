#!/usr/bin/env python3
"""
Audio-Reactive GIF Face Renderer for HowdyVox
Loads pre-rendered GIF animations and controls playback based on audio features
"""

import pygame as pg
from PIL import Image
import socket
import threading
import queue
import time
import os

# Configuration
CFG = {
    "size": 200,                    # Display window size (will scale GIFs)
    "gif_dir": "faceStates",        # Directory containing GIF files
    "udp_port": 31337,              # UDP port for audio features
    "fps": 30,                      # Base FPS for rendering
}

# GIF file mapping to states
GIF_MAP = {
    "idle": "waiting_blink_loop.gif",
    "listening": "listening_glow_loop.gif",
    "thinking": "thinking_stars_motion.gif",
    "speaking": "speaking_face.gif",
}


class UdpEvents:
    """Ultra-light UDP listener for audio feature messages."""

    def __init__(self, port, q):
        self.addr = ("0.0.0.0", port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        self.q = q
        self._stop = threading.Event()

    def start(self):
        t = threading.Thread(target=self._rx, daemon=True)
        t.start()

    def _rx(self):
        self.sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(256)
                msg = data.decode("utf-8", "ignore").strip()
                if msg:
                    try:
                        self.q.put_nowait(msg)
                    except queue.Full:
                        pass
            except socket.timeout:
                pass
            except OSError:
                break

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class GifAnimation:
    """
    Stores frames from a GIF and provides audio-reactive playback.
    """

    def __init__(self, gif_path, display_size=200):
        """
        Load GIF frames and prepare for audio-reactive playback.

        Args:
            gif_path: Path to GIF file
            display_size: Target display size (square)
        """
        self.frames = []
        self.frame_durations = []
        self.display_size = display_size

        # Load GIF frames
        with Image.open(gif_path) as img:
            try:
                frame_idx = 0
                while True:
                    # Convert frame to RGBA
                    frame = img.convert("RGBA")

                    # Resize to display size
                    frame = frame.resize((display_size, display_size), Image.Resampling.LANCZOS)

                    # Convert to pygame surface
                    mode = frame.mode
                    size = frame.size
                    data = frame.tobytes()

                    pg_surface = pg.image.fromstring(data, size, mode)

                    self.frames.append(pg_surface)
                    self.frame_durations.append(img.info.get('duration', 100))

                    frame_idx += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

        self.frame_count = len(self.frames)
        self.current_frame = 0
        self.last_update = time.time()

        # Calculate base frame time (average)
        self.base_frame_time = sum(self.frame_durations) / len(self.frame_durations) / 1000.0

        print(f"  Loaded {self.frame_count} frames, base frame time: {self.base_frame_time*1000:.1f}ms")

    def get_frame(self, speed_multiplier=1.0, reverse=False):
        """
        Get current frame and advance based on speed multiplier.

        Args:
            speed_multiplier: Speed factor (0.5 = half speed, 2.0 = double speed)
            reverse: If True, play animation in reverse

        Returns:
            pygame.Surface: Current frame
        """
        now = time.time()
        elapsed = now - self.last_update

        # Calculate frame time with speed multiplier
        frame_time = self.base_frame_time / max(0.1, speed_multiplier)

        # Advance frame if enough time has passed
        if elapsed >= frame_time:
            if reverse:
                self.current_frame = (self.current_frame - 1) % self.frame_count
            else:
                self.current_frame = (self.current_frame + 1) % self.frame_count
            self.last_update = now

        return self.frames[self.current_frame]

    def jump_to_frame(self, frame_idx):
        """Jump to specific frame (for peak-triggered effects)"""
        self.current_frame = frame_idx % self.frame_count
        self.last_update = time.time()


class AudioReactiveGifFace:
    """
    Audio-reactive face using pre-rendered GIF animations.

    Audio Feature Mapping:
    - RMS (volume) → playback speed (0.5x - 3.0x)
    - Peak detection → momentary speedup or frame jump
    - ZCR (sibilance) → subtle speed variation
    """

    def __init__(self, size=CFG["size"], gif_dir=CFG["gif_dir"]):
        pg.init()
        self.size = size
        self.screen = pg.display.set_mode((size, size))
        pg.display.set_caption("HowdyVox — Audio-Reactive Face")
        self.clock = pg.time.Clock()

        # State management
        self.state = "idle"
        self.level = 0.0        # RMS volume (0.0-1.0)
        self.zcr = 0.0          # Zero-crossing rate (0.0-1.0)
        self.peak_frames = 0    # Peak animation counter

        # Load GIF animations for each state
        print("\nLoading GIF animations...")
        self.animations = {}
        for state, gif_file in GIF_MAP.items():
            gif_path = os.path.join(gif_dir, gif_file)
            if os.path.exists(gif_path):
                print(f"  {state}: {gif_file}")
                self.animations[state] = GifAnimation(gif_path, display_size=size)
            else:
                print(f"  {state}: WARNING - {gif_file} not found!")

        print(f"\nLoaded {len(self.animations)} animations")

    def _parse_msg(self, msg: str):
        """
        Parse UDP message for audio features.

        Formats:
        - "idle" / "listening" / "thinking" / "speaking"
        - "speaking:0.63;zcr=0.18;peak=1"
        """
        state = msg.strip().lower()
        level = None
        zcr = None
        peak = 0

        if ";" in msg or ":" in msg:
            parts = [p for p in msg.split(";") if p]
            head = parts[0]

            if ":" in head:
                s, v = head.split(":", 1)
                state = s.lower().strip()
                try:
                    level = float(v)
                except ValueError:
                    level = None
            else:
                state = head.lower().strip()

            for p in parts[1:]:
                if p.startswith("zcr="):
                    try:
                        zcr = float(p.split("=", 1)[1])
                    except ValueError:
                        pass
                elif p.startswith("peak="):
                    try:
                        peak = int(p.split("=", 1)[1])
                    except ValueError:
                        pass

        # Update state
        self.state = state
        if level is not None:
            self.level = max(0.0, min(1.0, level))
        if zcr is not None:
            self.zcr = max(0.0, min(1.0, zcr))
        if peak:
            self.peak_frames = 3  # Brief peak effect (3 frames)

    def calculate_speed_multiplier(self):
        """
        Calculate playback speed based on audio features.

        Returns:
            float: Speed multiplier (0.5 - 3.0)
        """
        if self.state == "speaking":
            # Base speed from volume (0.5x at silence, 2.5x at loud)
            base_speed = 0.5 + 2.0 * self.level

            # Add ZCR influence (sibilants speed up slightly)
            zcr_influence = 0.3 * self.zcr

            # Peak effect (temporary speedup)
            peak_influence = 1.0 if self.peak_frames > 0 else 0.0

            speed = base_speed + zcr_influence + peak_influence
            return max(0.5, min(3.0, speed))

        elif self.state == "thinking":
            # Gentle pulsing for thinking state
            pulse = 0.8 + 0.4 * abs(time.time() % 2 - 1)
            return pulse

        elif self.state == "listening":
            # Moderate speed for listening glow
            return 1.2

        else:  # idle/waiting
            # Slow, calm animation
            return 0.8

    def draw(self):
        """Render current frame"""
        self.screen.fill((0, 0, 0))  # Black background

        # Get current animation
        anim = self.animations.get(self.state)
        if not anim:
            # Fallback to idle if state not found
            anim = self.animations.get("idle")

        if anim:
            # Calculate speed multiplier from audio
            speed = self.calculate_speed_multiplier()

            # Get current frame
            frame_surface = anim.get_frame(speed_multiplier=speed)

            # Blit to center
            x = (self.size - anim.display_size) // 2
            y = (self.size - anim.display_size) // 2
            self.screen.blit(frame_surface, (x, y))

        # Decrement peak counter
        if self.peak_frames > 0:
            self.peak_frames -= 1

    def run(self, q=None):
        """Main render loop"""
        running = True
        demo_t0 = time.time()

        while running:
            # Handle pygame events
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False

            # Process UDP messages
            if q:
                try:
                    while True:
                        self._parse_msg(q.get_nowait())
                except queue.Empty:
                    pass
            else:
                # Self-demo mode: cycle through states
                t = (time.time() - demo_t0) % 20.0
                if t < 4.0:
                    self._parse_msg("idle")
                elif t < 7.0:
                    self._parse_msg("listening")
                elif t < 10.0:
                    self._parse_msg("thinking")
                else:
                    # Speaking with simulated audio
                    lvl = 0.3 + 0.7 * abs((t - 10) / 5.0 - 1.0)
                    zcr = 0.1 + 0.4 * abs((t - 10) / 3.0 - 1.0)
                    peak = 1 if int(t * 2) % 7 == 0 else 0
                    self._parse_msg(f"speaking:{lvl:.3f};zcr={zcr:.3f};peak={peak}")

            # Draw frame
            self.draw()
            pg.display.flip()

            # Maintain frame rate
            self.clock.tick(CFG["fps"])

        pg.quit()


def main():
    """Run audio-reactive GIF face with UDP control"""
    print("🎨 HowdyVox Audio-Reactive GIF Face")
    print("=" * 60)
    print(f"Listening on UDP port {CFG['udp_port']}")
    print("States: idle, listening, thinking, speaking")
    print("Format: speaking:0.63;zcr=0.18;peak=1")
    print("=" * 60)

    # Check if GIF directory exists
    if not os.path.exists(CFG["gif_dir"]):
        print(f"\nERROR: GIF directory '{CFG['gif_dir']}' not found!")
        print("Please ensure your GIF files are in the faceStates/ directory")
        return

    q = queue.Queue(maxsize=64)
    udp = UdpEvents(CFG["udp_port"], q)
    udp.start()

    try:
        # Pass q=None to see self-demo, q=q for UDP control
        AudioReactiveGifFace().run(q=q)
    finally:
        udp.stop()


if __name__ == "__main__":
    main()
