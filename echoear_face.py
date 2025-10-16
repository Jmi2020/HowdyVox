#!/usr/bin/env python3
"""
EchoEar-style face renderer for HowdyVox
Audio-reactive face with volume, sibilance, and emphasis detection
Based on the EchoEar minimalist aesthetic
"""

import math
import random
import socket
import threading
import queue
import time
import pygame as pg

# Set process name for macOS dock/Activity Monitor
try:
    import setproctitle
    setproctitle.setproctitle("HowdyVox")
except ImportError:
    pass  # setproctitle not available, will show as python3.10

# Configuration
CFG = {
    "size": 200,                 # Window size (200x200 for better visibility)
    "bg": (0, 0, 0),            # Black background
    "eye_cyan": (0, 235, 255),  # Cyan eyes (HowdyVox theme)
    "ring": (40, 40, 40),       # Outer ring color
    "fps_idle": 6,              # FPS when idle
    "fps_speaking": 12,         # FPS when speaking
    "udp_port": 31337,          # UDP port for state updates
    "head_nod_px": 4,           # Downshift in pixels for head nod
}


class UdpEvents:
    """Ultra-light UDP listener placing decoded messages on a queue."""

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


class EchoEarFace:
    """
    EchoEar-style animated face.

    Features:
    - Circular stage with cyan glowing eyes
    - Audio-reactive scaling (RMS → eye size)
    - Sibilance detection (ZCR → horizontal squeeze)
    - Emphasis detection (peak → head nod)
    - Random blinking when not speaking
    """

    def __init__(self, size=CFG["size"], color=CFG["eye_cyan"]):
        pg.init()
        self.S = int(size)
        self.screen = pg.display.set_mode((self.S, self.S))
        pg.display.set_caption("HowdyVox — EchoEar Face")

        # Set custom window icon (rounded version)
        try:
            import os
            icon_path = os.path.join(os.path.dirname(__file__), "assets", "glowface_rounded.png")
            if os.path.exists(icon_path):
                icon = pg.image.load(icon_path)
                pg.display.set_icon(icon)
            else:
                # Fallback to non-rounded version
                icon_path = os.path.join(os.path.dirname(__file__), "assets", "glowface.png")
                if os.path.exists(icon_path):
                    icon = pg.image.load(icon_path)
                    pg.display.set_icon(icon)
        except Exception as e:
            print(f"Could not load window icon: {e}")

        self.clock = pg.time.Clock()

        # State & feature controls
        self.state = "idle"
        self.level = 0.0      # 0..1 (RMS volume)
        self.zcr = 0.0        # 0..1 (sibilance proxy)
        self._blink = False
        self._next_blink = time.time() + random.uniform(3.0, 6.0)
        self._blink_end = 0.0
        self._nod_frames = 0

        # Precompute eye surfaces (base + glow)
        self.eye_base, self.eye_glow = self._make_eye_surfaces(color)

    def _make_eye_surfaces(self, color):
        """Precompute eye surfaces with glow effect"""
        eye_w = self.S // 9
        eye_h = self.S // 9
        pad = self.S // 18  # glow padding

        W = eye_w + pad * 2
        H = eye_h + pad * 2
        base = pg.Surface((W, H), pg.SRCALPHA)
        glow = pg.Surface((W * 2, H * 2), pg.SRCALPHA)

        # Base rounded-rect (cyan eye)
        rect = pg.Rect(pad, pad, eye_w, eye_h)
        pg.draw.rect(base, color, rect, border_radius=eye_w // 3)

        # Glow effect: multiple expanding rects with decaying alpha
        layers = 12
        for i in range(layers):
            alpha = int(100 * (1.0 - i / layers) ** 2)
            kx = int(i * 1.8) + pad // 2
            ky = int(i * 1.8) + pad // 2
            r = eye_w + 2 * kx
            h = eye_h + 2 * ky
            rr = max(2, r // 3)

            tmp = pg.Surface((r, h), pg.SRCALPHA)
            pg.draw.rect(tmp, (*color, alpha), pg.Rect(0, 0, r, h), border_radius=rr)

            gx = glow.get_width() // 2 - r // 2
            gy = glow.get_height() // 2 - h // 2
            glow.blit(tmp, (gx, gy), special_flags=pg.BLEND_PREMULTIPLIED)

        return base, glow

    def _maybe_blink(self):
        """Handle random blinking (only when not speaking)"""
        now = time.time()
        if not self._blink and now >= self._next_blink and self.state != "speaking":
            self._blink = True
            self._blink_end = now + 0.12  # 120ms blink
        if self._blink and now >= self._blink_end:
            self._blink = False
            self._next_blink = now + random.uniform(3.0, 6.0)

    def _parse_msg(self, msg: str):
        """
        Parse UDP message.

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
            self._nod_frames = 2  # Brief nod (2 frames)

    def draw(self):
        """Draw the face"""
        S = self.S
        scr = self.screen
        scr.fill(CFG["bg"])

        # Circular stage with head nod offset
        nod_offset = CFG["head_nod_px"] if self._nod_frames > 0 else 0
        center = (S // 2, S // 2 + nod_offset)
        radius = S // 2 - 3

        # Draw stage circle
        pg.draw.circle(scr, (0, 0, 0), center, radius)
        pg.draw.circle(scr, CFG["ring"], center, radius, 2)

        # Eye placement relative to center
        eye_gap = S // 7
        eye_y = center[1] - self.eye_base.get_height() // 2
        left_x = S // 2 - eye_gap - self.eye_base.get_width() // 2
        right_x = S // 2 + eye_gap - self.eye_base.get_width() // 2

        # Calculate scaling based on state and audio features
        if self.state == "speaking":
            # Volume drives eye size (0.2 → 1.0 = 20% → 100% louder)
            scale_base = 0.85 + 0.4 * self.level  # 0.85..1.25
        elif self.state == "thinking":
            # Gentle pulse for thinking
            pulse = 0.5 + 0.5 * abs(math.sin(time.time() * 2))
            scale_base = 0.95 + 0.1 * pulse
        elif self.state == "listening":
            scale_base = 1.05  # Slightly larger when listening
        else:
            scale_base = 1.0  # Idle

        # ZCR → horizontal squeeze (sibilants make eyes narrower)
        # High ZCR = more zero crossings = "s", "sh" sounds
        squeeze = 1.0 - 0.3 * min(1.0, self.zcr * 1.5)  # 0.70..1.0
        sx = max(0.70, scale_base * squeeze)
        sy = min(1.35, scale_base * (1.0 + (1.0 - squeeze)))  # Compensate height

        def blit_eye(x, y):
            """Draw a single eye with scaling"""
            if self._blink and self.state != "speaking":
                # Thin cyan blink line
                w = self.eye_base.get_width()
                pg.draw.rect(scr, CFG["eye_cyan"],
                           (x, y + self.eye_base.get_height() // 2, w, 2))
                return

            # Scale eye surfaces
            eb = pg.transform.scale(
                self.eye_base,
                (int(self.eye_base.get_width() * sx),
                 int(self.eye_base.get_height() * sy))
            )
            gl = pg.transform.scale(
                self.eye_glow,
                (int(self.eye_glow.get_width() * sx),
                 int(self.eye_glow.get_height() * sy))
            )

            # Draw glow
            gx = x + self.eye_base.get_width() // 2 - gl.get_width() // 2
            gy = y + self.eye_base.get_height() // 2 - gl.get_height() // 2
            scr.blit(gl, (gx, gy), special_flags=pg.BLEND_PREMULTIPLIED)

            # Draw eye base
            bx = x + self.eye_base.get_width() // 2 - eb.get_width() // 2
            by = y + self.eye_base.get_height() // 2 - eb.get_height() // 2
            scr.blit(eb, (bx, by))

        # Draw both eyes
        blit_eye(left_x, eye_y)
        blit_eye(right_x, eye_y)

        # Draw thinking indicator if in thinking state
        if self.state == "thinking":
            dot_size = 4
            dot_spacing = 10
            dot_y = center[1] - S // 4
            num_dots = int((time.time() * 2) % 4)  # 0-3 dots cycling

            for i in range(num_dots):
                dot_x = S // 2 - dot_spacing + i * dot_spacing
                pg.draw.circle(scr, CFG["eye_cyan"], (dot_x, dot_y), dot_size)

        # Decrement nod counter
        if self._nod_frames > 0:
            self._nod_frames -= 1

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
                # Self-demo mode: cycle through states with simulated audio
                t = (time.time() - demo_t0) % 12.0
                if t < 3.0:
                    self._parse_msg("idle")
                elif t < 5.0:
                    self._parse_msg("listening")
                elif t < 7.0:
                    self._parse_msg("thinking")
                else:
                    # Speaking with sinusoidal level and varying zcr
                    lvl = 0.3 + 0.7 * abs(math.sin(t * 3.14))
                    zcr = 0.1 + 0.6 * abs(math.sin(t * 5.2))
                    peak = 1 if int(t * 4) % 7 == 0 else 0
                    self._parse_msg(f"speaking:{lvl:.3f};zcr={zcr:.3f};peak={peak}")

            # Update blink state
            self._maybe_blink()

            # Draw frame
            self.draw()
            pg.display.flip()

            # Frame rate based on state
            fps = CFG["fps_speaking"] if self.state == "speaking" else CFG["fps_idle"]
            self.clock.tick(fps)

        pg.quit()


def main():
    """Run the EchoEar face with UDP control"""
    print("🎨 HowdyVox EchoEar Face")
    print("=" * 50)
    print(f"Listening on UDP port {CFG['udp_port']}")
    print("States: idle, listening, thinking, speaking")
    print("Format: speaking:0.63;zcr=0.18;peak=1")
    print("=" * 50)

    q = queue.Queue(maxsize=64)
    udp = UdpEvents(CFG["udp_port"], q)
    udp.start()

    try:
        # Pass q=None to see self-demo, q=q for UDP control
        EchoEarFace().run(q=q)
    finally:
        udp.stop()


if __name__ == "__main__":
    main()
