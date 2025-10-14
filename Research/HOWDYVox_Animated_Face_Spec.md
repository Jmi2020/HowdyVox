# HowdyVox — Animated 8‑Bit Face Microservice (Low‑Overhead, Linux/RPi Ready)

|Expert(s)|Embedded Systems Engineer; Real‑Time Graphics Engineer; Python Performance Engineer; Linux Systems Engineer; UX for Voice/Conversational UI|
|:--|:--|
|Question|Design and implement an ultra‑lightweight, animated 8‑bit face that reacts to HowdyVox TTS activity, optimized for Raspberry Pi/Linux with minimal CPU/RAM overhead; deliver libraries, repos, and a development route suitable for a coding agent.|
|Plan|Treat the face as a tiny **event‑driven renderer** with a minimal draw loop and no audio DSP. Recommend **out‑of‑process microservice** on the Pi (or locally on macOS during dev) receiving “start/stop speaking” (and optional “idle/listening/thinking”) events from Howdy. Implement two pluggable render backends: (A) **Pygame/SDL2** window (works everywhere), (B) **direct-to-small-display** (SSD1306 OLED or ST77xx TFT) using PIL/Luma or ST7789 drivers. Provide two transports: **UDP** (ultra‑low overhead) and **HTTP (FastAPI)** (dev‑friendly). Ship reference code, packaging, and a systemd unit for the Pi.|

> ⏯️ **This document**: architecture, library options, repositories, code skeletons (UDP + Pygame), Howdy integration hooks, packaging/deployment to Raspberry Pi, performance guidelines, and a step‑by‑step development route.

---

## 🧭 Overview & Goals

You want **maximum responsiveness at minimum cost**. The winning pattern is:

- **State‑driven animation**, not per‑sample audio analysis.  
- **Tiny surfaces, low FPS (10–15)**, precomputed frames, simple shapes.  
- **Decoupled service**: the face runs in its own process (or host), receiving events from Howdy (e.g., `speaking_on`, `speaking_off`), so it can live on a Raspberry Pi while the LLM/TTS runs on your Mac Studio.

---

## 🧱 Architecture (lowest overhead first)

1) **Face Microservice (Renderer)**  
   Runs on the Pi (or locally during dev). Minimal main loop + non‑blocking event queue. Draws an 8‑bit face with **speaking** mouth cycle, **idle** blink, optional **listening**/**thinking** animations.

2) **Transport**
- **Option A (recommended for prod): UDP datagrams** (no connections, tiny overhead).  
  📡 [UDP sockets (Python stdlib)](https://www.google.com/search?q=python+udp+socket+example+nonblocking)
- **Option B (for DX): HTTP + FastAPI** (human‑readable, easy to script).  
  🌐 [FastAPI quickstart](https://www.google.com/search?q=fastapi+uvicorn+quickstart)

3) **Render Backends**
- **Backend‑A: Pygame (SDL2)** → cross‑platform window; can target X/Wayland or KMSDRM/fb on Pi.  
  🎮 [Pygame 2 / SDL2 performance](https://www.google.com/search?q=pygame+2+SDL2+Raspberry+Pi+performance)
- **Backend‑B: Small displays** (when Pi drives an OLED/TFT):  
  🖥️ [luma.oled (SSD1306/SH1106)](https://www.google.com/search?q=luma.oled+python+github) ·
  📱 [ST7789 Python driver](https://www.google.com/search?q=ST7789+python+raspberry+pi+github)

Start with **Pygame** for speed of iteration; swap to OLED/TFT by drawing to Pillow and pushing to device.

---

## 📦 Libraries & Repositories (shortlist)

- Core rendering:  
  🎮 [Pygame install & docs](https://www.google.com/search?q=install+pygame+2+pip+linux+macOS) ·
  🖼️ [Pillow docs](https://www.google.com/search?q=pillow+python+documentation)

- Pi small displays (optional backends):  
  🖥️ [luma.oled GitHub](https://www.google.com/search?q=luma.oled+github+raspberry+pi) ·
  📱 [ST7789 python driver](https://www.google.com/search?q=python+ST7789+raspberry+pi+driver)

- Messaging transports:  
  📡 [Python UDP socket example](https://www.google.com/search?q=python+udp+socket+example) ·
  🌐 [FastAPI quickstart](https://www.google.com/search?q=fastapi+uvicorn+quickstart) ·
  🔌 [paho‑mqtt client](https://www.google.com/search?q=paho+mqtt+python+client) (optional)

- Service on Pi:  
  ⚙️ [systemd service unit tutorial](https://www.google.com/search?q=systemd+create+service+unit)

---

## 🧠 Face State Machine

- `idle` → eyes blink every 3–6 s (randomized); mouth closed.  
- `listening` (optional) → subtle eye widen or cheek pulse.  
- `thinking` (optional) → slow head bob or “…” bubble.  
- `speaking` → cycle mouth frames `[closed → half → open → half → closed]` at 6–10 Hz.

Keep **FPS ~12** while speaking; **5–8 FPS** idle. Resolution **≤ 256×256** with nearest‑neighbor scale for chunky 8‑bit look.

---

# Minimal Anime-Style Face for Offline Voice Assistant  
**Goal:** Create a lightweight, cute, anime-inspired facial UI for a Raspberry Pi-based offline voice assistant. The face should display only **eyes and a mouth** on a **black background**, suitable for real-time rendering with minimal CPU/GPU load.

---

## 🧩 Design Overview

**Core principles:**
- Minimal and expressive: only eyes + mouth
- Lightweight for Raspberry Pi hardware (PyGame or LVGL)
- Black background, soft colors
- Supports simple emotional states and animation

**Visual style:**
- Background: `#000000`
- Eye whites: `#FFFFFF`
- Pupils: `#222222`
- Mouth (soft anime pink): `#FFB6C1` or `#FF9999`

---

## 🎨 Face Concepts (ASCII Sketches)

| Expression | Example | Description |
|:--|:--|:--|
| **Idle / Happy** | `( ^‿^)` | Default neutral smile |
| **Blinking Idle** | `( -‿-)` → `( ^‿^)` | Alternating blink animation |
| **Listening** | `( •_•)` | Focused, neutral |
| **Speaking** | `( ᵔ◡ᵔ )` | Open smile or animated mouth |
| **Surprised** | `( °o° )` | Large eyes and open mouth |

---

## 💻 Implementation Plan

### 1. Environment Setup
- Install Python 3 and `pygame`:
  ```bash
  sudo apt update
  sudo apt install python3-pip
  pip3 install pygame

---

## 🧪 Code Skeletons (ready to drop into a repo)

### Project layout

```
howdy_face/
  pyproject.toml
  howdy_face/
    __init__.py
    config.py
    states.py
    backend_pygame.py
    backend_oled.py          # optional (luma.oled)
    server_udp.py
    server_http.py           # optional (FastAPI)
    run_face.py              # entry point
  client/
    face_client_udp.py
    face_client_http.py
  scripts/
    install_pi.sh
    howdy-face.service       # systemd unit
```

### `pyproject.toml`

```toml
[project]
name = "howdy-face"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
  "pygame>=2.5",         # core
  "pillow>=10.0",        # if generating pixel art / PNGs
  # "fastapi>=0.111", "uvicorn[standard]>=0.30",   # optional HTTP
  # "luma.oled>=3.13.0", "luma.core>=2.4.0",       # optional OLED backend
  # "paho-mqtt>=2.1.0",                            # optional MQTT
]

[project.scripts]
howdy-face = "howdy_face.run_face:main"
```

### UDP server (ultra‑light) + Pygame backend

`howdy_face/server_udp.py`
```python
import socket, threading, queue
from typing import Optional

class UdpEventServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 31337, q: Optional[queue.Queue] = None):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        self.q = q or queue.Queue(maxsize=64)
        self._stop = threading.Event()

    def start(self):
        t = threading.Thread(target=self._rx, daemon=True)
        t.start()
        return t

    def _rx(self):
        self.sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                data, _peer = self.sock.recvfrom(128)
                msg = data.decode("utf-8", "ignore").strip().lower()
                # expected: speaking_on, speaking_off, idle, listening, thinking, shutdown
                if msg:
                    try:
                        self.q.put_nowait(msg)
                    except queue.Full:
                        pass
            except socket.timeout:
                continue

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
```

`howdy_face/backend_pygame.py`
```python
import time, random, queue
import pygame as pg

class FaceRenderer:
    def __init__(self, q: queue.Queue, size=160, title="Howdy Face"):
        self.q = q
        self.size = int(size)
        self.state = "idle"
        self.mouth_phase = 0
        self.last_blink = time.time()
        self.next_blink_in = random.uniform(3.0, 6.0)
        self.clock = pg.time.Clock()

        pg.init()
        self.screen = pg.display.set_mode((self.size, self.size))
        pg.display.set_caption(title)

    def _handle_events(self):
        # pygame window events
        for evt in pg.event.get():
            if evt.type == pg.QUIT:
                return False
        # udp control events
        while True:
            try:
                msg = self.q.get_nowait()
            except queue.Empty:
                break
            m = msg.strip().lower()
            if m in ("speaking_on", "speaking"):
                self.state = "speaking"
            elif m in ("speaking_off", "idle"):
                self.state = "idle"
            elif m in ("listening", "thinking"):
                self.state = m
            elif m == "shutdown":
                return False
        return True

    def _maybe_blink(self):
        now = time.time()
        if self.state != "speaking" and (now - self.last_blink) > self.next_blink_in:
            self.last_blink = now
            self.next_blink_in = random.uniform(3.0, 6.0)
            return True
        return False

    def _draw_face(self, blink=False):
        S = self.size
        scr = self.screen
        scr.fill((16,16,16))                  # dark bg

        # 8-bit head (big pixel rectangle with border)
        margin = S//16
        pg.draw.rect(scr, (246,222,180), (margin, margin, S-2*margin, S-2*margin))
        pg.draw.rect(scr, (0,0,0), (margin, margin, S-2*margin, S-2*margin), width=2)

        # eyes
        eye_w = S//12; eye_h = S//12
        eye_y = S//3
        eye_x1 = S//3 - eye_w//2
        eye_x2 = 2*S//3 - eye_w//2

        if blink:
            # thin line for blink
            pg.draw.rect(scr, (0,0,0), (eye_x1, eye_y, eye_w, 2))
            pg.draw.rect(scr, (0,0,0), (eye_x2, eye_y, eye_w, 2))
        else:
            pg.draw.rect(scr, (0,0,0), (eye_x1, eye_y, eye_w, eye_h))
            pg.draw.rect(scr, (0,0,0), (eye_x2, eye_y, eye_w, eye_h))

        # mouth
        mouth_y = int(S*0.65); mouth_w = S//3; mouth_h = S//14
        mouth_x = S//2 - mouth_w//2

        if self.state == "speaking":
            self.mouth_phase = (self.mouth_phase + 1) % 4
            phases = {0: mouth_h//6, 1: mouth_h//2, 2: mouth_h, 3: mouth_h//2}
            h = max(2, phases[self.mouth_phase])
        elif self.state == "thinking":
            # simple bouncing dot above head
            dot_y = int(mouth_y - S*0.25 + 4 * (1 + time.time() % 1))
            pg.draw.rect(scr, (0,0,0), (S//2-2, dot_y, 4, 4))
            h = 2
        else:
            h = 2  # closed

        pg.draw.rect(scr, (0,0,0), (mouth_x, mouth_y, mouth_w, h))

    def run(self):
        running = True
        blink_active = False
        blink_until = 0.0

        while running:
            running = self._handle_events()

            # blink timing (~120ms)
            now = time.time()
            if blink_active and now >= blink_until:
                blink_active = False
            if not blink_active and self._maybe_blink():
                blink_active = True
                blink_until = now + 0.12

            self._draw_face(blink=blink_active)
            pg.display.flip()

            # cap FPS by state
            fps = 12 if self.state == "speaking" else 7
            self.clock.tick(fps)

        pg.quit()
```

`howdy_face/run_face.py`
```python
import queue
from .server_udp import UdpEventServer
from .backend_pygame import FaceRenderer

def main():
    q = queue.Queue(maxsize=128)
    server = UdpEventServer(port=31337, q=q)
    server.start()
    FaceRenderer(q=q, size=160, title="Howdy Face").run()
    server.stop()
```

### Minimal UDP client (use from Howdy)

`client/face_client_udp.py`
```python
import socket

def send_state(state: str, host="127.0.0.1", port=31337):
    msg = state.strip().encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(msg, (host, port))

if __name__ == "__main__":
    send_state("speaking_on")
```

### HTTP (optional, dev‑friendly)

- 🌐 Endpoints: `POST /state { "state": "speaking" }`.  
  See: [FastAPI tutorial](https://www.google.com/search?q=fastapi+first+steps)

> If you prefer HTTP, create `server_http.py` with FastAPI and feed an `asyncio.Queue` to the same `FaceRenderer`. Overhead is slightly higher vs UDP; useful during integration and testing.

---

## 🔌 Integrate with Howdy (STT→LLM→TTS)

Hook into your TTS playback boundary (before/after audio out).

```python
# howdy/integration/face_hooks.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "client"))
from face_client_udp import send_state

def on_tts_start():
    send_state("speaking_on", host="pi.local")  # or 127.0.0.1 during dev

def on_tts_end():
    send_state("speaking_off", host="pi.local")
```

Wire these into your existing TTS call site:

```python
def speak_text(text: str):
    on_tts_start()
    try:
        # ... synthesize & play audio (blocking) ...
        # e.g., simpleaudio/playbuffer(...) or your current player
        pass
    finally:
        on_tts_end()
```

Optional: also send `listening` when mic is hot; `thinking` while waiting for LLM.

---

## 🖥️ Raspberry Pi specifics

**Install script (`scripts/install_pi.sh`)**
```bash
#!/usr/bin/env bash
set -e
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libsdl2-dev
python3 -m pip install --upgrade pip
python3 -m pip install "pygame>=2.5" "pillow>=10.0"
# Optional backends:
# python3 -m pip install "luma.oled" "luma.core"
# python3 -m pip install "fastapi" "uvicorn[standard]"
```

**Run without X (optional)**  
Pygame/SDL2 can target the console on modern Pi OS:
- Try KMS/DRM: `export SDL_VIDEODRIVER=kmsdrm`  
- Or use X/Wayland during dev.  
🔧 [SDL2 KMSDRM on Raspberry Pi](https://www.google.com/search?q=SDL2+KMSDRM+Raspberry+Pi+pygame)

**systemd unit (`scripts/howdy-face.service`)**
```ini
[Unit]
Description=Howdy Face (UDP renderer)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Environment=PYTHONUNBUFFERED=1
# Uncomment if running headless on KMSDRM
# Environment=SDL_VIDEODRIVER=kmsdrm
ExecStart=/usr/bin/python3 -m howdy_face
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo cp scripts/howdy-face.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now howdy-face.service
```

---

## 🧰 Optional: Small Display Backend

If you attach an OLED/TFT:

- Draw to a **Pillow Image** and blit that image to the display each frame.  
  🖥️ [luma.oled usage](https://www.google.com/search?q=luma.oled+examples) ·
  📱 [ST7789 python examples](https://www.google.com/search?q=st7789+python+examples+raspberry+pi)

Skeleton (replace `backend_pygame.py` drawing calls):

```python
# backend_oled.py
from PIL import Image, ImageDraw
import time, random, queue

class OledFaceRenderer:
    def __init__(self, device, q: queue.Queue, size=128):
        self.device = device         # luma device
        self.q = q
        self.size = size
        self.state = "idle"
        self._blink_t = time.time() + random.uniform(3,6)

    def run(self):
        while True:
            # ... read queue, update state (as in pygame example) ...
            img = Image.new("1", (self.size, self.size), 0)
            d = ImageDraw.Draw(img)
            # draw head/eyes/mouth in 1-bit
            # ...
            self.device.display(img)
            time.sleep(0.08)  # ~12.5 FPS speaking, adjust per state
```

---

## 📏 Performance Guidelines

- **Window size:** 128–256 px square; integer scale for pixel‑art.  
- **FPS:** 12 when speaking; 5–8 idle.  
- **Avoid** per‑frame allocations; reuse surfaces/Images.  
- **No alpha‑blending gradients**; stick to flat shapes or tiny sprites.  
- **Isolate I/O**: UDP listener thread → `Queue` → render loop; no locks in hot path.  
- **Monitor** with `htop`; target **<5–10% CPU** on Pi 4 while speaking (often far less).

---

## 🧪 Test Plan (dev→Pi)

1. **Local (macOS)**  
   - `pip install -e .`  
   - `howdy-face` (window opens).  
   - `python client/face_client_udp.py` with `speaking_on/speaking_off`.  
   - Verify mouth cycles while “speaking.”

2. **Pi (X/Wayland)**  
   - Run `scripts/install_pi.sh`.  
   - `howdy-face` → confirm window runs.  
   - Send states via UDP from laptop:  
     ```bash
     python -c 'import socket;s=socket.socket(2,2);s.sendto(b"speaking_on",("pi.local",31337))'
     ```

3. **Pi (headless, optional)**  
   - Export `SDL_VIDEODRIVER=kmsdrm` and run.  
   - Or switch backend to OLED/TFT and verify on hardware.

4. **Howdy integration**  
   - Call `on_tts_start()` / `on_tts_end()` around audio playback.  
   - Add optional `listening` when mic opens; `thinking` while LLM runs.

5. **Service**  
   - Install systemd unit; reboot; ensure face comes up before Howdy signals.

---

## 🛣️ Development Route (for a coding agent)

**Milestone 1 — Pygame UDP face (desktop)**  
- [ ] Scaffold repo; add `pyproject.toml`.  
- [ ] Implement `UdpEventServer`, `FaceRenderer`, `run_face.py`.  
- [ ] Add `client/face_client_udp.py`.  
- [ ] **Acceptance:** local manual toggles animate; CPU < ~2–3% on laptop when speaking.

**Milestone 2 — Howdy integration (local)**  
- [ ] Add `on_tts_start/on_tts_end` hooks; UDP to `127.0.0.1`.  
- [ ] **Acceptance:** face animates during actual TTS playback.

**Milestone 3 — Raspberry Pi target**  
- [ ] Ship to Pi; run under X/Wayland first.  
- [ ] Add `scripts/install_pi.sh` and systemd unit; test autostart.  
- [ ] **Acceptance:** receives states over LAN (`pi.local`), smooth animation, CPU < ~5–10% speaking.

**Milestone 4 — Optional small display backend**  
- [ ] Implement `backend_oled.py` or ST7789 backend and config switch.  
- [ ] **Acceptance:** identical behavior on OLED/TFT at ~12 FPS speaking.

**Milestone 5 — Developer‑friendly transport (optional)**  
- [ ] Add `server_http.py` (FastAPI) + `client/face_client_http.py`.  
- [ ] **Acceptance:** identical behavior via HTTP; note higher overhead.

**Milestone 6 — Polish**  
- [ ] Config (YAML/ENV): FPS per state, size, palette, UDP port, host allowlist.  
- [ ] Graceful shutdown (`shutdown` message).  
- [ ] Logging at INFO (suppress in hot loop).  
- [ ] README with run/dev instructions and troubleshooting.

---

## 🧷 Troubleshooting

- **Black screen on Pi headless:** try under X first; then `SDL_VIDEODRIVER=kmsdrm`.  
- **Stutter:** lower FPS; reduce window to 128×128; avoid per‑frame allocations.  
- **No events received:** verify UDP port; try `nc -u pi.local 31337` and send `speaking_on`.  
- **High CPU:** confirm no alpha blending; keep shapes simple; reuse surfaces; reduce FPS.

---

### See also
- 🎮 [Pygame 2 (SDL2) on Pi](https://www.google.com/search?q=pygame+2+raspberry+pi+performance) — renderer foundation  
- 📡 [Python UDP sockets](https://www.google.com/search?q=python+udp+socket+example) — ultra‑light transport  
- 🖼️ [Pillow pixel art scaling](https://www.google.com/search?q=pillow+pixel+art+nearest+neighbor+python) — crisp 8‑bit look  
- 🖥️ [luma.oled docs & examples](https://www.google.com/search?q=luma.oled+examples) — tiny OLED backend  
- 📱 [ST7789 Python driver](https://www.google.com/search?q=st7789+python+examples+raspberry+pi) — tiny TFT backend  
- 🌐 [FastAPI quickstart](https://www.google.com/search?q=fastapi+uvicorn+quickstart) — dev‑friendly HTTP alternative  
- ⚙️ [systemd service how‑to](https://www.google.com/search?q=systemd+create+service+unit) — run at boot

### You may also enjoy
- 🧱 [Pixel art face inspiration](https://www.google.com/search?q=pixel+art+faces+8+bit) — expression ideas  
- 🧩 [Nearest‑neighbor scaling](https://www.google.com/search?q=nearest+neighbor+scaling+pixel+art) — preserve “chunky” pixels  
- 🧰 [Raspberry Pi performance tuning](https://www.google.com/search?q=raspberry+pi+performance+tweaks) — free up headroom  
- 🛰️ [mDNS / Avahi on Pi](https://www.google.com/search?q=raspberry+pi+avahi+mdns+setup) — use `pi.local` without IPs
