
# HowdyVox — EchoEar‑Style Face **+** Reactive Audio Analyzer (Combined Integration Guide)

|Expert(s)|Audio DSP Engineer; Real‑Time Graphics Engineer; Embedded Systems Engineer; Python Performance Engineer|
|:--|:--|
|Question|Recreate EchoEar’s minimalist face animation style and make it more expressive by reacting to TTS audio in real time, with near‑zero overhead on Raspberry Pi/Linux (dev on macOS).|
|Plan|Ship two tiny components: (1) a **Pygame/SDL2 face renderer** that mimics EchoEar’s style (circular stage, cyan eyes, blinks, pulses) and (2) a **C‑fast audio analyzer** (using stdlib `audioop`) that computes light features (RMS envelope, ZCR, onset) **near the TTS** and sends **low‑rate UDP** control messages. Keep FPS small, surfaces tiny, and avoid per‑frame allocations. Provide end‑to‑end wiring, tests, and performance budgets.|

> ⏯️ This combined doc merges the previous two specs: the EchoEar‑style renderer and the lightweight waveform analysis/publisher. Paste it into your repo and hand it to a coding agent.

---

## 0) TL;DR

- **Renderer (Pi or macOS)**: small Pygame process draws an EchoEar‑style face (circular stage, cyan rounded‑square eyes, random idle blinks). It accepts UDP state messages, including **audio‑driven** accents.  
- **Analyzer (Mac, next to TTS)**: processes PCM in 20 ms hops with `audioop.rms` (C‑accelerated), **attack/decay** smoothing, **ZCR** (zero‑crossing rate), and a simple **onset** hint. Publishes **~12 msgs/s** to the renderer.  
- **Overhead**: Analyzer ≪ 1% CPU on Mac; Renderer ~**< 5–10%** CPU on Pi 4 at 160×160 px and 12 FPS speaking / 6 FPS idle.

---

## 1) Face Renderer — EchoEar‑Style (UDP‑Controlled, Ultra‑Light)

> **File:** `echoear_face.py` — run standalone for demo; or with UDP messages from the analyzer.  
> **Style:** circular stage, cyan rounded‑rect eyes, idle blink, speaking/listening pulses.  
> **Expressive extras:** uses `level` (RMS), `zcr` (sibilance proxy) and `peak` (onset) to animate without DSP on the Pi.

```python
# echoear_face.py — EchoEar-style face renderer (Pygame/SDL2)
# Run: python echoear_face.py
# UDP control messages (ASCII, newline not required):
#   "idle" | "listening" | "speaking"
#   Optional fields: "speaking:<level>;zcr=<0..1>;peak=<0|1>"
#     level -> eye scale pulse, zcr -> horizontal squeeze, peak -> brief head nod
#
# Example: speaking:0.63;zcr=0.18;peak=1

import math, random, socket, threading, queue, time
import pygame as pg

CFG = {
    "size": 160,                 # keep small; integer-scale externally if you like
    "bg":   (0, 0, 0),
    "eye_cyan": (0, 235, 255),   # cyan eyes
    "ring": (40, 40, 40),        # outer ring color
    "fps_idle": 6,
    "fps_speaking": 12,
    "udp_port": 31337,           # listen for state updates
    "head_nod_px": 3,            # downshift in pixels when peak is true (for 1–2 frames)
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
                    try: self.q.put_nowait(msg)
                    except queue.Full: pass
            except socket.timeout:
                pass
            except OSError:
                break

    def stop(self):
        self._stop.set()
        try: self.sock.close()
        except OSError: pass

class EchoEarFace:
    def __init__(self, size=CFG["size"], color=CFG["eye_cyan"]):
        pg.init()
        self.S = int(size)
        self.screen = pg.display.set_mode((self.S, self.S))
        pg.display.set_caption("HowdyVox — EchoEar-style Face")
        self.clock = pg.time.Clock()
        # state & feature controls
        self.state = "idle"
        self.level = 0.0          # 0..1
        self.zcr = 0.0            # 0..1 (sibilance proxy)
        self._blink = False
        self._next_blink = time.time() + random.uniform(3.0, 6.0)
        self._blink_end  = 0.0
        self._nod_frames = 0

        # Precompute eye surfaces (base + glow)
        self.eye_base, self.eye_glow = self._make_eye_surfaces(color)

    def _make_eye_surfaces(self, color):
        eye_w = self.S // 9
        eye_h = self.S // 9
        pad   = self.S // 18  # glow pad
        W = eye_w + pad * 2
        H = eye_h + pad * 2
        base = pg.Surface((W, H), pg.SRCALPHA)
        glow = pg.Surface((W*2, H*2), pg.SRCALPHA)  # larger for feathered glow

        # Base rounded-rect (EchoEar-like cyan eye)
        rect = pg.Rect(pad, pad, eye_w, eye_h)
        pg.draw.rect(base, color, rect, border_radius=eye_w//3)

        # Simple glow: multiple expanding rects with decaying alpha
        layers = 10
        for i in range(layers):
            a = int(80 * (1.0 - i / layers) ** 2)
            kx = int(i * 1.5) + pad//2
            ky = int(i * 1.5) + pad//2
            r  = eye_w + 2*kx
            h  = eye_h + 2*ky
            rr = max(2, r // 3)
            tmp = pg.Surface((r, h), pg.SRCALPHA)
            pg.draw.rect(tmp, (*color, a), pg.Rect(0, 0, r, h), border_radius=rr)
            gx = glow.get_width()//2  - r//2
            gy = glow.get_height()//2 - h//2
            glow.blit(tmp, (gx, gy), special_flags=pg.BLEND_PREMULTIPLIED)

        return base, glow

    def _maybe_blink(self):
        now = time.time()
        if not self._blink and now >= self._next_blink:
            self._blink = True
            self._blink_end = now + 0.12  # ~120 ms
        if self._blink and now >= self._blink_end:
            self._blink = False
            self._next_blink = now + random.uniform(3.0, 6.0)

    def _parse_msg(self, msg: str):
        # Accept: "state" or "state:level" + optional ";k=v" pairs
        # Example: speaking:0.63;zcr=0.18;peak=1
        state = msg.strip().lower()
        level = None; zcr = None; peak = 0
        if ";" in msg or ":" in msg:
            parts = [p for p in msg.split(";") if p]
            head = parts[0]
            if ":" in head:
                s, v = head.split(":", 1)
                state = s.lower().strip()
                try: level = float(v)
                except ValueError: level = None
            else:
                state = head.lower().strip()
            for p in parts[1:]:
                if p.startswith("zcr="):
                    try: zcr = float(p.split("=",1)[1])
                    except ValueError: pass
                elif p.startswith("peak="):
                    try: peak = int(p.split("=",1)[1])
                    except ValueError: pass
        # update
        self.state = state
        if level is not None: self.level = max(0.0, min(1.0, level))
        if zcr   is not None: self.zcr   = max(0.0, min(1.0, zcr))
        if peak: self._nod_frames = 2  # brief nod

    def draw(self):
        S = self.S; scr = self.screen
        scr.fill(CFG["bg"])

        # circular "stage"
        center = (S//2, S//2 + (CFG["head_nod_px"] if self._nod_frames>0 else 0))
        radius = S//2 - 2
        pg.draw.circle(scr, (0,0,0), center, radius)
        pg.draw.circle(scr, CFG["ring"], center, radius, 2)

        # eye placement relative to center
        eye_gap = S // 7
        # baseline eye top-left (before scaling)
        eye_y   = center[1] - self.eye_base.get_height()//2
        left_x  = S//2 - eye_gap - self.eye_base.get_width()//2
        right_x = S//2 + eye_gap - self.eye_base.get_width()//2

        # speaking/listening scale
        if self.state == "speaking":
            scale_base = 1.0 + 0.25 * self.level         # 1.00..1.25
        elif self.state == "listening":
            scale_base = 1.05
        else:
            scale_base = 1.0

        # ZCR -> horizontal squeeze (sibilants): more ZCR => narrow X, taller Y
        squeeze = 1.0 - 0.25 * min(1.0, self.zcr * 1.5)   # 0.75..1.0
        sx = max(0.75, scale_base * squeeze)
        sy = min(1.30, scale_base * (1.0 + (1.0 - squeeze)))  # widen vertically when X squeezes

        def blit_eye(x, y):
            if self._blink and self.state != "speaking":
                # thin cyan blink line
                w = self.eye_base.get_width()
                pg.draw.rect(scr, CFG["eye_cyan"], (x, y + self.eye_base.get_height()//2, w, 2))
                return

            eb = pg.transform.scale(self.eye_base, (int(self.eye_base.get_width()*sx),
                                                    int(self.eye_base.get_height()*sy)))
            gl = pg.transform.scale(self.eye_glow, (int(self.eye_glow.get_width()*sx),
                                                    int(self.eye_glow.get_height()*sy)))

            gx = x + self.eye_base.get_width()//2 - gl.get_width()//2
            gy = y + self.eye_base.get_height()//2 - gl.get_height()//2
            scr.blit(gl, (gx, gy), special_flags=pg.BLEND_PREMULTIPLIED)
            bx = x + self.eye_base.get_width()//2 - eb.get_width()//2
            by = y + self.eye_base.get_height()//2 - eb.get_height()//2
            scr.blit(eb, (bx, by))

        blit_eye(left_x,  eye_y)
        blit_eye(right_x, eye_y)

        if self._nod_frames > 0:
            self._nod_frames -= 1

    def run(self, q=None):
        running = True
        t0 = time.time()
        while running:
            for e in pg.event.get():
                if e.type == pg.QUIT: running = False

            # message pump
            if q:
                try:
                    while True:
                        self._parse_msg(q.get_nowait())
                except queue.Empty:
                    pass
            else:
                # self‑demo: 2s idle → 2s listening → 4s speaking with sinusoid level
                t = (time.time() - t0) % 8.0
                if t < 2.0:
                    self._parse_msg("idle")
                elif t < 4.0:
                    self._parse_msg("listening")
                else:
                    lvl = 0.5 + 0.5 * math.sin(t * 3.14)  # 0..1
                    self._parse_msg(f"speaking:{lvl:.3f};zcr=0.2;peak=0")

            self._maybe_blink()
            self.draw()
            pg.display.flip()

            fps = CFG["fps_speaking"] if self.state == "speaking" else CFG["fps_idle"]
            self.clock.tick(fps)

        pg.quit()

def main():
    q = queue.Queue(maxsize=64)
    udp = UdpEvents(CFG["udp_port"], q); udp.start()
    try:
        EchoEarFace().run(q=q)  # use q=None to see the built‑in demo
    finally:
        udp.stop()

if __name__ == "__main__":
    main()
```

**Quick local test**
```bash
python echoear_face.py    # window opens
# In another terminal:
python - <<'PY'
import socket, time, math
s = socket.socket(2,2)
host, port = "127.0.0.1", 31337
# speak for ~3s with varying level/zcr, then idle
t0=time.time()
while time.time()-t0<3.0:
    t=time.time()-t0
    lvl=0.2+0.8*abs(math.sin(t*2.2))
    zcr=0.1+0.5*abs(math.sin(t*3.3))
    peak = 1 if int(t*4)%7==0 else 0
    s.sendto(f"speaking:{lvl:.3f};zcr={zcr:.3f};peak={peak}".encode(), (host,port))
    time.sleep(1/12)
s.sendto(b"idle", (host,port))
PY
```

---

## 2) Reactive Audio Analyzer — Tiny DSP Near the TTS

> **File:** `tts_reactive_meter.py` — processes PCM16 next to your TTS playback (Mac), publishes **~12 Hz** control messages over UDP to the renderer (Pi or local). No NumPy; uses stdlib `audioop` (C‑fast).

**Features (cheap but expressive):**
- **RMS envelope** (20 ms windows) → `level` (0..1) with **attack/decay** smoothing.  
- **Zero‑Crossing Rate (ZCR)** → `zcr` (0..1) (sibilants/brightness proxy).  
- **Onset hint** → `peak=1` with 150–200 ms refractory (tiny head nod).  
- Optional **AGC** via a drifting crest so `level` stays meaningful across voices/volumes.

```python
# tts_reactive_meter.py — ultra-light meter for HowdyVox
# Call meter.process(pcm_chunk) for each PCM buffer you play, and meter.tick() ~12 Hz.

from __future__ import annotations
import audioop, socket, time
from array import array

class ReactiveMeter:
    def __init__(self, samplerate=24000, sample_width=2, channels=1,
                 udp_host="127.0.0.1", udp_port=31337,
                 win_ms=20, update_hz=12):
        self.sr = samplerate
        self.sw = sample_width         # bytes per sample (2 for 16-bit)
        self.ch = channels
        self.host = udp_host
        self.port = udp_port
        self.win = int(self.sr * win_ms / 1000)
        self.buf = bytearray()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # envelope + AGC
        self.env = 0.0
        self.noise_floor = 200.0       # tune as needed
        self.crest = 2000.0            # running max-ish
        self.last_send = 0.0
        self.send_interval = 1.0 / update_hz
        self.last_peak_t = 0.0         # ms

    def _mono16(self, data: bytes) -> bytes:
        if self.ch == 1:
            return data
        return audioop.tomono(data, self.sw, 0.5, 0.5)

    def _rms(self, mono16: bytes) -> float:
        return float(audioop.rms(mono16, self.sw))  # 0..32767

    def _zcr(self, mono16: bytes) -> float:
        a = array('h'); a.frombytes(mono16)
        n = len(a)
        if n < 2: return 0.0
        crossings = 0
        prev = a[0]
        for i in range(1, n):
            cur = a[i]
            if (prev < 0 and cur > 0) or (prev > 0 and cur < 0):
                crossings += 1
            prev = cur
        return min(1.0, crossings / (n - 1))

    def _smooth_env(self, x: float) -> float:
        # AGC crest tracking
        if x > self.crest: self.crest = 0.9*self.crest + 0.1*x
        else:              self.crest = 0.999*self.crest + 0.001*x
        ref = max(self.noise_floor, self.crest * 0.6)
        lvl = max(0.0, min(1.0, (x - self.noise_floor) / (ref - self.noise_floor)))
        # attack/decay (fast rise, slower fall)
        attack_a, decay_a = 0.35, 0.10
        if lvl > self.env: self.env = (1-attack_a)*self.env + attack_a*lvl
        else:              self.env = (1-decay_a )*self.env + decay_a*lvl
        return self.env

    def _peak(self, lvl: float) -> bool:
        now = time.time()*1000.0
        if lvl > 0.55 and (now - self.last_peak_t) > 180.0:
            self.last_peak_t = now
            return True
        return False

    def process(self, pcm_chunk: bytes):
        """Feed raw PCM bytes (16-bit signed), once per produced/played chunk."""
        self.buf.extend(pcm_chunk)
        bytes_per_frame = self.sw * self.ch
        need = self.win * bytes_per_frame
        while len(self.buf) >= need:
            frame = self.buf[:need]; del self.buf[:need]
            mono = self._mono16(frame)
            rms = self._rms(mono)
            env = self._smooth_env(rms)
            zcr = self._zcr(mono)
            self._last_env = env
            self._last_zcr = zcr
            self._last_peak = self._peak(env)

    def tick(self):
        """Call ~10-15 Hz to publish state to the face."""
        now = time.time()
        if now - self.last_send < self.send_interval:
            return
        self.last_send = now

        env = getattr(self, "_last_env", 0.0)
        zcr = getattr(self, "_last_zcr", 0.0)
        peak = 1 if getattr(self, "_last_peak", False) else 0

        state = "speaking" if env > 0.07 else "idle"
        msg = f"{state}:{env:.3f};zcr={zcr:.3f};peak={peak}"
        try:
            self.sock.sendto(msg.encode("utf-8"), (self.host, self.port))
        except OSError:
            pass
```

**How to wire into your TTS playback loop** (pseudo‑example):

```python
# around your existing TTS playback
from tts_reactive_meter import ReactiveMeter

meter = ReactiveMeter(samplerate=24000, sample_width=2, channels=1,
                      udp_host="pi.local", udp_port=31337,
                      win_ms=20, update_hz=12)

on_tts_start()  # optional: also send a 'listening'/'speaking' hint

for pcm_chunk in generate_pcm_chunks(text):   # your TTS PCM16 chunks
    audio_device_write(pcm_chunk)             # your audio playback
    meter.process(pcm_chunk)
    meter.tick()                              # sends throttled face updates

on_tts_end()    # optional: send 'idle'
```

---

## 3) Directory Layout & Handoff

```
howdy_face/
  echoear_face.py            # renderer (runs on Pi or dev Mac)
  tts_reactive_meter.py      # analyzer (runs near TTS on Mac)
  client_example.py          # optional: manual sender for quick tests
  README_FACE.md             # this document (rename as needed)
  scripts/
    install_pi.sh
    howdy-face.service       # systemd unit (optional, Pi)
```

**Pi install script (`scripts/install_pi.sh`)**
```bash
#!/usr/bin/env bash
set -e
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libsdl2-dev
python3 -m pip install --upgrade pip
python3 -m pip install "pygame>=2.5"
```

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
# Environment=SDL_VIDEODRIVER=kmsdrm   # uncomment for headless console
ExecStart=/usr/bin/python3 /home/pi/howdy_face/echoear_face.py
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

## 4) Animation Mapping (cheap but expressive)

- **level (RMS)** → eye size & glow intensity (pulse while speaking).  
- **zcr** → horizontal squeeze (sibilants) = “narrower eyes”, vertical compensation for shape.  
- **peak** → brief head nod (1–2 frames).  
- **state** → `idle` (random blinks @ 3–6 s), `listening` (gentle 1.05 scale), `speaking` (use features).

**Budget targets**
- Renderer: **≤ 160×160 px**, **12 FPS speaking / 6 FPS idle**.  
- Analyzer: **12 Hz** messages, packets **< 64 B**.  
- CPU on Pi 4 while speaking: **< 5–10%**; Analyzer on Mac: **≪ 1%**.

---

## 5) Tests

**Local (single machine)**
1. `python echoear_face.py` (window opens).  
2. Run the “Quick local test” snippet above; verify pulses, squeezes, and nods.  

**E2E (Mac → Pi)**
1. Launch `echoear_face.py` on Pi (X/Wayland or `SDL_VIDEODRIVER=kmsdrm`).  
2. On Mac, hook `tts_reactive_meter.py` into your TTS loop and set `udp_host="pi.local"`.  
3. Speak a few sentences; confirm face reacts to loudness (bigger), sibilants (narrower), and peaks (nod).

---

## 6) Troubleshooting

- **No reaction**: check UDP port (default 31337), host/IP, and firewall.  
- **High CPU on Pi**: reduce `size` to 128, drop FPS, keep per‑frame transforms minimal.  
- **Blink too frequent**: widen blink interval to 4–8 s; shorten blink duration to 100 ms.  
- **Analyzer too quiet/loud**: tweak `noise_floor` or scale crest factor in `_smooth_env`.  
- **Latency**: analyzer `update_hz=12` is plenty; raising it increases CPU and traffic with little gain.

---

## 7) Future Extras (still cheap)

- **Visemes** (if your TTS exposes phoneme timings) → higher fidelity mouth/eye shapes without DSP.  
- **Small OLED/TFT** on Pi via `luma.oled` / ST7789 by drawing to a PIL image (same states/features).  
- **Theme packs**: swap cyan for other palettes; pre‑render different “eye” sprites at init for zero runtime cost.

---

**End of combined handoff** — plug `tts_reactive_meter.py` beside your TTS, run `echoear_face.py` on Pi or Mac, and enjoy an EchoEar‑style face that visibly **tracks speech** with expressive but **ultra‑light** cues.
