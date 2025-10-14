#!/usr/bin/env python3
"""
Animated 8-bit face widget for HowdyVox UI
Implements state-driven animation with minimal overhead
"""

import tkinter as tk
import random
import time


class FaceAnimator:
    """
    Lightweight animated 8-bit face using tkinter Canvas.

    States:
    - idle: Eyes blink randomly every 3-6s, mouth closed
    - listening: Eyes slightly widened, mouth closed
    - thinking: Slow subtle animation, "..." indicator
    - speaking: Mouth cycles through open/closed frames at ~10 Hz
    """

    def __init__(self, canvas, size=160):
        """
        Initialize the face animator.

        Args:
            canvas: tkinter Canvas widget to draw on
            size: Size of the face in pixels (square)
        """
        self.canvas = canvas
        self.size = size
        self.state = 'idle'

        # Animation state
        self.mouth_phase = 0  # 0-3 for speaking animation
        self.blink_active = False
        self.last_blink = time.time()
        self.next_blink_in = random.uniform(3.0, 6.0)
        self.thinking_phase = 0  # For "..." animation

        # Drawing elements (will store canvas item IDs)
        self.elements = {}

        # Colors (8-bit palette)
        self.colors = {
            'bg': '#101010',           # Dark background
            'head': '#F6DEB4',         # Skin tone
            'outline': '#000000',      # Black outlines
            'eye': '#000000',          # Black eyes
            'mouth': '#000000',        # Black mouth
            'thinking': '#FFD700'      # Gold for thinking dots
        }

        # Animation control
        self.running = False
        self.animation_id = None

        # Initialize the face
        self._init_face()

    def _init_face(self):
        """Initialize the face drawing"""
        # Clear canvas
        self.canvas.delete('all')

        # Background
        self.canvas.config(bg=self.colors['bg'])

        # Draw initial face
        self._draw_face()

    def _draw_face(self):
        """Draw the face based on current state and animation frame"""
        # Clear previous frame
        self.canvas.delete('all')

        S = self.size
        margin = S // 16

        # Head (rounded rectangle for 8-bit look)
        head_size = S - 2 * margin
        self.canvas.create_rectangle(
            margin, margin, S - margin, S - margin,
            fill=self.colors['head'],
            outline=self.colors['outline'],
            width=2
        )

        # Eyes
        eye_w = S // 12
        eye_h = S // 12 if not self.blink_active else 2
        eye_y = S // 3
        eye_x1 = S // 3 - eye_w // 2
        eye_x2 = 2 * S // 3 - eye_w // 2

        # Left eye
        self.canvas.create_rectangle(
            eye_x1, eye_y, eye_x1 + eye_w, eye_y + eye_h,
            fill=self.colors['eye'],
            outline=''
        )

        # Right eye
        self.canvas.create_rectangle(
            eye_x2, eye_y, eye_x2 + eye_w, eye_y + eye_h,
            fill=self.colors['eye'],
            outline=''
        )

        # Mouth (changes based on state)
        mouth_y = int(S * 0.65)
        mouth_w = S // 3
        mouth_h = S // 14
        mouth_x = S // 2 - mouth_w // 2

        if self.state == 'speaking':
            # Animate mouth phases: closed → half → open → half → closed
            phases = {
                0: mouth_h // 6,   # Nearly closed
                1: mouth_h // 2,   # Half open
                2: mouth_h,        # Fully open
                3: mouth_h // 2    # Half open
            }
            h = max(2, phases[self.mouth_phase])

        elif self.state == 'thinking':
            # Draw "..." thinking indicator
            dot_spacing = 8
            dot_size = 4
            start_x = S // 2 - dot_spacing - dot_size
            dot_y = mouth_y - 20

            # Animate dots appearing one by one
            num_dots = (self.thinking_phase % 4)  # 0-3 dots
            for i in range(num_dots):
                self.canvas.create_oval(
                    start_x + i * dot_spacing,
                    dot_y,
                    start_x + i * dot_spacing + dot_size,
                    dot_y + dot_size,
                    fill=self.colors['thinking'],
                    outline=''
                )
            h = 2  # Closed mouth

        else:
            # Idle or listening - closed mouth
            h = 2

        # Draw mouth
        self.canvas.create_rectangle(
            mouth_x, mouth_y, mouth_x + mouth_w, mouth_y + h,
            fill=self.colors['mouth'],
            outline=''
        )

    def _maybe_blink(self):
        """Check if it's time to blink (only in non-speaking states)"""
        now = time.time()
        if self.state != 'speaking' and (now - self.last_blink) > self.next_blink_in:
            self.last_blink = now
            self.next_blink_in = random.uniform(3.0, 6.0)
            return True
        return False

    def _animate(self):
        """Main animation loop - called periodically"""
        if not self.running:
            return

        # Handle blinking
        now = time.time()
        if self.blink_active:
            # Blink lasts ~120ms
            if (now - self.blink_start) > 0.12:
                self.blink_active = False
        elif self._maybe_blink():
            self.blink_active = True
            self.blink_start = now

        # Update animation based on state
        if self.state == 'speaking':
            # Cycle mouth through phases
            self.mouth_phase = (self.mouth_phase + 1) % 4
            fps = 12  # 12 FPS for speaking

        elif self.state == 'thinking':
            # Slow thinking animation
            self.thinking_phase = (self.thinking_phase + 1) % 16  # Slower cycle
            fps = 4  # 4 FPS for thinking

        else:
            # Idle or listening - just blink occasionally
            fps = 6  # 6 FPS for idle

        # Redraw face
        self._draw_face()

        # Schedule next frame
        delay = int(1000 / fps)  # Convert FPS to milliseconds
        self.animation_id = self.canvas.after(delay, self._animate)

    def set_state(self, state):
        """
        Change the face state.

        Args:
            state: One of 'idle', 'listening', 'thinking', 'speaking'
        """
        if state not in ['idle', 'listening', 'thinking', 'speaking']:
            state = 'idle'

        old_state = self.state
        self.state = state

        # Reset animation counters when state changes
        if old_state != state:
            if state == 'speaking':
                self.mouth_phase = 0
            elif state == 'thinking':
                self.thinking_phase = 0

    def start(self):
        """Start the animation loop"""
        if not self.running:
            self.running = True
            self._animate()

    def stop(self):
        """Stop the animation loop"""
        self.running = False
        if self.animation_id:
            self.canvas.after_cancel(self.animation_id)
            self.animation_id = None

    def cleanup(self):
        """Clean up resources"""
        self.stop()
        self.canvas.delete('all')


# Example usage
if __name__ == "__main__":
    # Test the face animator standalone
    root = tk.Tk()
    root.title("Face Animator Test")
    root.geometry("200x250")
    root.configure(bg='#1e1e1e')

    # Create canvas
    canvas = tk.Canvas(
        root,
        width=160,
        height=160,
        bg='#101010',
        highlightthickness=0
    )
    canvas.pack(pady=20)

    # Create face animator
    face = FaceAnimator(canvas, size=160)
    face.start()

    # Control buttons for testing
    button_frame = tk.Frame(root, bg='#1e1e1e')
    button_frame.pack(pady=10)

    tk.Button(
        button_frame,
        text="Idle",
        command=lambda: face.set_state('idle')
    ).pack(side=tk.LEFT, padx=2)

    tk.Button(
        button_frame,
        text="Listening",
        command=lambda: face.set_state('listening')
    ).pack(side=tk.LEFT, padx=2)

    tk.Button(
        button_frame,
        text="Thinking",
        command=lambda: face.set_state('thinking')
    ).pack(side=tk.LEFT, padx=2)

    tk.Button(
        button_frame,
        text="Speaking",
        command=lambda: face.set_state('speaking')
    ).pack(side=tk.LEFT, padx=2)

    root.protocol("WM_DELETE_WINDOW", lambda: (face.cleanup(), root.destroy()))
    root.mainloop()
