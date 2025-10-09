#!/usr/bin/env python3
"""
Lightweight UI for HowdyVox Voice Assistant
Uses tkinter for minimal overhead and processing power
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import subprocess
import re
from datetime import datetime


class HowdyVoxUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤠 HowdyVox")
        self.root.geometry("600x500")
        self.root.configure(bg='#1e1e1e')

        # Queue for thread-safe UI updates
        self.update_queue = queue.Queue()

        # Status colors - updated to match Howdy's states
        self.status_colors = {
            'waiting': '#00BCD4',    # Cyan - matches console color
            'listening': '#4CAF50',  # Green - matches console color
            'processing': '#FFC107', # Yellow/Amber - matches "Thinking..."
            'thinking': '#FFC107',   # Yellow/Amber - alias for processing
            'speaking': '#9C27B0',   # Magenta/Purple - matches console color
            'ending': '#2196F3',     # Blue - matches console color
            'error': '#F44336'       # Red
        }

        self.create_widgets()
        self.current_status = 'waiting'
        self.process = None

        # Start queue checker
        self.root.after(100, self.check_queue)

    def create_widgets(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(
            header_frame,
            text="🤠 HowdyVox",
            font=('Arial', 20, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        title_label.pack(side=tk.LEFT, padx=10)

        # Status Indicator
        self.status_frame = tk.Frame(header_frame, bg='#2d2d2d')
        self.status_frame.pack(side=tk.RIGHT, padx=10)

        self.status_indicator = tk.Canvas(
            self.status_frame,
            width=15,
            height=15,
            bg='#2d2d2d',
            highlightthickness=0
        )
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_dot = self.status_indicator.create_oval(
            2, 2, 13, 13,
            fill=self.status_colors['waiting']
        )

        self.status_label = tk.Label(
            self.status_frame,
            text="Waiting",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        self.status_label.pack(side=tk.LEFT)

        # Conversation Display
        conv_frame = tk.Frame(self.root, bg='#1e1e1e')
        conv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        tk.Label(
            conv_frame,
            text="Conversation",
            font=('Arial', 12, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff'
        ).pack(anchor=tk.W, pady=(0, 5))

        self.conversation_text = scrolledtext.ScrolledText(
            conv_frame,
            wrap=tk.WORD,
            font=('Courier', 10),
            bg='#2d2d2d',
            fg='#ffffff',
            insertbackground='#ffffff',
            state=tk.DISABLED,
            height=15
        )
        self.conversation_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for coloring
        self.conversation_text.tag_config('user', foreground='#4CAF50')
        self.conversation_text.tag_config('assistant', foreground='#2196F3')
        self.conversation_text.tag_config('system', foreground='#FFA500')
        self.conversation_text.tag_config('error', foreground='#F44336')

        # Control Buttons
        button_frame = tk.Frame(self.root, bg='#1e1e1e')
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_conversation,
            bg='#4CAF50',
            fg='#ffffff',
            font=('Arial', 10),
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = tk.Button(
            button_frame,
            text="Quit",
            command=self.quit_application,
            bg='#F44336',
            fg='#ffffff',
            font=('Arial', 10),
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        self.quit_button.pack(side=tk.RIGHT, padx=5)

    def update_status(self, status):
        """Update the status indicator"""
        self.current_status = status
        color = self.status_colors.get(status, '#FFA500')
        self.status_indicator.itemconfig(self.status_dot, fill=color)

        # Map status to display text
        status_text = {
            'waiting': 'Waiting',
            'listening': 'Listening',
            'processing': 'Thinking',
            'thinking': 'Thinking',
            'speaking': 'Speaking',
            'ending': 'Ending',
            'error': 'Error'
        }.get(status, status.capitalize())

        self.status_label.config(text=status_text)

    def add_message(self, message, tag='system'):
        """Add a message to the conversation display"""
        self.conversation_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")

        if tag == 'user':
            prefix = f"[{timestamp}] You: "
        elif tag == 'assistant':
            prefix = f"[{timestamp}] Howdy: "
        else:
            prefix = f"[{timestamp}] "

        self.conversation_text.insert(tk.END, prefix, tag)
        self.conversation_text.insert(tk.END, f"{message}\n\n")
        self.conversation_text.see(tk.END)
        self.conversation_text.config(state=tk.DISABLED)

    def clear_conversation(self):
        """Clear the conversation display"""
        self.conversation_text.config(state=tk.NORMAL)
        self.conversation_text.delete(1.0, tk.END)
        self.conversation_text.config(state=tk.DISABLED)

    def quit_application(self):
        """Quit the application"""
        if self.process:
            self.process.terminate()
        self.root.quit()

    def check_queue(self):
        """Check for UI updates from the worker thread"""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()

                if update_type == 'status':
                    self.update_status(data)
                elif update_type == 'message':
                    message, tag = data
                    self.add_message(message, tag)

        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.check_queue)

    def queue_status_update(self, status):
        """Thread-safe status update"""
        self.update_queue.put(('status', status))

    def queue_message(self, message, tag='system'):
        """Thread-safe message addition"""
        self.update_queue.put(('message', (message, tag)))

    def parse_output_line(self, line):
        """Parse voice assistant output and update UI accordingly"""
        line = line.strip()

        if not line:
            return

        # Status updates - updated to match current output format
        if 'HowdyVox initialized' in line and 'Say \'Hey Howdy\'' in line:
            # Only show this once by checking if it's the exact initialization message
            self.queue_status_update('waiting')
            self.queue_message("HowdyVox initialized. Say 'Hey Howdy' to start!", 'system')
        elif 'wake word detection active' in line.lower():
            self.queue_status_update('waiting')
        elif 'Wake word detected' in line or 'Listening...' in line:
            self.queue_status_update('listening')
        elif 'Thinking...' in line:
            self.queue_status_update('processing')
        elif 'Ready for your next question' in line:
            self.queue_status_update('listening')
        elif 'Voice assistant stopped' in line or 'Goodbye' in line:
            self.queue_status_update('waiting')
            if 'stopped' in line:
                self.queue_message(line, 'error')
            return
        elif 'error' in line.lower() and 'ERROR' in line:
            # Only treat actual ERROR level logs as errors, not words containing "error"
            self.queue_status_update('error')
            self.queue_message(line, 'error')
            return

        # User transcripts - updated format
        if line.startswith('You:'):
            text = line[4:].strip()  # Remove "You:" prefix
            self.queue_message(text, 'user')

        # Assistant responses - updated format
        elif line.startswith('Howdy:'):
            text = line[6:].strip()  # Remove "Howdy:" prefix
            self.queue_message(text, 'assistant')
            # Set status to speaking when Howdy responds
            self.queue_status_update('speaking')

        # Handle continuation lines (indented paragraphs)
        elif line.startswith('       ') and self.current_status == 'speaking':
            # This is a continuation paragraph
            text = line.strip()
            # Append to last message instead of creating new one
            self.conversation_text.config(state=tk.NORMAL)
            # Remove the last newline
            self.conversation_text.delete("end-2c", "end-1c")
            # Add the continuation
            self.conversation_text.insert(tk.END, f"\n{text}\n\n")
            self.conversation_text.see(tk.END)
            self.conversation_text.config(state=tk.DISABLED)

    def monitor_process(self, process):
        """Monitor the voice assistant process output"""
        self.process = process

        while True:
            line = process.stdout.readline()
            if not line:
                break

            try:
                line = line.decode('utf-8', errors='ignore')
                self.parse_output_line(line)
            except Exception as e:
                pass

        # Process ended
        self.queue_status_update('error')
        self.queue_message('Voice assistant stopped', 'error')


def create_ui():
    """Create and return the UI instance"""
    root = tk.Tk()
    ui = HowdyVoxUI(root)
    return root, ui
