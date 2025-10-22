#!/usr/bin/env python3
"""
Test script for the animated face UI
Tests all face states without needing the full voice assistant
"""

import tkinter as tk
from ui_interface import create_ui
import time


def test_face_states(ui_instance):
    """Cycle through all face states to test animations"""
    states = [
        ('waiting', 3000),      # Wait 3 seconds
        ('listening', 3000),    # Listen for 3 seconds
        ('thinking', 3000),     # Think for 3 seconds
        ('speaking', 5000),     # Speak for 5 seconds (longer to see mouth animation)
        ('ending', 2000),       # End for 2 seconds
        ('waiting', 2000)       # Back to waiting
    ]

    def cycle_states(index=0):
        if index < len(states):
            state, duration = states[index]

            # Update status (which automatically updates face)
            ui_instance.queue_status_update(state)

            # Add test message
            messages = {
                'waiting': ('System ready', 'system'),
                'listening': ('User is speaking...', 'system'),
                'thinking': ('Processing your request...', 'system'),
                'speaking': ('This is a test response from Howdy!', 'assistant'),
                'ending': ('Conversation ending...', 'system')
            }

            if state in messages:
                msg, tag = messages[state]
                ui_instance.queue_message(msg, tag)

            # Schedule next state
            ui_instance.root.after(duration, lambda: cycle_states(index + 1))
        else:
            # Test complete
            ui_instance.queue_message('Face animation test completed!', 'system')

    # Start the state cycle after a short delay
    ui_instance.root.after(1000, cycle_states)


def main():
    """Main test function"""
    print("🧪 Testing HowdyVox Animated Face UI")
    print("=" * 50)
    print("\nThis will cycle through all face states:")
    print("  • Waiting (idle, blinking)")
    print("  • Listening (eyes open)")
    print("  • Thinking ('...' animation)")
    print("  • Speaking (mouth animation)")
    print("  • Ending (back to idle)")
    print("\nWatch the face on the left side of the UI!")
    print("=" * 50)

    # Create UI
    root, ui = create_ui()

    # Add initial message
    ui.add_message("Face animation test starting in 1 second...", 'system')

    # Start test cycle
    test_face_states(ui)

    # Run UI
    root.mainloop()


if __name__ == "__main__":
    main()
