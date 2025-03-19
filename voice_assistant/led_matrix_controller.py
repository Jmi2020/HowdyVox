# voice_assistant/led_matrix_controller.py

import requests
import logging
import threading
import time
from urllib.parse import quote
from colorama import Fore

class LEDMatrixController:
    """
    Controller for the ESP32-S3 LED Matrix display.
    Sends state updates to the ESP32 to display scrolling messages.
    """
    
    def __init__(self, esp32_ip=None):
        """
        Initialize the LED Matrix controller.
        
        Args:
            esp32_ip (str): IP address of the ESP32-S3 device
        """
        self.esp32_ip = esp32_ip
        self.state_url = f"http://{esp32_ip}/state" if esp32_ip else None
        self.speak_url = f"http://{esp32_ip}/speak" if esp32_ip else None
        self.enabled = esp32_ip is not None
        self.lock = threading.Lock()  # Thread safety for state changes
        self.current_state = None
        
        if self.enabled:
            logging.info(f"{Fore.CYAN}LED Matrix controller initialized with ESP32 at {esp32_ip}{Fore.RESET}")
            # Try to connect to the ESP32 to verify it's reachable
            self._check_connection()
        else:
            logging.warning(f"{Fore.YELLOW}LED Matrix controller disabled (no ESP32 IP provided){Fore.RESET}")
    
    def _check_connection(self):
        """Check if the ESP32 is reachable and working."""
        try:
            # Send a test request
            self.set_waiting()
            logging.info(f"{Fore.GREEN}Successfully connected to ESP32 LED Matrix at {self.esp32_ip}{Fore.RESET}")
            return True
        except Exception as e:
            logging.error(f"{Fore.RED}Failed to connect to ESP32 LED Matrix: {e}{Fore.RESET}")
            self.enabled = False
            return False
    
    def set_esp32_ip(self, ip):
        """Set or update the ESP32 IP address."""
        with self.lock:
            self.esp32_ip = ip
            self.state_url = f"http://{ip}/state"
            self.speak_url = f"http://{ip}/speak"
            self.enabled = True
            logging.info(f"{Fore.CYAN}ESP32 IP updated to {ip}{Fore.RESET}")
            self._check_connection()
    
    def update_state(self, state):
        """
        Update the LED matrix state.
        
        Args:
            state (str): One of 'listening', 'thinking', 'ending', or 'waiting'
        
        Returns:
            bool: True if the update was successful, False otherwise
        """
        if not self.enabled:
            return False
            
        # Don't send duplicate state updates
        if state == self.current_state:
            return True
            
        try:
            with self.lock:
                response = requests.post(self.state_url, data={"state": state}, timeout=2.0)
                
                if response.status_code == 200:
                    logging.info(f"{Fore.CYAN}LED Matrix state updated to: {state}{Fore.RESET}")
                    self.current_state = state
                    return True
                else:
                    logging.warning(f"{Fore.YELLOW}Failed to update LED Matrix state. Status code: {response.status_code}{Fore.RESET}")
                    return False
        except Exception as e:
            logging.error(f"{Fore.RED}Error updating LED Matrix state: {e}{Fore.RESET}")
            return False
    
    def set_speaking(self, text):
        """
        Set matrix to 'speaking' state with custom text.
        
        Args:
            text (str): The response text to display
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            with self.lock:
                # URL encode the text to handle special characters
                encoded_text = quote(text)
                response = requests.post(self.speak_url, data={"text": encoded_text}, timeout=2.0)
                
                if response.status_code == 200:
                    logging.info(f"{Fore.CYAN}LED Matrix set to speaking mode with text: {text[:30]}...{Fore.RESET}")
                    self.current_state = "speaking"
                    return True
                else:
                    logging.warning(f"{Fore.YELLOW}Failed to set LED Matrix to speaking mode. Status code: {response.status_code}{Fore.RESET}")
                    return False
        except Exception as e:
            logging.error(f"{Fore.RED}Error setting LED Matrix to speaking mode: {e}{Fore.RESET}")
            return False
    
    def set_listening(self):
        """Set matrix to 'listening' state."""
        return self.update_state("listening")
    
    def set_thinking(self):
        """Set matrix to 'thinking' state."""
        return self.update_state("thinking")
    
    def set_ending(self):
        """Set matrix to 'ending' state."""
        return self.update_state("ending")
    
    def set_waiting(self):
        """Set matrix to 'waiting' state."""
        return self.update_state("waiting")