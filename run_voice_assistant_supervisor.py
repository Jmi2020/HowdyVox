#!/usr/bin/env python3
"""
Voice Assistant Supervisor - Automatically restarts the voice assistant on crashes
This ensures infinite operation even with segmentation faults
"""

import subprocess
import time
import logging
import signal
import sys
import os
from datetime import datetime
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SUPERVISOR - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/Shared/Coding/HowdyTTS/supervisor.log'),
        logging.StreamHandler()
    ]
)

class VoiceAssistantSupervisor:
    def __init__(self):
        self.process = None
        self.restart_count = 0
        self.running = True
        self.script_path = '/Users/Shared/Coding/HowdyTTS/run_voice_assistant.py'
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n{Fore.YELLOW}SUPERVISOR: Received shutdown signal...{Fore.RESET}")
        self.running = False
            
        if self.process:
            try:
                print(f"{Fore.YELLOW}SUPERVISOR: Terminating voice assistant...{Fore.RESET}")
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"{Fore.RED}SUPERVISOR: Force killing voice assistant...{Fore.RESET}")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"{Fore.RED}SUPERVISOR: Error terminating process: {e}{Fore.RESET}")
        sys.exit(0)
    
    def start_voice_assistant(self):
        """Start the voice assistant process - simple and reliable approach"""
        try:
            print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Fore.RESET}")
            print(f"{Fore.GREEN}SUPERVISOR: Starting HowdyTTS Voice Assistant (attempt #{self.restart_count + 1})...{Fore.RESET}")
            print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Fore.RESET}")
            logging.info(f"Starting voice assistant process (restart #{self.restart_count})")
            
            # Simple approach - just run the command and let it operate normally
            # Use os.system for simplest possible execution that preserves terminal behavior
            conda_cmd = f'cd /Users/Shared/Coding/HowdyTTS && /opt/anaconda3/bin/conda run -n howdy310 python run_voice_assistant.py'
            
            # Start the process in the background but keep terminal control
            self.process = subprocess.Popen(
                conda_cmd,
                shell=True,
                cwd='/Users/Shared/Coding/HowdyTTS'
            )
            
            print(f"{Fore.GREEN}SUPERVISOR: ✅ Voice assistant started (PID: {self.process.pid})!{Fore.RESET}")
            print(f"{Fore.YELLOW}SUPERVISOR: Voice assistant is now running - check for its output above/below{Fore.RESET}")
            print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Fore.RESET}")
            print()
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}SUPERVISOR ERROR: Failed to start voice assistant: {e}{Fore.RESET}")
            logging.error(f"Failed to start voice assistant: {e}")
            return False
    
    def monitor_process(self):
        """Monitor the voice assistant process"""
        if not self.process:
            return False
        
        try:
            # Non-blocking check if process is still running
            return_code = self.process.poll()
            
            if return_code is not None:
                # Process has terminated - use voice assistant styling
                print(f"\n{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Fore.RESET}")
                
                if return_code == 0:
                    logging.info("Voice assistant exited normally")
                    print(f"{Fore.BLUE}SUPERVISOR: Voice assistant exited normally{Fore.RESET}")
                elif return_code == -11:  # SIGSEGV (segmentation fault)
                    logging.warning(f"Voice assistant crashed with segmentation fault (code: {return_code})")
                    print(f"{Fore.RED}SUPERVISOR: ⚠️  SEGMENTATION FAULT DETECTED - Auto-restarting...{Fore.RESET}")
                else:
                    logging.warning(f"Voice assistant exited with code: {return_code}")
                    print(f"{Fore.YELLOW}SUPERVISOR: Process exited with code: {return_code} - Restarting...{Fore.RESET}")
                
                print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Fore.RESET}")
                return False
            
            # Process is still running
            return True
            
        except Exception as e:
            logging.error(f"Error monitoring process: {e}")
            return False
    
    def run(self):
        """Main supervisor loop"""
        print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════╗{Fore.RESET}")
        print(f"{Fore.CYAN}║                    🤠 HowdyTTS Voice Assistant Supervisor               ║{Fore.RESET}")
        print(f"{Fore.CYAN}║                          Infinite Operation Mode                        ║{Fore.RESET}")
        print(f"{Fore.CYAN}╚══════════════════════════════════════════════════════════════════════════╝{Fore.RESET}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop the supervisor{Fore.RESET}")
        print()
        logging.info("Voice Assistant Supervisor started")
        
        while self.running:
            try:
                # Start the voice assistant if not running
                if not self.process or not self.monitor_process():
                    self.restart_count += 1
                    
                    # Add a brief delay before restart to prevent rapid cycling
                    if self.restart_count > 1:
                        delay = min(5, self.restart_count)  # Max 5 second delay
                        print(f"{Fore.YELLOW}SUPERVISOR: Waiting {delay} seconds before restart...{Fore.RESET}")
                        for i in range(delay):
                            print(f"{Fore.YELLOW}  ⏳ {delay - i} seconds remaining...{Fore.RESET}")
                            time.sleep(1)
                    
                    # Cleanup old process
                    if self.process:
                        try:
                            self.process.terminate()
                            self.process.wait(timeout=3)
                        except:
                            pass
                        self.process = None
                    
                    # Start new process
                    if not self.start_voice_assistant():
                        print(f"{Fore.RED}SUPERVISOR ERROR: Failed to start voice assistant, retrying in 10 seconds...{Fore.RESET}")
                        time.sleep(10)
                        continue
                    
                    # Let the voice assistant output flow for a moment before our success message
                    time.sleep(1)
                    print(f"{Fore.GREEN}SUPERVISOR: ✅ Voice assistant is running - output will appear below{Fore.RESET}")
                    print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Fore.RESET}")
                    print()
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(2)  # Longer sleep since we don't need to monitor as frequently
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Supervisor error: {e}")
                print(f"{Fore.RED}SUPERVISOR ERROR: {e}{Fore.RESET}")
                time.sleep(5)
        
        # Cleanup on exit
        print(f"\n{Fore.YELLOW}SUPERVISOR: Shutting down...{Fore.RESET}")
            
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════╗{Fore.RESET}")
        print(f"{Fore.CYAN}║                    🤠 HowdyTTS Supervisor Stopped                       ║{Fore.RESET}")
        print(f"{Fore.CYAN}║                     Total Restarts: {self.restart_count:<3}                           ║{Fore.RESET}")
        print(f"{Fore.CYAN}╚══════════════════════════════════════════════════════════════════════════╝{Fore.RESET}")
        logging.info(f"Voice Assistant Supervisor stopped after {self.restart_count} restarts")

if __name__ == "__main__":
    supervisor = VoiceAssistantSupervisor()
    supervisor.run()
