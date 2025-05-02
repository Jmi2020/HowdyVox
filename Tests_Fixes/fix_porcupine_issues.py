#!/usr/bin/env python3
"""
Fix script for Porcupine wake word detection issues on Apple Silicon Macs.
This script addresses error 00000136 and other Porcupine-related problems.
"""

import os
import sys
import logging
import subprocess
import re
import platform
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_platform():
    """Check if running on Apple Silicon Mac"""
    print(f"{Fore.BLUE}Checking platform...{Fore.RESET}")
    
    is_macos = platform.system() == "Darwin"
    processor = platform.processor()
    is_arm = "arm" in processor.lower()
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Processor: {processor}")
    
    if is_macos and is_arm:
        print(f"{Fore.GREEN}Running on Apple Silicon Mac.{Fore.RESET}")
        return True
    else:
        print(f"{Fore.YELLOW}Not running on Apple Silicon Mac.{Fore.RESET}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    print(f"{Fore.BLUE}Checking Python version...{Fore.RESET}")
    
    version = platform.python_version()
    version_tuple = tuple(map(int, version.split('.')))
    
    print(f"Python version: {version}")
    
    if version_tuple >= (3, 9):
        print(f"{Fore.GREEN}Python version {version} is compatible with Porcupine 3.x{Fore.RESET}")
        return True
    else:
        print(f"{Fore.RED}Python version {version} may not be fully compatible with Porcupine 3.x{Fore.RESET}")
        print(f"{Fore.YELLOW}Recommended: Python 3.9 or newer{Fore.RESET}")
        return False

def check_porcupine_version():
    """Check Porcupine version and install/update if needed"""
    print(f"{Fore.BLUE}Checking Porcupine version...{Fore.RESET}")
    
    try:
        # Try to import pvporcupine
        import pvporcupine
        version = pvporcupine.__version__
        print(f"Installed Porcupine version: {version}")
        
        # Parse version string to tuple
        version_tuple = tuple(map(int, version.split('.')))
        
        if version_tuple >= (3, 0, 5):
            print(f"{Fore.GREEN}Porcupine version {version} is up-to-date{Fore.RESET}")
            return True
        else:
            print(f"{Fore.YELLOW}Porcupine version {version} is outdated{Fore.RESET}")
            print(f"{Fore.YELLOW}Recommended: Porcupine 3.0.5 or newer{Fore.RESET}")
            
            # Ask to update
            update = input(f"{Fore.CYAN}Update Porcupine to v3.0.5? (y/n): {Fore.RESET}").lower() == 'y'
            
            if update:
                print(f"{Fore.BLUE}Updating Porcupine...{Fore.RESET}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pvporcupine==3.0.5"])
                print(f"{Fore.GREEN}Porcupine updated to v3.0.5{Fore.RESET}")
                return True
            else:
                print(f"{Fore.YELLOW}Continuing with current version{Fore.RESET}")
                return False
                
    except ImportError:
        print(f"{Fore.RED}Porcupine not installed{Fore.RESET}")
        
        # Ask to install
        install = input(f"{Fore.CYAN}Install Porcupine v3.0.5? (y/n): {Fore.RESET}").lower() == 'y'
        
        if install:
            print(f"{Fore.BLUE}Installing Porcupine...{Fore.RESET}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pvporcupine==3.0.5"])
            print(f"{Fore.GREEN}Porcupine v3.0.5 installed{Fore.RESET}")
            return True
        else:
            print(f"{Fore.RED}Porcupine is required for wake word detection{Fore.RESET}")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error checking Porcupine version: {e}{Fore.RESET}")
        return False

def check_wake_word_model():
    """Check wake word model compatibility"""
    print(f"{Fore.BLUE}Checking wake word model...{Fore.RESET}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "Hey-howdy_en_mac_v3_0_0.ppn")
    
    if os.path.exists(model_path):
        print(f"{Fore.GREEN}Found wake word model at:{Fore.RESET} {model_path}")
        
        # Check model platform compatibility
        print(f"{Fore.BLUE}Checking model platform compatibility...{Fore.RESET}")
        print(f"{Fore.YELLOW}Note: The model filename indicates it's for macOS (mac_v3_0_0){Fore.RESET}")
        print(f"{Fore.YELLOW}and should be compatible with Apple Silicon if generated correctly.{Fore.RESET}")
        
        # Try loading the model to test compatibility
        try:
            import pvporcupine
            access_key = os.getenv("PORCUPINE_ACCESS_KEY")
            
            if not access_key:
                print(f"{Fore.RED}PORCUPINE_ACCESS_KEY not found in environment variables{Fore.RESET}")
                set_key = input(f"{Fore.CYAN}Would you like to set it now? (y/n): {Fore.RESET}").lower() == 'y'
                
                if set_key:
                    access_key = input(f"{Fore.CYAN}Enter your Porcupine access key: {Fore.RESET}")
                    # Export key to environment variable for current session
                    os.environ["PORCUPINE_ACCESS_KEY"] = access_key
                    print(f"{Fore.GREEN}Access key set for current session{Fore.RESET}")
                    print(f"{Fore.YELLOW}To make this permanent, add it to your environment variables{Fore.RESET}")
                else:
                    print(f"{Fore.RED}Access key is required to test model compatibility{Fore.RESET}")
                    return False
            
            print(f"{Fore.BLUE}Testing model compatibility...{Fore.RESET}")
            porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[model_path],
                sensitivities=[0.5]
            )
            porcupine.delete()
            print(f"{Fore.GREEN}Wake word model is compatible with your system!{Fore.RESET}")
            return True
            
        except pvporcupine.PorcupineError as e:
            error_str = str(e)
            if "00000136" in error_str:
                print(f"{Fore.RED}Model compatibility issue detected (Error 00000136){Fore.RESET}")
                print(f"{Fore.RED}This indicates the wake word model is not compatible with Apple Silicon.{Fore.RESET}")
                print(f"{Fore.YELLOW}Solution: Generate a new wake word model specifically for macOS arm64.{Fore.RESET}")
                
                # Guide for creating a new model
                print(f"\n{Fore.CYAN}Steps to create a compatible wake word model:{Fore.RESET}")
                print(f"1. Go to Picovoice Console: https://console.picovoice.ai/")
                print(f"2. Log in or create an account")
                print(f"3. Navigate to 'Porcupine' > 'Train Custom Wake Word'")
                print(f"4. Enter 'Hey Howdy' as your wake word")
                print(f"5. Under platforms, select 'macOS'")
                print(f"6. Make sure to select the ARM64 architecture option")
                print(f"7. Train and download the model")
                print(f"8. Replace the existing model in the models directory")
                
                return False
            else:
                print(f"{Fore.RED}Error testing model: {e}{Fore.RESET}")
                return False
        except Exception as e:
            print(f"{Fore.RED}Error testing model: {e}{Fore.RESET}")
            return False
    else:
        print(f"{Fore.RED}Wake word model not found at:{Fore.RESET} {model_path}")
        print(f"{Fore.YELLOW}You need to obtain a compatible wake word model for Apple Silicon.{Fore.RESET}")
        return False

def check_access_key():
    """Check if Porcupine access key is correctly set"""
    print(f"{Fore.BLUE}Checking Porcupine access key...{Fore.RESET}")
    
    access_key = os.getenv("PORCUPINE_ACCESS_KEY")
    
    if not access_key:
        print(f"{Fore.RED}PORCUPINE_ACCESS_KEY not found in environment variables{Fore.RESET}")
        set_key = input(f"{Fore.CYAN}Would you like to set it now? (y/n): {Fore.RESET}").lower() == 'y'
        
        if set_key:
            access_key = input(f"{Fore.CYAN}Enter your Porcupine access key: {Fore.RESET}")
            # Export key to environment for current session
            os.environ["PORCUPINE_ACCESS_KEY"] = access_key
            print(f"{Fore.GREEN}Access key set for current session{Fore.RESET}")
            print(f"{Fore.YELLOW}To make this permanent, add it to your environment variables{Fore.RESET}")
            
            # Validate key format
            is_valid = re.match(r'^[A-Za-z0-9+/=]+$', access_key) and len(access_key) > 20
            if not is_valid:
                print(f"{Fore.RED}Access key format appears invalid{Fore.RESET}")
                print(f"{Fore.YELLOW}It should be a Base64-encoded string without spaces{Fore.RESET}")
                return False
            return True
        else:
            print(f"{Fore.RED}Access key is required for Porcupine wake word detection{Fore.RESET}")
            return False
    else:
        print(f"{Fore.GREEN}PORCUPINE_ACCESS_KEY found in environment variables{Fore.RESET}")
        
        # Check key format
        is_valid = re.match(r'^[A-Za-z0-9+/=]+$', access_key) and len(access_key) > 20
        if not is_valid:
            print(f"{Fore.RED}Access key format appears invalid{Fore.RESET}")
            print(f"{Fore.YELLOW}It should be a Base64-encoded string without spaces{Fore.RESET}")
            return False
            
        return True

def fix_wake_word_file():
    """Check and fix wake word file path case sensitivity"""
    print(f"{Fore.BLUE}Checking wake word file path...{Fore.RESET}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for both casings
    model_path_lowercase = os.path.join(project_root, "models", "Hey-howdy_en_mac_v3_0_0.ppn")
    model_path_uppercase = os.path.join(project_root, "models", "Hey-Howdy_en_mac_v3_0_0.ppn")
    
    lowercase_exists = os.path.exists(model_path_lowercase)
    uppercase_exists = os.path.exists(model_path_uppercase)
    
    if lowercase_exists and not uppercase_exists:
        print(f"{Fore.GREEN}Found wake word model with lowercase 'howdy'{Fore.RESET}")
        
        # Look for references to uppercase in code
        wake_word_py = os.path.join(project_root, "voice_assistant", "wake_word.py")
        
        if os.path.exists(wake_word_py):
            print(f"{Fore.BLUE}Checking wake_word.py for case mismatches...{Fore.RESET}")
            
            with open(wake_word_py, 'r') as f:
                content = f.read()
            
            if "Hey-Howdy_en_mac" in content:
                print(f"{Fore.YELLOW}Found uppercase 'Howdy' in code, but model filename uses lowercase{Fore.RESET}")
                
                fix = input(f"{Fore.CYAN}Fix case in code to match model filename? (y/n): {Fore.RESET}").lower() == 'y'
                
                if fix:
                    new_content = content.replace("Hey-Howdy_en_mac", "Hey-howdy_en_mac")
                    
                    with open(wake_word_py, 'w') as f:
                        f.write(new_content)
                        
                    print(f"{Fore.GREEN}Updated wake_word.py to use lowercase 'howdy'{Fore.RESET}")
                    return True
                else:
                    print(f"{Fore.YELLOW}No changes made{Fore.RESET}")
                    return False
            else:
                print(f"{Fore.GREEN}No case mismatches found in wake_word.py{Fore.RESET}")
                return True
        else:
            print(f"{Fore.RED}Could not find wake_word.py{Fore.RESET}")
            return False
            
    elif uppercase_exists and not lowercase_exists:
        print(f"{Fore.YELLOW}Found wake word model with uppercase 'Howdy'{Fore.RESET}")
        
        # Look for references to lowercase in code
        wake_word_py = os.path.join(project_root, "voice_assistant", "wake_word.py")
        
        if os.path.exists(wake_word_py):
            print(f"{Fore.BLUE}Checking wake_word.py for case mismatches...{Fore.RESET}")
            
            with open(wake_word_py, 'r') as f:
                content = f.read()
            
            if "Hey-howdy_en_mac" in content:
                print(f"{Fore.YELLOW}Found lowercase 'howdy' in code, but model filename uses uppercase{Fore.RESET}")
                
                fix = input(f"{Fore.CYAN}Fix case in code to match model filename? (y/n): {Fore.RESET}").lower() == 'y'
                
                if fix:
                    new_content = content.replace("Hey-howdy_en_mac", "Hey-Howdy_en_mac")
                    
                    with open(wake_word_py, 'w') as f:
                        f.write(new_content)
                        
                    print(f"{Fore.GREEN}Updated wake_word.py to use uppercase 'Howdy'{Fore.RESET}")
                    return True
                else:
                    print(f"{Fore.YELLOW}No changes made{Fore.RESET}")
                    return False
            else:
                print(f"{Fore.GREEN}No case mismatches found in wake_word.py{Fore.RESET}")
                return True
        else:
            print(f"{Fore.RED}Could not find wake_word.py{Fore.RESET}")
            return False
    
    elif lowercase_exists and uppercase_exists:
        print(f"{Fore.YELLOW}Found both lowercase and uppercase wake word models{Fore.RESET}")
        print(f"{Fore.YELLOW}This could cause confusion. Consider removing one.{Fore.RESET}")
        return False
    else:
        print(f"{Fore.RED}No wake word model found in models directory{Fore.RESET}")
        return False

def update_error_handling():
    """Update error handling in wake_word.py"""
    print(f"{Fore.BLUE}Checking for improved error handling...{Fore.RESET}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wake_word_py = os.path.join(project_root, "voice_assistant", "wake_word.py")
    
    if not os.path.exists(wake_word_py):
        print(f"{Fore.RED}Could not find wake_word.py{Fore.RESET}")
        return False
    
    with open(wake_word_py, 'r') as f:
        content = f.read()
    
    # Check if our improved error handling for 00000136 exists
    if "00000136" in content:
        print(f"{Fore.GREEN}Improved error handling for platform mismatch already exists{Fore.RESET}")
        return True
    else:
        print(f"{Fore.YELLOW}Error handling for platform mismatch (00000136) not found{Fore.RESET}")
        
        fix = input(f"{Fore.CYAN}Add improved error handling for platform mismatch? (y/n): {Fore.RESET}").lower() == 'y'
        
        if fix:
            # Print out instructions rather than automatically modifying the file
            # as the exact modifications would depend on the current code structure
            print(f"\n{Fore.YELLOW}=== Manual Fix Instructions ==={Fore.RESET}")
            print(f"1. Open {wake_word_py}")
            print(f"2. Find the try-except block for Porcupine initialization")
            print(f"3. Add specific error handling for PorcupineError with code 00000136:")
            print(f"\n{Style.BRIGHT}Add this code within the try-except block:{Style.RESET_ALL}")
            print("""
try:
    self.porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[model_path],
        sensitivities=[sensitivity]
    )
    logging.info("Porcupine initialized successfully with custom wake word")
except pvporcupine.PorcupineError as e:
    if "00000136" in str(e):
        logging.error("Platform compatibility issue (00000136): %s", e)
        logging.error("This indicates the wake word model is not compatible with Apple Silicon.")
        raise ValueError("Wake word model not compatible with this platform (Apple Silicon)")
    else:
        raise
            """)
            print(f"\n{Fore.YELLOW}This change will properly detect and report platform compatibility issues{Fore.RESET}")
            return False
        else:
            print(f"{Fore.YELLOW}No changes made to error handling{Fore.RESET}")
            return False

def main():
    """Main function to check and fix Porcupine issues"""
    print(f"{Style.BRIGHT}{Fore.CYAN}=== Porcupine Issue Fixer for Apple Silicon Macs ==={Style.RESET_ALL}")
    print(f"This script helps resolve the 00000136 error and other Porcupine issues\n")
    
    issues_found = 0
    issues_fixed = 0
    
    # Check platform
    if check_platform():
        print(f"{Fore.GREEN}✓ Platform check passed{Fore.RESET}\n")
    else:
        print(f"{Fore.YELLOW}⚠ Platform check warning (not Apple Silicon Mac){Fore.RESET}\n")
        issues_found += 1
    
    # Check Python version
    if check_python_version():
        print(f"{Fore.GREEN}✓ Python version check passed{Fore.RESET}\n")
    else:
        print(f"{Fore.YELLOW}⚠ Python version warning{Fore.RESET}\n")
        issues_found += 1
    
    # Check Porcupine version
    if check_porcupine_version():
        print(f"{Fore.GREEN}✓ Porcupine version check passed{Fore.RESET}\n")
        issues_fixed += 1
    else:
        print(f"{Fore.YELLOW}⚠ Porcupine version warning{Fore.RESET}\n")
        issues_found += 1
    
    # Check access key
    if check_access_key():
        print(f"{Fore.GREEN}✓ Access key check passed{Fore.RESET}\n")
    else:
        print(f"{Fore.YELLOW}⚠ Access key warning{Fore.RESET}\n")
        issues_found += 1
    
    # Fix wake word file case sensitivity
    if fix_wake_word_file():
        print(f"{Fore.GREEN}✓ Wake word file check passed{Fore.RESET}\n")
        issues_fixed += 1
    else:
        print(f"{Fore.YELLOW}⚠ Wake word file warning{Fore.RESET}\n")
        issues_found += 1
    
    # Update error handling
    if update_error_handling():
        print(f"{Fore.GREEN}✓ Error handling check passed{Fore.RESET}\n")
    else:
        print(f"{Fore.YELLOW}⚠ Error handling warning{Fore.RESET}\n")
        issues_found += 1
    
    # Check wake word model compatibility
    if check_wake_word_model():
        print(f"{Fore.GREEN}✓ Wake word model compatibility check passed{Fore.RESET}\n")
    else:
        print(f"{Fore.YELLOW}⚠ Wake word model compatibility warning{Fore.RESET}\n")
        issues_found += 1
    
    # Print summary
    print(f"{Style.BRIGHT}{Fore.CYAN}=== Summary ==={Style.RESET_ALL}")
    print(f"Issues found: {issues_found}")
    print(f"Issues fixed: {issues_fixed}")
    
    if issues_found == 0:
        print(f"{Fore.GREEN}No issues found. Porcupine should work correctly.{Fore.RESET}")
    elif issues_fixed == issues_found:
        print(f"{Fore.GREEN}All issues fixed. Porcupine should now work correctly.{Fore.RESET}")
    else:
        print(f"{Fore.YELLOW}Some issues remain. Please follow the instructions above to fix them.{Fore.RESET}")
    
    print(f"\n{Style.BRIGHT}Next steps:{Style.RESET_ALL}")
    print(f"1. Run the test script: python Tests_Fixes/test_porcupine_fixes.py")
    print(f"2. If issues persist, consider using the SpeechRecognitionWakeWord class as a fallback")
    print(f"3. For more information, refer to the Porcupine documentation")

if __name__ == "__main__":
    main()