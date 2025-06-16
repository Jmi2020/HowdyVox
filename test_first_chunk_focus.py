#!/usr/bin/env python3
"""
Focused TTS stuttering test that specifically tests for the original issue:
First chunk stuttering when playback starts before system stabilization.
"""

import os
import time
import logging
from voice_assistant.text_to_speech import text_to_speech, get_next_chunk, generation_complete
from voice_assistant.audio import play_audio
from voice_assistant.config import Config
from voice_assistant.utils import delete_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_first_chunk_stuttering():
    """Test specifically for first chunk stuttering issues."""
    
    print("🎯 Focused First-Chunk Stuttering Test")
    print("=" * 50)
    print("This test focuses on the original issue: first chunk stuttering")
    print("when the system tries to play before full stabilization.")
    print()
    
    # Test the critical case that was failing
    test_text = "Howdy, partner! Welcome to the wonderful world of text-to-speech testing with enhanced buffering capabilities. We're going to put this system through its paces to make sure that the first chunk plays smoothly without any stuttering or audio artifacts. The goal is to ensure that when the TTS system starts generating chunks in the background, the first chunk has enough time to be properly prepared before playback begins. This should eliminate the stuttering issue that was occurring when chunk number one tried to play before the system had fully stabilized."
    
    print(f"🧪 Testing with {len(test_text)} character text")
    print(f"Preview: {test_text[:100]}...")
    print()
    
    try:
        # Generate TTS
        print("⚙️  Generating TTS...")
        start_time = time.time()
        
        success, first_chunk_file = text_to_speech(
            Config.TTS_MODEL,
            "",
            test_text,
            "first_chunk_test.wav",
            Config.LOCAL_MODEL_PATH
        )
        
        generation_time = time.time() - start_time
        print(f"✅ Generation completed in {generation_time:.3f}s")
        
        if not success or not first_chunk_file:
            print("❌ Generation failed!")
            return
        
        print(f"First chunk file: {first_chunk_file}")
        
        # Test immediate playback (the problematic scenario)
        print("\n🎵 Testing immediate playback (original problematic scenario)...")
        print("   Listening for: stuttering, clicks, pops, or audio artifacts")
        print("   Expected: smooth, clean audio from the very start")
        
        immediate_start = time.time()
        play_audio(first_chunk_file)
        immediate_duration = time.time() - immediate_start
        
        print(f"✅ First chunk played in {immediate_duration:.3f}s")
        print("   Did you hear any stuttering or artifacts? (You should NOT)")
        
        # Clean up and test remaining chunks
        files_to_cleanup = [first_chunk_file]
        
        print(f"\n🎵 Playing remaining chunks...")
        chunk_count = 1
        
        while True:
            next_chunk = get_next_chunk()
            
            if next_chunk is None and generation_complete.is_set():
                break
            
            if next_chunk:
                files_to_cleanup.append(next_chunk)
                chunk_count += 1
                print(f"   Playing chunk {chunk_count}...")
                play_audio(next_chunk)
            else:
                if not generation_complete.is_set():
                    time.sleep(0.1)
                else:
                    break
        
        print(f"\n📊 Results:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   First chunk duration: {immediate_duration:.3f}s")
        
        # Cleanup
        for chunk_file in files_to_cleanup:
            if os.path.exists(chunk_file):
                delete_file(chunk_file)
        
        print(f"\n🎯 Success Criteria:")
        print(f"   ✅ First chunk should play smoothly without stuttering")
        print(f"   ✅ No clicking, popping, or audio artifacts")
        print(f"   ✅ Consistent audio quality throughout")
        
        # Manual verification prompt
        print(f"\n❓ Manual Verification Required:")
        user_feedback = input("   Did the first chunk play smoothly without stuttering? (y/n): ").lower().strip()
        
        if user_feedback == 'y' or user_feedback == 'yes':
            print(f"\n🎉 SUCCESS: First chunk stuttering issue RESOLVED!")
            print(f"   The enhanced buffering and timing fixes are working correctly.")
        else:
            print(f"\n⚠️  ISSUE DETECTED: First chunk still showing problems")
            print(f"   Consider adjusting the timing parameters in the implementation.")
            
            timing_feedback = input("   What did you hear? (stuttering/clicking/gaps/other): ").lower().strip()
            print(f"   Feedback noted: {timing_feedback}")
            print(f"   Check TTS_ENHANCEMENT_IMPLEMENTATION.md for tuning options")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        logging.exception("Test failed")

def test_timing_variations():
    """Test different timing scenarios to verify fixes."""
    
    print(f"\n🔧 Additional Timing Tests")
    print("=" * 30)
    
    test_cases = [
        ("Short text", "Hello, this is a quick test."),
        ("Medium text", "This is a medium-length test to verify that our TTS system works properly with different text lengths and doesn't have any stuttering issues."),
        ("Complex text", "Now we're testing with more complex sentences, including various punctuation marks, numbers like 123, and longer phrases that should challenge the system's ability to maintain smooth playback without any audio artifacts or stuttering issues that were present before the enhancement.")
    ]
    
    for name, text in test_cases:
        print(f"\n📝 {name} ({len(text)} chars)")
        
        try:
            success, first_chunk = text_to_speech(
                Config.TTS_MODEL, "", text, f"timing_test_{name.replace(' ', '_')}.wav", Config.LOCAL_MODEL_PATH
            )
            
            if success and first_chunk:
                print(f"   ▶️  Playing {name}...")
                play_audio(first_chunk)
                delete_file(first_chunk)
                print(f"   ✅ Completed")
                
                # Quick cleanup of any additional chunks
                while True:
                    next_chunk = get_next_chunk()
                    if next_chunk is None and generation_complete.is_set():
                        break
                    if next_chunk:
                        play_audio(next_chunk)
                        delete_file(next_chunk)
                    else:
                        if generation_complete.is_set():
                            break
                        time.sleep(0.05)
            else:
                print(f"   ❌ Failed to generate {name}")
                
        except Exception as e:
            print(f"   ❌ Error with {name}: {e}")

if __name__ == "__main__":
    test_first_chunk_stuttering()
    test_timing_variations()
    
    print(f"\n🏁 Testing Complete!")
    print(f"=" * 50)
    print(f"The key metric is whether the FIRST chunk plays smoothly.")
    print(f"Inter-chunk gaps are less critical than first-chunk quality.")
    print(f"If first chunks are smooth, the stuttering fix is working! 🎉")
