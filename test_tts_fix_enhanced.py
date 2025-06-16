#!/usr/bin/env python3
"""
Comprehensive test script to verify the enhanced TTS stuttering fix with detailed timing analysis.
This script tests the text-to-speech function with various lengths of text and monitors
timing, gap analysis, and performance metrics to identify any remaining issues.
"""

import os
import time
import logging
import statistics
from voice_assistant.text_to_speech import text_to_speech, get_next_chunk, generation_complete, get_chunk_generation_stats
from voice_assistant.audio import play_audio
from voice_assistant.config import Config
from voice_assistant.utils import delete_file

# Configure logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_timing_performance(test_name, start_time, chunk_times, gaps):
    """Analyze and report detailed timing performance."""
    total_time = time.time() - start_time
    
    print(f"\n📊 Timing Analysis for {test_name}:")
    print(f"   Total execution time: {total_time:.3f}s")
    print(f"   Number of chunks: {len(chunk_times)}")
    
    if chunk_times:
        avg_chunk_time = statistics.mean(chunk_times)
        print(f"   Average chunk playback: {avg_chunk_time:.3f}s")
        print(f"   Chunk time range: {min(chunk_times):.3f}s - {max(chunk_times):.3f}s")
    
    if gaps:
        avg_gap = statistics.mean(gaps)
        max_gap = max(gaps)
        print(f"   Average inter-chunk gap: {avg_gap:.3f}s")
        print(f"   Maximum gap: {max_gap:.3f}s")
        print(f"   Gap consistency: {'✅ GOOD' if max_gap < 1.0 else '⚠️  NEEDS ATTENTION' if max_gap < 2.0 else '❌ POOR'}")
    
    # Performance rating
    if not gaps:
        rating = "✅ SINGLE CHUNK"
    elif max(gaps) < 0.5:
        rating = "✅ EXCELLENT"
    elif max(gaps) < 1.0:
        rating = "✅ GOOD"
    elif max(gaps) < 2.0:
        rating = "⚠️  ACCEPTABLE"
    else:
        rating = "❌ NEEDS IMPROVEMENT"
    
    print(f"   Performance Rating: {rating}")
    return total_time, avg_gap if gaps else 0

def test_tts_stuttering_fix():
    """Test the TTS system with enhanced timing analysis."""
    
    print("Enhanced TTS Stuttering Fix Test with Comprehensive Analysis")
    print("=" * 70)
    
    # Enhanced test cases with more comprehensive coverage
    test_cases = [
        {
            "name": "Very Short text (minimal processing)",
            "text": "Hello world!",
            "expected_chunks": 1
        },
        {
            "name": "Short text (single chunk)",
            "text": "Hello, this is a short test to verify basic functionality.",
            "expected_chunks": 1
        },
        {
            "name": "Medium text (2-3 chunks)", 
            "text": "Hello there, partner! This is a medium-length test to check if the TTS system works properly without stuttering. We're testing the adaptive delay mechanism that should prevent audio artifacts and ensure smooth playback transitions.",
            "expected_chunks": 2
        },
        {
            "name": "Long text (4+ chunks - CRITICAL TEST)",
            "text": "Howdy, partner! Welcome to the wonderful world of text-to-speech testing with enhanced buffering capabilities. We're going to put this system through its paces to make sure that the first chunk plays smoothly without any stuttering or audio artifacts. The goal is to ensure that when the TTS system starts generating chunks in the background, the first chunk has enough time to be properly prepared before playback begins. This should eliminate the stuttering issue that was occurring when chunk number one tried to play before the system had fully stabilized. The new adaptive buffering system should handle even longer texts like this one with improved chunk sizing and pre-buffering strategies. Let's see how this performs with the enhanced timing mechanisms and whether we can maintain smooth audio playback throughout the entire sequence!",
            "expected_chunks": 5
        },
        {
            "name": "Very Long text (stress test)",
            "text": "This is an extensive stress test for the text-to-speech system that will push the boundaries of our adaptive buffering and chunk management capabilities. We need to ensure that even with very long responses, like the kind you might get from a detailed explanation or a comprehensive answer to a complex question, the system maintains its stability and audio quality. The enhanced TTS implementation should handle multiple chunks seamlessly, with proper timing between each segment to prevent any stuttering, gaps, or audio artifacts that could disrupt the user experience. This test will verify that our pre-buffering strategies, adaptive delays, and queue management systems all work together harmoniously to deliver smooth, professional-quality text-to-speech output regardless of the input length. The system should demonstrate consistent performance across various text complexities while maintaining optimal resource usage and minimizing any potential latency issues that could affect real-time conversation flow.",
            "expected_chunks": 8
        }
    ]
    
    overall_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"🧪 Test {i}: {test_case['name']}")
        print(f"{'='*70}")
        print(f"Text length: {len(test_case['text'])} characters")
        print(f"Expected chunks: {test_case['expected_chunks']}")
        print(f"Text preview: {test_case['text'][:120]}{'...' if len(test_case['text']) > 120 else ''}")
        
        try:
            # Reset timing arrays
            chunk_times = []
            inter_chunk_gaps = []
            
            # Generation Phase
            print(f"\n⚙️  Generation Phase:")
            start_time = time.time()
            
            success, first_chunk_file = text_to_speech(
                Config.TTS_MODEL,
                "",  # No API key needed for Kokoro
                test_case['text'],
                f"test_output_{i}.wav",
                Config.LOCAL_MODEL_PATH
            )
            
            generation_time = time.time() - start_time
            print(f"   Initial generation: {generation_time:.3f}s")
            print(f"   Success: {success}")
            print(f"   First chunk: {first_chunk_file}")
            
            if not success or not first_chunk_file:
                print("❌ Generation failed!")
                continue
            
            # Playback Phase with Enhanced Monitoring
            print(f"\n🎵 Playback Phase with Detailed Monitoring:")
            
            # Track files for cleanup
            files_to_cleanup = [first_chunk_file]
            
            # Simulate the enhanced playback logic from the main application
            playback_start_time = time.time()
            
            # Determine adaptive delay based on text length (matching main app logic)
            response_length = len(test_case['text'])
            if response_length < 80:
                playback_delay = 0.08
            elif response_length < 200:
                playback_delay = 0.12
            elif response_length < 400:
                playback_delay = 0.18
            elif response_length < 800:
                playback_delay = 0.25
            else:
                playback_delay = 0.32
            
            print(f"   Using {playback_delay:.3f}s stabilization delay for {response_length} characters")
            
            # Apply stabilization delay
            time.sleep(playback_delay)
            
            # Play first chunk with timing
            first_chunk_start = time.time()
            print(f"   ▶️  Playing first chunk...")
            play_audio(first_chunk_file)
            first_chunk_duration = time.time() - first_chunk_start
            chunk_times.append(first_chunk_duration)
            print(f"   ✅ First chunk completed in {first_chunk_duration:.3f}s")
            
            # Process remaining chunks with detailed monitoring
            chunk_index = 1
            last_chunk_time = time.time()
            
            while True:
                # Get comprehensive stats
                stats = get_chunk_generation_stats()
                
                # Try to get next chunk
                next_chunk = get_next_chunk()
                
                if next_chunk is None and generation_complete.is_set():
                    break
                
                if next_chunk:
                    # Measure the actual gap between when we finished the last chunk and when we start the next
                    gap_start_time = time.time()
                    inter_chunk_time = gap_start_time - last_chunk_time
                    inter_chunk_gaps.append(inter_chunk_time)
                    files_to_cleanup.append(next_chunk)
                    
                    # Play chunk with timing
                    chunk_start = time.time()
                    print(f"   ▶️  Playing chunk {chunk_index+1} (gap: {inter_chunk_time:.3f}s, queue: {stats['queue_size']})")
                    play_audio(next_chunk)
                    chunk_end = time.time()
                    chunk_duration = chunk_end - chunk_start
                    chunk_times.append(chunk_duration)
                    print(f"   ✅ Chunk {chunk_index+1} completed in {chunk_duration:.3f}s")
                    
                    chunk_index += 1
                    last_chunk_time = chunk_end  # Set to when this chunk finished playing
                else:
                    if stats['generation_complete']:
                        break
                    print(f"   ⏳ Waiting for next chunk... (status: {stats['status']})")
                    time.sleep(0.1)
            
            # Analyze performance
            total_time, avg_gap = analyze_timing_performance(test_case['name'], playback_start_time, chunk_times, inter_chunk_gaps)
            overall_results.append({
                'name': test_case['name'],
                'total_time': total_time,
                'avg_gap': avg_gap,
                'max_gap': max(inter_chunk_gaps) if inter_chunk_gaps else 0,
                'chunks': len(chunk_times)
            })
            
            # Cleanup
            for chunk_file in files_to_cleanup:
                if os.path.exists(chunk_file):
                    delete_file(chunk_file)
            
            print(f"✅ Test {i} completed successfully")
            
        except Exception as e:
            print(f"❌ Error in test {i}: {str(e)}")
            logging.exception(f"Test {i} failed")
    
    # Final Summary Report
    print(f"\n{'='*70}")
    print("🎯 FINAL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    if overall_results:
        for result in overall_results:
            status = "✅ EXCELLENT" if result['max_gap'] < 0.5 else "✅ GOOD" if result['max_gap'] < 1.0 else "⚠️  NEEDS ATTENTION"
            print(f"{result['name']:40} | Max Gap: {result['max_gap']:.3f}s | {status}")
        
        print(f"\n📈 Overall Statistics:")
        avg_total_time = sum(r['total_time'] for r in overall_results) / len(overall_results)
        avg_max_gap = sum(r['max_gap'] for r in overall_results) / len(overall_results)
        print(f"   Average execution time: {avg_total_time:.3f}s")
        print(f"   Average maximum gap: {avg_max_gap:.3f}s")
        
        # Overall system rating
        critical_test = next((r for r in overall_results if "CRITICAL TEST" in r['name']), None)
        if critical_test and critical_test['max_gap'] < 1.0:
            print(f"\n🎉 SYSTEM STATUS: ✅ STUTTERING ISSUE RESOLVED!")
            print(f"   Critical test max gap: {critical_test['max_gap']:.3f}s (< 1.0s threshold)")
        elif critical_test and critical_test['max_gap'] < 2.0:
            print(f"\n⚠️  SYSTEM STATUS: Improved but may need fine-tuning")
            print(f"   Critical test max gap: {critical_test['max_gap']:.3f}s (1.0-2.0s range)")
        else:
            print(f"\n❌ SYSTEM STATUS: Stuttering issue persists - needs investigation")
            if critical_test:
                print(f"   Critical test max gap: {critical_test['max_gap']:.3f}s (> 2.0s)")
    
    print(f"\n🎯 What to listen for during testing:")
    print("✅ Smooth, natural playback from the very first chunk")
    print("✅ Consistent audio quality across all chunks")
    print("✅ No gaps, stutters, or audio artifacts")
    print("❌ Any clicking, popping, or stuttering sounds")
    print("❌ Unnatural pauses between chunks")
    print("❌ Audio quality degradation")
    
    print(f"\n📋 Test completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    test_tts_stuttering_fix()
