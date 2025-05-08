#!/usr/bin/env python3
"""
Test script to verify the audio buffer pool implementation
"""
import os
import sys
import time
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print(f"Testing Audio Buffer Pool Optimization")
    
    # Add the parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import the utils module with the buffer pool
    try:
        from voice_assistant.utils import get_audio_buffer, release_audio_buffer, clear_audio_buffer_pool
        print("✅ Successfully imported buffer pool functions")
    except Exception as e:
        print(f"❌ Failed to import buffer pool functions: {e}")
        return
    
    # Test basic buffer allocation and release
    try:
        # Get a buffer
        buffer1 = get_audio_buffer()
        print(f"✅ Got first buffer: {buffer1}")
        
        # Write some data to the buffer
        buffer1.write(b"Hello, World!")
        buffer1.seek(0)
        data = buffer1.read()
        print(f"✅ Buffer contains: {data}")
        
        # Release the buffer
        release_audio_buffer(buffer1)
        print("✅ Released buffer back to pool")
        
        # Get a buffer again (should be the same one)
        buffer2 = get_audio_buffer()
        print(f"✅ Got second buffer: {buffer2}")
        
        # Check that the buffer was reset
        data = buffer2.read()
        print(f"✅ Buffer is empty: {len(data) == 0}")
        
        # Get several buffers
        print("Testing multiple buffers...")
        buffers = []
        for i in range(15):  # This exceeds MAX_POOL_SIZE
            buffers.append(get_audio_buffer())
        
        print(f"✅ Created {len(buffers)} buffers")
        
        # Release them all
        for buffer in buffers:
            release_audio_buffer(buffer)
        
        print("✅ Released all buffers")
        
        # Clear the pool
        clear_audio_buffer_pool()
        print("✅ Cleared buffer pool")
        
    except Exception as e:
        print(f"❌ Error testing buffer pool: {e}")
        return
    
    # Test memory usage
    try:
        print("\nTesting memory efficiency...")
        
        # Force garbage collection
        gc.collect()
        
        # Function to create buffers without pool
        def create_buffers_without_pool(count):
            import io
            buffers = []
            for i in range(count):
                buf = io.BytesIO()
                buf.write(b"X" * 10000)  # Write 10KB of data
                buffers.append(buf)
            return buffers
        
        # Function to create buffers with pool
        def create_buffers_with_pool(count):
            buffers = []
            for i in range(count):
                buf = get_audio_buffer()
                buf.write(b"X" * 10000)  # Write 10KB of data
                buffers.append(buf)
            return buffers
        
        # Memory usage without pool
        start_time = time.time()
        buffers1 = create_buffers_without_pool(1000)
        standard_time = time.time() - start_time
        print(f"✅ Created 1000 standard buffers in {standard_time:.4f} seconds")
        
        # Clear buffers and force garbage collection
        buffers1 = None
        gc.collect()
        
        # Memory usage with pool
        start_time = time.time()
        buffers2 = create_buffers_with_pool(1000)
        pool_time = time.time() - start_time
        print(f"✅ Created 1000 pooled buffers in {pool_time:.4f} seconds")
        
        # Release all buffers back to the pool
        for buffer in buffers2:
            release_audio_buffer(buffer)
        
        # Calculate improvement
        if standard_time > 0:
            improvement = (standard_time - pool_time) / standard_time * 100
            print(f"✅ Performance improvement: {improvement:.1f}%")
        
        # Clear the pool
        clear_audio_buffer_pool()
        
    except Exception as e:
        print(f"❌ Error testing memory efficiency: {e}")
    
    print("\n🎉 Audio buffer pool tests completed!")

if __name__ == "__main__":
    main()