# HowdyDust - Current Development Status

**Date**: 2025-09-30
**Branch**: 001-a-private-low
**Last Updated**: After SD Card Recorder implementation

---

## Recent Work Completed

### 1. Audio Playback Issues - RESOLVED ✅

**Problem Timeline**:
- Commit 5a28673: 50ms delay between chunks → SLOW playback (11.7s for 7.3s audio)
- Commit a09c3cb: Removed delay → NO audio (queue overflow)
- Commit fa5496c: **10ms delay → FIXED** ✅

**Root Cause**: ESP32 has fixed-size queue (~10-20 chunks). No delay = all chunks arrive in <100ms → overflow → dropped chunks.

**Solution**: 10ms delay paces transmission without making it glacial:
- 229 chunks × 10ms = 2.3s transmission
- Acceptable for 7.3s audio duration
- Prevents queue overflow while maintaining reasonable speed

**Status**: Audio playback should now work at normal speed. **Needs testing**.

---

### 2. SD Card Recording Component - IMPLEMENTED ✅

**Files Created**:
```
components/sd_recorder/
├── CMakeLists.txt              # Build configuration
├── Kconfig                     # Menuconfig options
├── include/
│   └── sd_recorder.h          # Public API
└── src/
    └── sd_recorder.c          # Full implementation
```

**Features Implemented**:
- SDMMC 4-bit initialization (40 MHz, 20 MB/s throughput)
- GPIO configuration (pins 39-45, no conflicts verified)
- FATFS mounting with error handling
- WAV file format (16kHz, 16-bit, mono PCM)
- Ring buffer architecture (64 KB, PSRAM-backed)
- Automatic file rotation (size-based)
- Separate recording contexts for TTS and microphone
- Non-blocking writes to prevent audio dropouts

**GPIO Mapping** (from specification):
```
SD_PIN_CLK:  GPIO 43
SD_PIN_CMD:  GPIO 44
SD_PIN_D0:   GPIO 39
SD_PIN_D1:   GPIO 40
SD_PIN_D2:   GPIO 41
SD_PIN_D3:   GPIO 42
SD_PIN_PWR:  GPIO 45 (active-low power enable)
```

**Status**: Component complete but **NOT YET INTEGRATED** into audio pipeline.

---

## What Needs to Be Done Next

### Immediate Tasks

#### 1. Test Audio Playback (Python side)
```bash
cd /Users/silverlinings/Desktop/Coding/HowdyDust
python ./HowdyTTS/launch_howdy_shell.py --wireless
```
**Expected Result**: Audio should play at normal speed (not slow, not broken).

#### 2. Build ESP32 Firmware with SD Recorder (ESP-IDF side)

**IMPORTANT**: Must run in IDF environment with `IDF_TARGET=esp32p4`

```bash
# In IDF terminal:
cd /Users/silverlinings/Desktop/Coding/HowdyDust

# Fix WiFi directories (required after fullclean)
./fix_wifi_directories.sh

# Configure SD recorder (optional - defaults are good)
idf.py menuconfig
# Navigate to: SD Card Recorder Configuration
# Default settings:
#   - Enable SD card audio recording: YES
#   - Buffer size: 65536 bytes
#   - Batch size: 16384 bytes
#   - Max file size: 100 MB
#   - SDMMC freq: 40 MHz

# Build
idf.py build

# Flash (user handles this)
idf.py flash monitor
```

#### 3. Integrate SD Recorder into Audio Pipeline

**Files to modify**:
- `components/websocket_client/src/esp32_p4_vad_feedback.c`
  - Add `#include "sd_recorder.h"`
  - Call `sd_recorder_init()` in initialization
  - Call `sd_recorder_write(SD_RECORDER_SOURCE_TTS, ...)` when TTS audio is processed

- `main/HowdyPhase6.c`
  - Initialize SD recorder early in startup
  - Start/stop recording based on conversation state

**Integration points** (from specification):
1. **TTS audio**: Hook into `tts_audio_handler.c` after volume application
2. **Microphone audio**: Hook into I2S capture task after RTP encoding

---

## Known Issues

### Build System
- **IDF_TARGET mismatch**: Environment was set to `esp32s3` but project is `esp32p4`
  - **Solution**: User reset environment variable
  - **Requirement**: Must open in new IDF window for recognition

### Audio Playback
- **Status**: Fixed in commit fa5496c but **NOT TESTED YET**
- **Test required**: Python server with 10ms delay

### SD Card Recording
- **Status**: Component complete but **NOT INTEGRATED**
- **Requires**: Firmware rebuild and integration code

---

## Recent Commits

| Commit | Description |
|--------|-------------|
| fa5496c | fix(tts): add 10ms delay to prevent ESP32 queue overflow |
| 4fff203 | fix(tts): disable Opus file generation and reduce chunks to 1KB |
| 6c135d2 | fix(tts): disable broken Opus encoder and reduce chunk size to 4KB |
| 5a28673 | fix(tts): increase chunk size to 16KB for buffer-then-play approach |
| e71c68b | feat(sd-recorder): add SD card audio recording component |

---

## Configuration Summary

### Python Server (HowdyTTS)
```python
# websocket_tts_server.py
CHUNK_SIZE = 1024  # 1KB chunks
DELAY = 0.01       # 10ms between chunks
USE_OPUS = False   # Disabled (broken opuslib on Apple Silicon)
```

### ESP32 Firmware
```c
// SD Card Recorder (Kconfig defaults)
CONFIG_SD_RECORDER_ENABLED=y
CONFIG_SD_RECORDER_BUFFER_SIZE=65536
CONFIG_SD_RECORDER_BATCH_SIZE=16384
CONFIG_SD_RECORDER_FILE_MAX_SIZE_MB=100
CONFIG_SD_RECORDER_SDMMC_FREQ_MHZ=40
CONFIG_SD_RECORDER_TASK_PRIORITY=5
CONFIG_SD_RECORDER_TASK_CORE=0
```

---

## Technical Specifications

### Audio Format
- **Sample Rate**: 16 kHz
- **Bit Depth**: 16-bit PCM
- **Channels**: Mono (1)
- **Byte Rate**: 32,000 bytes/sec

### SD Card Performance
- **Interface**: SDMMC 4-bit mode
- **Frequency**: 40 MHz
- **Throughput**: 20 MB/s practical (156x margin over required 64 KB/s)
- **Write Pattern**: 16 KB batches every 500ms

### Memory Architecture
- **Ring Buffers**: 2 × 64 KB (TTS + Mic) in PSRAM
- **Total PSRAM Usage**: 128 KB (0.8% of 16 MB)
- **Core Assignment**: Core 0 for SD writes, Core 1 for real-time audio

---

## Documentation References

- `/Users/silverlinings/Desktop/Coding/HowdyDust/HowdyTTS/docs/SD_CARD_STORAGE_SPECIFICATION.md` - Complete SD card specification
- `/Users/silverlinings/Desktop/Coding/HowdyDust/Research/WifiStreamingwSdCard.md` - SD card implementation research
- `/Users/silverlinings/Desktop/Coding/HowdyDust/components/sd_recorder/include/sd_recorder.h` - SD recorder API documentation

---

## Next Session Checklist

When resuming in new IDF window:

- [ ] Verify IDF_TARGET=esp32p4
- [ ] Run `./fix_wifi_directories.sh` if needed
- [ ] Test Python audio playback
- [ ] Build ESP32 firmware with SD recorder
- [ ] Integrate SD recorder into audio pipeline
- [ ] Test SD recording with real TTS and microphone audio
- [ ] Verify WAV files are created correctly
- [ ] Test file rotation at size limits

---

## Contact Points for Integration

### SD Recorder API (Public Functions)
```c
esp_err_t sd_recorder_init(const sd_recorder_config_t *config);
esp_err_t sd_recorder_start(sd_recorder_source_t source);
esp_err_t sd_recorder_write(sd_recorder_source_t source, const uint8_t *data, size_t len);
esp_err_t sd_recorder_stop(sd_recorder_source_t source);
esp_err_t sd_recorder_get_status(sd_recorder_source_t source, sd_recorder_status_t *status);
bool sd_recorder_is_ready(void);
```

### Integration Example
```c
// In initialization:
sd_recorder_config_t config = SD_RECORDER_CONFIG_DEFAULT();
sd_recorder_init(&config);

// Start recording TTS:
sd_recorder_start(SD_RECORDER_SOURCE_TTS);

// In TTS audio callback (after processing):
sd_recorder_write(SD_RECORDER_SOURCE_TTS, pcm_data, pcm_len);

// Stop recording:
sd_recorder_stop(SD_RECORDER_SOURCE_TTS);
```
