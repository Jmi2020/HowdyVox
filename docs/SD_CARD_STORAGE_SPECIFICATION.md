# SD Card Storage Specification for HowdyDust ESP32-P4

## Executive Summary

This specification defines the integration of SD card storage into the HowdyDust speech companion system to record:
1. Incoming TTS audio from the Python server (WebSocket)
2. Captured microphone audio from ES7210 ADC (I2S)

The implementation prioritizes real-time audio performance while providing persistent storage for debugging, training data collection, and user playback features.

---

## 1. Hardware Requirements

### 1.1 SD Card Interface

**Interface Type**: SDMMC (4-bit mode, NOT SPI)
- **Reasoning**: 4-bit SDMMC provides 40 MB/s sustained writes vs. SPI's ~1 MB/s
- **Clock Speed**: 40 MHz High-Speed mode (SDMMC_FREQ_HIGHSPEED)
- **Bus Width**: 4-bit for maximum throughput

### 1.2 GPIO Pin Mapping (ESP32-P4)

Based on the SD card implementation guide and ESP32-P4 GPIO matrix:

| Function | GPIO Pin | Notes |
|----------|----------|-------|
| SDMMC CLK | GPIO 43 | Clock signal |
| SDMMC CMD | GPIO 44 | Command line |
| SDMMC D0 | GPIO 39 | Data bit 0 |
| SDMMC D1 | GPIO 40 | Data bit 1 |
| SDMMC D2 | GPIO 41 | Data bit 2 |
| SDMMC D3 | GPIO 42 | Data bit 3 |
| SD Power Enable | GPIO 45 | Active-low, controls SD card power |

**Pin Conflicts Check**: These pins do NOT conflict with existing audio peripherals:
- **Audio I2S** (ES7210/ES8311): GPIO 9-13, 53
- **Audio I2C**: GPIO 7, 8
- **Display MIPI DSI**: Dedicated DSI lanes (not GPIO)
- **Touch GT9271**: Shares I2C bus (GPIO 7/8)

### 1.3 SD Card Specifications

- **Card Type**: microSD/SDHC (4-32 GB recommended)
- **Speed Class**: A1 or better (random I/O optimized)
- **UHS Rating**: UHS-I compatible (50 MB/s minimum)
- **Filesystem**: FAT32 (ESP-IDF FATFS driver)
- **Power Requirements**:
  - Idle: ~1 mA
  - Write peaks: 100-200 mA (design power supply accordingly)
  - Use GPIO 45 for controlled power enable/disable

### 1.4 Physical Integration

- **Mounting**: Waveshare ESP32-P4 board has onboard microSD slot
- **Pull-ups**: Internal pull-ups on CMD and D0-D3 (configure via `SDMMC_SLOT_FLAG_INTERNAL_PULLUP`)
- **Signal Integrity**: Keep traces short (<50mm), use matched impedances for D0-D3

---

## 2. Software Architecture

### 2.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     ESP32-P4 Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌─────────────────┐              │
│  │  WebSocket   │────────▶│  TTS Audio      │              │
│  │  RX Task     │  Audio  │  Handler        │──┐           │
│  │  (Core 0)    │  Chunks │  (Core 1)       │  │           │
│  └──────────────┘         └─────────────────┘  │           │
│                                                  │           │
│  ┌──────────────┐         ┌─────────────────┐  │           │
│  │  I2S Capture │────────▶│  Mic Audio      │  │           │
│  │  Task (I2S)  │  PCM    │  Buffer         │  │           │
│  │  ES7210 ADC  │  Data   │  (Core 0)       │  │           │
│  └──────────────┘         └─────────────────┘  │           │
│                                     │           │           │
│                                     │           │           │
│                                     ▼           ▼           │
│                            ┌─────────────────────────┐      │
│                            │  SD Recording Manager   │      │
│                            │  (Core 0, Priority 3)   │      │
│                            └─────────────────────────┘      │
│                                     │                        │
│                                     ▼                        │
│                            ┌─────────────────────────┐      │
│                            │  Ring Buffer (PSRAM)    │      │
│                            │  64 KB per stream       │      │
│                            └─────────────────────────┘      │
│                                     │                        │
│                                     ▼                        │
│                            ┌─────────────────────────┐      │
│                            │  SDMMC Host Driver      │      │
│                            │  (4-bit, 40 MHz)        │      │
│                            └─────────────────────────┘      │
│                                     │                        │
└─────────────────────────────────────┼────────────────────────┘
                                      ▼
                            ┌─────────────────────────┐
                            │  microSD Card (FAT32)   │
                            │  /sdcard/recordings/    │
                            └─────────────────────────┘
```

### 2.2 FreeRTOS Task Layout

| Task Name | Core | Priority | Stack | Purpose |
|-----------|------|----------|-------|---------|
| **websocket_rx** | 0 | 6 (High) | 8192 | Receive TTS audio from server |
| **i2s_capture** | 0 | 6 (High) | 4096 | Capture mic audio from ES7210 |
| **tts_playback** | 1 | 5 (Medium) | 8192 | Play TTS audio to ES8311 |
| **sd_recorder** | 0 | 3 (Low-Med) | 6144 | Write audio to SD card |
| **sd_housekeeping** | 0 | 1 (Low) | 4096 | File rotation, cleanup |

**Core Pinning Strategy**:
- **Core 0**: Network RX, I2S capture, SD writes (I/O intensive)
- **Core 1**: TTS playback, audio processing (real-time critical)

### 2.3 Memory Architecture

**Buffer Design** (using PSRAM for large buffers):

```c
// TTS audio ring buffer (PSRAM)
#define TTS_RECORDING_BUFFER_SIZE (64 * 1024)  // 64 KB = 4 seconds @ 16 kHz

// Mic audio ring buffer (PSRAM)
#define MIC_RECORDING_BUFFER_SIZE (64 * 1024)  // 64 KB = 4 seconds @ 16 kHz

// SD write buffer (internal RAM for DMA)
#define SD_WRITE_BUFFER_SIZE (16 * 1024)  // 16 KB batch writes
```

**Memory Allocation**:
- Ring buffers: PSRAM (ESP32-P4 has 16 MB PSRAM)
- SD DMA buffers: Internal SRAM (for SDMMC DMA compatibility)
- WAV headers: Static allocation (44 bytes)

---

## 3. Integration Points

### 3.1 TTS Audio Recording Integration

**Where to Hook In**: `tts_audio_handler.c` - `tts_playback_task()`

Current flow:
```c
// Existing code in tts_playback_task()
while (1) {
    tts_audio_chunk_t chunk;
    if (xQueueReceive(s_tts_audio.audio_queue, &chunk, portMAX_DELAY)) {
        // Apply volume
        apply_volume(chunk.data, chunk.length, s_tts_audio.config.volume);

        // Write to I2S for playback
        dual_i2s_write_output(chunk.data, chunk.length);

        // FREE CHUNK
    }
}
```

**Modified flow** (add SD recording):
```c
// Modified code with SD recording
while (1) {
    tts_audio_chunk_t chunk;
    if (xQueueReceive(s_tts_audio.audio_queue, &chunk, portMAX_DELAY)) {
        // Apply volume
        apply_volume(chunk.data, chunk.length, s_tts_audio.config.volume);

        // Write to I2S for playback
        dual_i2s_write_output(chunk.data, chunk.length);

        // **NEW**: Send copy to SD recorder
        sd_recorder_write_tts_audio(chunk.data, chunk.length, chunk.timestamp);

        // FREE CHUNK
    }
}
```

### 3.2 Microphone Audio Recording Integration

**Where to Hook In**: `i2s_input.c` or microphone capture task

Current flow:
```c
// Existing microphone capture
while (1) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_1, pcm_buffer, buffer_size, &bytes_read, portMAX_DELAY);

    // Send to RTP encoder for transmission to server
    rtp_encoder_send(pcm_buffer, bytes_read);
}
```

**Modified flow** (add SD recording):
```c
// Modified with SD recording
while (1) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_1, pcm_buffer, buffer_size, &bytes_read, portMAX_DELAY);

    // Send to RTP encoder for transmission to server
    rtp_encoder_send(pcm_buffer, bytes_read);

    // **NEW**: Send copy to SD recorder
    sd_recorder_write_mic_audio(pcm_buffer, bytes_read, esp_timer_get_time());
}
```

### 3.3 SD Recorder API Design

**Header File**: `components/sd_recorder/include/sd_recorder.h`

```c
#ifndef SD_RECORDER_H
#define SD_RECORDER_H

#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Recording configuration
typedef struct {
    uint32_t sample_rate;        // 16000 Hz
    uint16_t bits_per_sample;    // 16
    uint16_t channels;           // 1 (mono)
    bool enable_tts_recording;   // Record TTS audio
    bool enable_mic_recording;   // Record microphone audio
    size_t max_file_size_mb;     // Max file size before rotation (0 = unlimited)
    uint32_t max_recording_duration_sec; // Max duration before rotation (0 = unlimited)
} sd_recorder_config_t;

// Recording session info
typedef struct {
    char tts_filename[64];
    char mic_filename[64];
    uint64_t session_start_time;
    uint32_t tts_bytes_written;
    uint32_t mic_bytes_written;
} sd_recorder_session_info_t;

/**
 * @brief Initialize SD card and recorder
 *
 * @param config Recording configuration
 * @return ESP_OK on success
 */
esp_err_t sd_recorder_init(const sd_recorder_config_t *config);

/**
 * @brief Start a new recording session
 *
 * @param session_name Optional session name (NULL = timestamp-based)
 * @return ESP_OK on success
 */
esp_err_t sd_recorder_start_session(const char *session_name);

/**
 * @brief Stop current recording session
 *
 * @param out_info Optional pointer to receive session info
 * @return ESP_OK on success
 */
esp_err_t sd_recorder_stop_session(sd_recorder_session_info_t *out_info);

/**
 * @brief Write TTS audio chunk to SD card (non-blocking)
 *
 * @param audio_data PCM audio data
 * @param length Data length in bytes
 * @param timestamp Microsecond timestamp
 * @return ESP_OK on success
 */
esp_err_t sd_recorder_write_tts_audio(const uint8_t *audio_data, size_t length, uint64_t timestamp);

/**
 * @brief Write microphone audio chunk to SD card (non-blocking)
 *
 * @param audio_data PCM audio data
 * @param length Data length in bytes
 * @param timestamp Microsecond timestamp
 * @return ESP_OK on success
 */
esp_err_t sd_recorder_write_mic_audio(const uint8_t *audio_data, size_t length, uint64_t timestamp);

/**
 * @brief Get recording statistics
 *
 * @param out_stats Output statistics structure
 * @return ESP_OK on success
 */
esp_err_t sd_recorder_get_stats(sd_recorder_stats_t *out_stats);

/**
 * @brief Deinitialize SD recorder
 *
 * @return ESP_OK on success
 */
esp_err_t sd_recorder_deinit(void);

#ifdef __cplusplus
}
#endif

#endif // SD_RECORDER_H
```

---

## 4. Implementation Approach

### 4.1 Phase 1: SD Card Initialization (Priority 1)

**File**: `components/sd_recorder/src/sd_card_init.c`

**Implementation**:
```c
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"

#define SD_MOUNT_POINT "/sdcard"

esp_err_t sd_card_init(void) {
    ESP_LOGI(TAG, "Initializing SD card");

    // Power on SD card (GPIO 45, active-low)
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << 45),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);
    gpio_set_level(45, 0);  // Active-low enable
    vTaskDelay(pdMS_TO_TICKS(50));  // Wait for power stabilization

    // Configure SDMMC host
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;  // 40 MHz

    // Configure SDMMC slot (4-bit mode)
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 4;
    slot.clk = 43;
    slot.cmd = 44;
    slot.d0 = 39;
    slot.d1 = 40;
    slot.d2 = 41;
    slot.d3 = 42;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    // Mount FATFS
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 8,
        .allocation_unit_size = 16 * 1024  // 16 KB clusters for large files
    };

    sdmmc_card_t *card;
    esp_err_t ret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot, &mount_config, &card);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount SD card: %s", esp_err_to_name(ret));
        return ret;
    }

    // Log card info
    sdmmc_card_print_info(stdout, card);

    ESP_LOGI(TAG, "SD card mounted at %s", SD_MOUNT_POINT);
    return ESP_OK;
}
```

### 4.2 Phase 2: WAV File Management (Priority 2)

**File**: `components/sd_recorder/src/wav_file_writer.c`

**WAV Header Structure**:
```c
typedef struct {
    // RIFF header
    char riff[4];           // "RIFF"
    uint32_t file_size;     // File size - 8
    char wave[4];           // "WAVE"

    // fmt subchunk
    char fmt[4];            // "fmt "
    uint32_t fmt_size;      // 16 for PCM
    uint16_t audio_format;  // 1 for PCM
    uint16_t num_channels;  // 1 for mono
    uint32_t sample_rate;   // 16000
    uint32_t byte_rate;     // sample_rate * channels * bits_per_sample / 8
    uint16_t block_align;   // channels * bits_per_sample / 8
    uint16_t bits_per_sample; // 16

    // data subchunk
    char data[4];           // "data"
    uint32_t data_size;     // Number of data bytes
} __attribute__((packed)) wav_header_t;

void wav_write_header(FILE *fp, uint32_t sample_rate, uint16_t channels,
                      uint16_t bits_per_sample, uint32_t data_bytes) {
    wav_header_t header = {
        .riff = {'R', 'I', 'F', 'F'},
        .file_size = 36 + data_bytes,
        .wave = {'W', 'A', 'V', 'E'},
        .fmt = {'f', 'm', 't', ' '},
        .fmt_size = 16,
        .audio_format = 1,
        .num_channels = channels,
        .sample_rate = sample_rate,
        .byte_rate = sample_rate * channels * (bits_per_sample / 8),
        .block_align = channels * (bits_per_sample / 8),
        .bits_per_sample = bits_per_sample,
        .data = {'d', 'a', 't', 'a'},
        .data_size = data_bytes
    };

    fwrite(&header, sizeof(wav_header_t), 1, fp);
}

void wav_update_header(FILE *fp, uint32_t data_bytes) {
    fseek(fp, 0, SEEK_SET);
    uint32_t file_size = 36 + data_bytes;
    fseek(fp, 4, SEEK_SET);
    fwrite(&file_size, 4, 1, fp);
    fseek(fp, 40, SEEK_SET);
    fwrite(&data_bytes, 4, 1, fp);
    fflush(fp);
}
```

### 4.3 Phase 3: Ring Buffer and Recording Task (Priority 3)

**Ring Buffer Design** (PSRAM):
```c
typedef struct {
    uint8_t *buffer;         // PSRAM buffer
    size_t size;             // Total buffer size
    size_t write_idx;        // Write position
    size_t read_idx;         // Read position
    size_t available;        // Bytes available to read
    SemaphoreHandle_t mutex; // Thread-safe access
} audio_ring_buffer_t;

esp_err_t ring_buffer_init(audio_ring_buffer_t *rb, size_t size) {
    rb->buffer = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
    if (!rb->buffer) {
        return ESP_ERR_NO_MEM;
    }
    rb->size = size;
    rb->write_idx = 0;
    rb->read_idx = 0;
    rb->available = 0;
    rb->mutex = xSemaphoreCreateMutex();
    return ESP_OK;
}

esp_err_t ring_buffer_write(audio_ring_buffer_t *rb, const uint8_t *data, size_t len) {
    xSemaphoreTake(rb->mutex, portMAX_DELAY);

    // Check for overflow
    if (rb->available + len > rb->size) {
        // Overwrite oldest data (circular buffer)
        size_t overflow = (rb->available + len) - rb->size;
        rb->read_idx = (rb->read_idx + overflow) % rb->size;
        rb->available = rb->size - len;
    }

    // Write data (handle wrap-around)
    size_t first_chunk = rb->size - rb->write_idx;
    if (len <= first_chunk) {
        memcpy(rb->buffer + rb->write_idx, data, len);
    } else {
        memcpy(rb->buffer + rb->write_idx, data, first_chunk);
        memcpy(rb->buffer, data + first_chunk, len - first_chunk);
    }

    rb->write_idx = (rb->write_idx + len) % rb->size;
    rb->available += len;

    xSemaphoreGive(rb->mutex);
    return ESP_OK;
}
```

**SD Recording Task**:
```c
#define SD_WRITE_BATCH_SIZE (16 * 1024)  // 16 KB batch writes

static void sd_recorder_task(void *pvParameters) {
    uint8_t *write_buffer = malloc(SD_WRITE_BATCH_SIZE);

    while (1) {
        // Check if recording is active
        if (!s_recorder_state.recording) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // Write TTS audio
        if (s_recorder_state.tts_file && s_tts_ring_buffer.available >= SD_WRITE_BATCH_SIZE) {
            size_t read_len = ring_buffer_read(&s_tts_ring_buffer, write_buffer, SD_WRITE_BATCH_SIZE);
            size_t written = fwrite(write_buffer, 1, read_len, s_recorder_state.tts_file);
            s_recorder_state.tts_bytes_written += written;

            // Periodic flush (every 1 second of audio)
            if (s_recorder_state.tts_bytes_written % (16000 * 2) < SD_WRITE_BATCH_SIZE) {
                fflush(s_recorder_state.tts_file);
            }
        }

        // Write microphone audio
        if (s_recorder_state.mic_file && s_mic_ring_buffer.available >= SD_WRITE_BATCH_SIZE) {
            size_t read_len = ring_buffer_read(&s_mic_ring_buffer, write_buffer, SD_WRITE_BATCH_SIZE);
            size_t written = fwrite(write_buffer, 1, read_len, s_recorder_state.mic_file);
            s_recorder_state.mic_bytes_written += written;

            if (s_recorder_state.mic_bytes_written % (16000 * 2) < SD_WRITE_BATCH_SIZE) {
                fflush(s_recorder_state.mic_file);
            }
        }

        // Small delay to avoid busy-waiting
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    free(write_buffer);
    vTaskDelete(NULL);
}
```

### 4.4 Phase 4: File Management and Rotation (Priority 4)

**File Naming Convention**:
```
/sdcard/recordings/
  ├── session_20250930_123045_tts.wav
  ├── session_20250930_123045_mic.wav
  ├── session_20250930_130122_tts.wav
  ├── session_20250930_130122_mic.wav
  └── ...
```

**File Rotation Logic**:
```c
esp_err_t sd_recorder_check_rotation(void) {
    // Check file size
    if (s_recorder_state.config.max_file_size_mb > 0) {
        uint32_t file_size_mb = s_recorder_state.tts_bytes_written / (1024 * 1024);
        if (file_size_mb >= s_recorder_state.config.max_file_size_mb) {
            ESP_LOGI(TAG, "File size limit reached, rotating");
            return sd_recorder_rotate_files();
        }
    }

    // Check duration
    if (s_recorder_state.config.max_recording_duration_sec > 0) {
        uint64_t duration_sec = (esp_timer_get_time() - s_recorder_state.session_start_time) / 1000000;
        if (duration_sec >= s_recorder_state.config.max_recording_duration_sec) {
            ESP_LOGI(TAG, "Duration limit reached, rotating");
            return sd_recorder_rotate_files();
        }
    }

    return ESP_OK;
}

esp_err_t sd_recorder_rotate_files(void) {
    // Close current files
    if (s_recorder_state.tts_file) {
        wav_update_header(s_recorder_state.tts_file, s_recorder_state.tts_bytes_written);
        fclose(s_recorder_state.tts_file);
    }
    if (s_recorder_state.mic_file) {
        wav_update_header(s_recorder_state.mic_file, s_recorder_state.mic_bytes_written);
        fclose(s_recorder_state.mic_file);
    }

    // Start new session
    return sd_recorder_start_session(NULL);  // NULL = auto-generate timestamp name
}
```

---

## 5. Performance Considerations

### 5.1 Write Performance Targets

**Audio Data Rates**:
- TTS audio: 16 kHz × 2 bytes = 32 KB/s (worst case: 35 KB/s with Opus overhead)
- Mic audio: 16 kHz × 2 bytes = 32 KB/s
- **Combined**: ~64 KB/s sustained write rate

**SD Card Performance**:
- 4-bit SDMMC @ 40 MHz: ~20 MB/s theoretical, ~5-10 MB/s practical
- **Margin**: 10 MB/s ÷ 0.064 MB/s = **156x headroom**

### 5.2 Buffer Sizing Strategy

**Ring Buffer Sizes** (PSRAM):
- TTS buffer: 64 KB = 2 seconds @ 32 KB/s (accommodates WebSocket bursts)
- Mic buffer: 64 KB = 2 seconds @ 32 KB/s (accommodates I2S DMA bursts)
- **Total PSRAM usage**: 128 KB (0.8% of 16 MB PSRAM)

**SD Write Buffer** (internal SRAM):
- 16 KB batch size = 500ms of audio
- Batching reduces SD wear and improves throughput
- Flush every 1 second to ensure data safety

### 5.3 Task Priority Tuning

**Real-time audio MUST NOT be affected by SD writes**:
- **High priority (6)**: WebSocket RX, I2S capture (data producers)
- **Medium priority (5)**: TTS playback (real-time audio output)
- **Low-medium priority (3)**: SD recorder (can tolerate delays)
- **Low priority (1)**: Housekeeping (file rotation, cleanup)

**Core Pinning**:
- Core 0: Network, capture, SD (I/O ops don't block Core 1)
- Core 1: TTS playback (dedicated to real-time audio)

### 5.4 Write Pattern Optimization

**Batch Writes**:
- Accumulate 16 KB before calling `fwrite()`
- Reduces context switches and SD overhead
- Write time: ~1-2 ms per 16 KB batch

**Flush Strategy**:
- Flush every 1 second (not every write)
- Prevents long blocking on SD card sync
- Acceptable data loss in case of power failure: ≤1 second

**No DMA for File I/O**:
- FATFS uses CPU-based writes (not DMA)
- Acceptable since SD writes are low-priority and batched

---

## 6. Error Handling and Recovery

### 6.1 SD Card Mount Failures

**Causes**:
- Card not inserted
- Card format error (not FAT32)
- Hardware connection issues

**Recovery**:
```c
esp_err_t sd_card_init_with_retry(void) {
    const int MAX_RETRIES = 3;

    for (int i = 0; i < MAX_RETRIES; i++) {
        esp_err_t ret = sd_card_init();
        if (ret == ESP_OK) {
            return ESP_OK;
        }

        ESP_LOGW(TAG, "SD mount failed (attempt %d/%d): %s", i+1, MAX_RETRIES, esp_err_to_name(ret));

        // Power cycle SD card
        gpio_set_level(45, 1);  // Disable
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(45, 0);  // Enable
        vTaskDelay(pdMS_TO_TICKS(200));
    }

    ESP_LOGE(TAG, "SD card initialization failed after %d retries", MAX_RETRIES);
    return ESP_FAIL;
}
```

**Fallback Behavior**:
- Continue audio processing WITHOUT SD recording
- Log error to console
- Optionally notify user via UI

### 6.2 SD Card Full

**Detection**:
```c
esp_err_t sd_recorder_check_space(void) {
    struct statvfs stat;
    if (statvfs("/sdcard", &stat) != 0) {
        return ESP_FAIL;
    }

    uint64_t free_bytes = (uint64_t)stat.f_bfree * stat.f_frsize;
    uint64_t required_bytes = 60 * 64 * 1024;  // 60 seconds reserve

    if (free_bytes < required_bytes) {
        ESP_LOGW(TAG, "SD card low on space: %llu bytes free", free_bytes);
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}
```

**Recovery**:
- Stop recording
- Delete oldest files (if auto-cleanup enabled)
- Resume recording

### 6.3 Write Failures

**Detection and Handling**:
```c
size_t written = fwrite(write_buffer, 1, read_len, s_recorder_state.tts_file);
if (written < read_len) {
    ESP_LOGE(TAG, "SD write failed: only %zu/%zu bytes written", written, read_len);
    s_recorder_stats.write_errors++;

    // Check if card is still mounted
    if (access("/sdcard", F_OK) != 0) {
        ESP_LOGE(TAG, "SD card unmounted - stopping recording");
        sd_recorder_stop_session(NULL);
        return ESP_FAIL;
    }
}
```

---

## 7. Configuration and Testing

### 7.1 Menuconfig Integration

Add to `Kconfig`:
```kconfig
menu "SD Card Recorder Configuration"

config SD_RECORDER_ENABLE
    bool "Enable SD card audio recording"
    default y
    help
        Enable recording of TTS and microphone audio to SD card

config SD_RECORDER_TTS_ENABLE
    bool "Record TTS audio"
    depends on SD_RECORDER_ENABLE
    default y

config SD_RECORDER_MIC_ENABLE
    bool "Record microphone audio"
    depends on SD_RECORDER_ENABLE
    default y

config SD_RECORDER_MAX_FILE_SIZE_MB
    int "Max file size before rotation (MB, 0=unlimited)"
    depends on SD_RECORDER_ENABLE
    default 100
    range 0 1000

config SD_RECORDER_MAX_DURATION_SEC
    int "Max recording duration before rotation (seconds, 0=unlimited)"
    depends on SD_RECORDER_ENABLE
    default 300
    range 0 3600

endmenu
```

### 7.2 Test Plan

**Test 1: Basic SD Card Initialization**
```
1. Insert microSD card (FAT32 formatted)
2. Boot ESP32-P4
3. Verify: "SD card mounted at /sdcard" in logs
4. Verify: Card capacity and speed class logged
```

**Test 2: TTS Audio Recording**
```
1. Start recording session
2. Send TTS audio from server
3. Play audio on speaker (verify real-time playback NOT affected)
4. Stop recording session
5. Verify: WAV file created in /sdcard/recordings/
6. Verify: File is playable on PC
```

**Test 3: Microphone Audio Recording**
```
1. Start recording session
2. Speak into microphone
3. Verify: Audio transmitted to server (RTP)
4. Stop recording session
5. Verify: WAV file created and playable
```

**Test 4: Concurrent Recording**
```
1. Start recording session
2. Trigger TTS playback (speaker output)
3. Speak into microphone simultaneously
4. Stop recording session
5. Verify: Both TTS and mic WAV files created
6. Verify: Audio quality is good (no dropouts)
```

**Test 5: File Rotation**
```
1. Configure max_file_size_mb = 1 MB
2. Start recording session
3. Play continuous TTS audio
4. Verify: New file created when size exceeds 1 MB
5. Verify: WAV headers correctly updated in all files
```

**Test 6: Error Recovery**
```
1. Start recording with no SD card inserted
2. Verify: System continues operating (audio playback works)
3. Insert SD card while running
4. Restart recording session
5. Verify: Recording starts successfully
```

---

## 8. Future Enhancements

### 8.1 Compression (Opus Recording)

**Benefit**: 10x size reduction (320 KB/s → 32 KB/s)

**Implementation**: Save Opus frames directly instead of PCM
- Modify ring buffer to store Opus frames
- Write `.opus` files instead of `.wav`
- Update file rotation logic

### 8.2 Metadata and Indexing

**JSON Metadata File**:
```json
{
  "session_id": "20250930_123045",
  "start_time": "2025-09-30T12:30:45Z",
  "duration_sec": 125,
  "tts_file": "session_20250930_123045_tts.wav",
  "mic_file": "session_20250930_123045_mic.wav",
  "tts_bytes": 8000000,
  "mic_bytes": 8000000,
  "sample_rate": 16000,
  "channels": 1,
  "bits_per_sample": 16
}
```

### 8.3 Web Interface for Playback

**HTTP Server**:
- List recordings: `GET /api/recordings`
- Stream recording: `GET /api/recording/{session_id}/tts`
- Delete recording: `DELETE /api/recording/{session_id}`

### 8.4 Cloud Sync

**Upload to Server**:
- WiFi-based upload to HowdyTTS server
- Background task uploads recordings during idle time
- Delete local files after successful upload

---

## 9. Bill of Materials (BOM)

| Item | Specification | Quantity | Notes |
|------|---------------|----------|-------|
| microSD card | 16-32 GB, A1, UHS-I | 1 | SanDisk or Samsung recommended |
| Pull-up resistors | 10kΩ, 0402 | 5 | If not onboard (CMD, D0-D3) |
| Bulk capacitor | 100µF, 6.3V | 1 | SD power supply filtering |
| Ceramic capacitor | 10µF, 6.3V | 1 | SD power supply decoupling |

**Note**: Waveshare ESP32-P4 board likely includes pull-ups and power circuitry. Verify with board schematic.

---

## 10. Implementation Checklist

- [ ] **Phase 1**: SD card initialization
  - [ ] GPIO configuration (power enable)
  - [ ] SDMMC host setup (4-bit, 40 MHz)
  - [ ] FATFS mount
  - [ ] Error handling and retry logic

- [ ] **Phase 2**: WAV file management
  - [ ] WAV header generation
  - [ ] File open/close
  - [ ] Header update on close

- [ ] **Phase 3**: Recording engine
  - [ ] Ring buffer implementation (PSRAM)
  - [ ] SD recorder task
  - [ ] Integration with TTS handler
  - [ ] Integration with mic capture

- [ ] **Phase 4**: File rotation and management
  - [ ] Session start/stop
  - [ ] File size/duration limits
  - [ ] Automatic rotation
  - [ ] Filename generation

- [ ] **Phase 5**: Error handling
  - [ ] Mount failure recovery
  - [ ] Disk full handling
  - [ ] Write error detection

- [ ] **Phase 6**: Testing
  - [ ] Basic SD init test
  - [ ] TTS recording test
  - [ ] Mic recording test
  - [ ] Concurrent recording test
  - [ ] File rotation test
  - [ ] Error recovery test

---

## 11. References

1. ESP-IDF SDMMC Driver: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/storage/sdmmc.html
2. ESP-IDF FATFS Documentation: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/storage/fatfs.html
3. WAV File Format Specification: http://soundfile.sapp.org/doc/WaveFormat/
4. HowdyDust Constitution: `/Users/silverlinings/Desktop/Coding/HowdyDust/.specify/memory/constitution.md`
5. Research Document: `/Users/silverlinings/Desktop/Coding/HowdyDust/Research/WifiStreamingwSdCard.md`

---

## Appendix A: GPIO Conflict Analysis

**ESP32-P4 GPIO Allocation**:

| Peripheral | GPIO Pins | Conflicts |
|------------|-----------|-----------|
| **Audio I2S** (ES7210/ES8311) | 9, 10, 11, 12, 13, 53 | None |
| **Audio I2C** (codec control) | 7, 8 | None |
| **MIPI DSI** (display) | Dedicated DSI lanes | None |
| **Touch I2C** (GT9271) | 7, 8 (shared with audio I2C) | None |
| **SD Card SDMMC** | 39, 40, 41, 42, 43, 44, 45 | **None** |

**Conclusion**: SD card implementation has NO GPIO conflicts with existing peripherals.

---

## Appendix B: Example Directory Structure

```
/sdcard/
├── recordings/
│   ├── session_20250930_120000_tts.wav
│   ├── session_20250930_120000_mic.wav
│   ├── session_20250930_121500_tts.wav
│   ├── session_20250930_121500_mic.wav
│   └── metadata/
│       ├── session_20250930_120000.json
│       └── session_20250930_121500.json
├── config/
│   └── recorder_config.json
└── logs/
    └── recorder.log
```

---

## Appendix C: Power Budget Analysis

**SD Card Power Consumption**:
- Idle: 1 mA @ 3.3V = 3.3 mW
- Read: 30-50 mA @ 3.3V = 99-165 mW
- Write: 100-200 mA @ 3.3V = 330-660 mW

**System Power Budget**:
- ESP32-P4 (active): ~300 mA @ 3.3V = 990 mW
- Audio codecs: ~50 mA @ 3.3V = 165 mW
- Display: ~100 mA @ 3.3V = 330 mW
- SD card (write peak): 200 mA @ 3.3V = 660 mW
- **Total peak**: 650 mA @ 3.3V = 2145 mW

**Power Supply Requirements**:
- Minimum: 1A @ 3.3V (3.3W)
- Recommended: 2A @ 3.3V (6.6W) for margin

**Conclusion**: Ensure power supply can handle 2A continuous current.

---

**End of Specification**
