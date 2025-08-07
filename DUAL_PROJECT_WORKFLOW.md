# HowdyTTS + ESP32-P4 Dual-Project Development Workflow

## ⚠️ CRITICAL DEVELOPMENT GUIDELINE

**ALWAYS work in both projects when implementing solutions!**

## 📂 Project Structure

### **Project Locations:**
- **ESP32-P4 HowdyScreen**: `/Users/silverlinings/Desktop/Coding/ESP32P4/HowdyScreen/`
- **HowdyTTS Server**: `/Users/silverlinings/Desktop/Coding/RBP/HowdyTTS/`

## 🔄 Development Workflow

### **1. Solution Analysis Phase**
- **Identify full scope**: Consider requirements for both ESP32-P4 and HowdyTTS
- **Protocol compatibility**: Ensure data formats, ports, and protocols match
- **Error handling**: Coordinate timeout values and retry logic

### **2. Implementation Phase**
- **Make corresponding changes in both projects**
- **Maintain protocol compatibility** between client and server
- **Test integration points** between the two systems

### **3. Documentation Phase**
- **Update documentation in both locations**
- **Keep protocol specifications synchronized**
- **Document integration points and dependencies**

### **4. Testing Phase**
- **Test each component individually**
- **Verify end-to-end functionality**
- **Validate protocol compatibility**

## 🚀 Examples of Dual-Project Changes

### **Network Protocol Updates**
- **ESP32-P4**: Update WebSocket client configuration
- **HowdyTTS**: Update WebSocket server configuration
- **Both**: Ensure port numbers, message formats, and error handling match

### **Audio Format Changes**
- **ESP32-P4**: Update PCM encoding parameters (sample rate, bit depth, channels)
- **HowdyTTS**: Update PCM decoding to match ESP32-P4 format
- **Both**: Validate audio data integrity and compatibility

### **Discovery Mechanisms**
- **ESP32-P4**: Implement discovery broadcasting and server detection
- **HowdyTTS**: Implement discovery listening and device registration
- **Both**: Coordinate discovery protocols, timeouts, and retry logic

### **Error Handling & Recovery**
- **ESP32-P4**: Implement reconnection logic and failure detection
- **HowdyTTS**: Implement graceful disconnection handling and cleanup
- **Both**: Synchronize timeout values and recovery strategies

## 📋 Checklist for Every Change

### **Before Implementation:**
- [ ] Does this change affect both projects?
- [ ] What protocols need to stay synchronized?
- [ ] Are there dependencies between the two systems?

### **During Implementation:**
- [ ] Update ESP32-P4 code if needed
- [ ] Update HowdyTTS code if needed
- [ ] Ensure protocol compatibility
- [ ] Add appropriate error handling

### **After Implementation:**
- [ ] Update documentation in both projects
- [ ] Test individual components
- [ ] Test end-to-end integration
- [ ] Verify protocol compatibility

## 🎯 Current Integration Points

### **Network Communication**
- **UDP Discovery**: Port 8001 (ESP32-P4 broadcasts, HowdyTTS listens)
- **Audio Streaming**: Port 8000 (ESP32-P4 → HowdyTTS, UDP)
- **WebSocket TTS**: Port 8002 (HowdyTTS → ESP32-P4, WebSocket)

### **Audio Protocols**
- **Input Format**: 16kHz, 16-bit, mono PCM (ESP32-P4 → HowdyTTS)
- **Output Format**: 16kHz, 16-bit, mono PCM (HowdyTTS → ESP32-P4)
- **Packet Structure**: ESP32-P4 protocol with timestamps and metadata

### **Service Discovery**
- **ESP32-P4**: Broadcasts "HOWDYTTS_DISCOVERY" every 2 seconds
- **HowdyTTS**: Responds with "HOWDYTTS_SERVER_{hostname}"
- **Optional**: mDNS advertising as "_howdytts._udp.local."

## 🔧 Common Integration Scenarios

### **Adding New Audio Features**
1. **Design protocol** for the new feature
2. **Implement ESP32-P4 side** (capture, processing, transmission)
3. **Implement HowdyTTS side** (reception, processing, response)
4. **Test integration** with both components running
5. **Document the feature** in both project documentation

### **Network Protocol Changes**
1. **Update protocol specification** in both projects
2. **Modify ESP32-P4 client code** to use new protocol
3. **Modify HowdyTTS server code** to support new protocol
4. **Ensure backward compatibility** if needed
5. **Test with both old and new versions**

### **Error Handling Improvements**
1. **Identify error scenarios** that affect both systems
2. **Implement detection** on both ESP32-P4 and HowdyTTS sides
3. **Coordinate recovery strategies** between both systems
4. **Test failure scenarios** end-to-end
5. **Document error handling** in both projects

## 💡 Best Practices

### **Protocol Design**
- **Keep it simple**: Minimize complexity in cross-system protocols
- **Version compatibility**: Design for future protocol updates
- **Error resilience**: Handle network failures gracefully
- **Performance**: Consider latency and bandwidth requirements

### **Code Organization**
- **Shared constants**: Keep protocol constants synchronized
- **Clear interfaces**: Define clean APIs between systems
- **Error codes**: Use consistent error reporting across systems
- **Logging**: Maintain compatible logging formats

### **Testing Strategy**
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test cross-system communication
- **End-to-end tests**: Test complete user workflows
- **Failure tests**: Test error handling and recovery

## 🎯 Remember

**Every solution should be implemented with both projects in mind!**

The ESP32-P4 and HowdyTTS systems are tightly integrated and dependent on each other. Changes to one system almost always require corresponding changes to the other system to maintain compatibility and functionality.