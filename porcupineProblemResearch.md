# Porcupine's 00000136 error: Model-platform mismatch on Apple Silicon

Error code 00000136 in Picovoice Porcupine indicates an `INVALID_ARGUMENT` status, primarily triggered by a platform mismatch between your wake word models and Apple Silicon's arm64 architecture. This issue commonly occurs when mixing custom wake word models with built-in wake words in projects like HowdyTTS on Apple Silicon Macs.

## Platform-specific models cause the majority of errors

The error code 00000136 specifically means your keyword file (.ppn) has an incorrect format or was created for a different platform. When Porcupine attempts to load a wake word model built for x86_64 on an M-series Mac, it fails with this exact error. **Wake word models are strictly platform-specific** and not interchangeable between architectures.

This explains why HowdyTTS encounters errors when trying to use both custom and built-in wake words simultaneously – one or both models likely don't match your Mac's architecture.

## Apple Silicon support requirements

Picovoice added official Apple Silicon (arm64) support in version 2.0, with continuous improvements in subsequent versions. The current version (3.0.5 as of May 2025) fully supports macOS arm64 architecture with Python 3.10.

Common compatibility issues include:
- Using x86_64 wake word models on arm64 systems
- Running older versions lacking proper arm64 support
- Missing native arm64 binaries in certain language SDKs

Users consistently report error 00000136 when trying to use models not specifically built for Apple Silicon, regardless of access key validity.

## Access key configuration impact

While access key issues can cause Porcupine initialization failures, the 00000136 error occurs *after* the access key has been accepted. The error happens during keyword file loading, not during access key validation.

Your access key configuration might still need verification by:
- Checking environment variable loading (if using `os.getenv("PICOVOICE_ACCESS_KEY")`)
- Validating key format (Base64-encoded string without extra spaces)
- Confirming internet connectivity for initial activation

## Version compatibility requirements

The error can also emerge from version mismatches between components:

1. **Library-model synchronization**: Model files (.pv) and keyword files (.ppn) must match your pvporcupine library version
2. **No backward compatibility**: Models from version 1.x won't work with 2.x or 3.x libraries
3. **Simultaneous wake word API changes**: Parameter structures for mixing custom and built-in wake words evolved across versions

This explains why the error might occur specifically when trying to use both custom wake word models and built-in wake words together in HowdyTTS.

## Solutions to resolve error 00000136

1. **Generate platform-specific wake word models**: Create new models through Picovoice Console specifically for macOS arm64
2. **Update to latest version**: Install pvporcupine v3.0.5 which fully supports Apple Silicon
   ```bash
   pip install --upgrade pvporcupine==3.0.5
   ```
3. **Correct Python API usage**: Follow current parameter structure for mixed wake words
   ```python
   handle = pvporcupine.create(
       access_key="YOUR_ACCESS_KEY", 
       keywords=['porcupine'],  # Built-in wake word
       keyword_paths=['/path/to/custom/wake_word.ppn'],  # Custom wake word
       sensitivities=[0.5, 0.6]  # Match total number of wake words
   )
   ```
4. **Implement proper error handling**:
   ```python
   try:
       handle = pvporcupine.create(
           access_key=access_key,
           keywords=['porcupine'],
           keyword_paths=[keyword_path]
       )
   except pvporcupine.PorcupineInvalidArgumentError as e:
       if "00000136" in str(e):
           print("Platform compatibility issue: regenerate wake word model for Apple Silicon")
   ```

## Access key verification approaches

While not directly related to error 00000136, proper access key management ensures smooth initialization:

1. **Environment variable best practices**:
   ```python
   import os
   access_key = os.getenv("PICOVOICE_ACCESS_KEY")
   if not access_key:
       raise ValueError("Access key not set in environment variables")
   ```

2. **Format validation**:
   ```python
   def is_valid_access_key_format(key):
       pattern = r'^[A-Za-z0-9+/=]+$'
       return bool(re.match(pattern, key)) and len(key) > 20
   ```

## Compatible versions for Apple Silicon with Python 3.10

For optimal performance on Apple Silicon Macs running Python 3.10:
- Use pvporcupine v3.0.5 (latest version as of May 2025)
- Install via pip: `pip install pvporcupine==3.0.5`
- Models must be generated specifically for macOS arm64 platform
- Python 3.9+ is officially recommended, with 3.10 fully supported
- Ensure macOS 11+ (Big Sur or later) for optimal compatibility

## Conclusion

Error code 00000136 in HowdyTTS is almost certainly caused by using wake word models not built for Apple Silicon's arm64 architecture. By regenerating your custom wake word models specifically for macOS arm64 and using the latest Porcupine library version, you can successfully use both custom and built-in wake words on your M-series Mac.