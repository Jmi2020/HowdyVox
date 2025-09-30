# Opus Audio Codec Troubleshooting

This guide covers common issues with the Opus audio codec setup required for HowdyTTS wireless audio streaming.

## Issue: Architecture Mismatch on Apple Silicon

### Symptoms
```
OSError: dlopen(/usr/local/lib/libopus.dylib, 0x0006): tried: '/usr/local/lib/libopus.dylib'
(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64'))
```

HowdyTTS crashes on startup when trying to import `opuslib`.

### Root Cause
The libopus library was compiled for Intel (x86_64) architecture, but your Mac uses Apple Silicon (ARM64). This commonly happens when:
- Migrating from an Intel Mac to Apple Silicon
- Using Homebrew installed under Rosetta emulation (`/usr/local` instead of `/opt/homebrew`)
- Installing opus before Python environment was properly configured

### Diagnosis

**1. Check which architecture of opus is installed:**
```bash
# If opus is in /usr/local (Intel location):
file /usr/local/lib/libopus.dylib
# Bad: "Mach-O 64-bit dynamically linked shared library x86_64"

# If opus is in /opt/homebrew (ARM location):
file /opt/homebrew/lib/libopus.dylib
# Good: "Mach-O 64-bit dynamically linked shared library arm64"
```

**2. Check your Mac's architecture:**
```bash
uname -m
# Should show: arm64 (Apple Silicon) or x86_64 (Intel)
```

**3. Check which Homebrew you're using:**
```bash
which brew
# Intel Homebrew: /usr/local/bin/brew
# ARM Homebrew:   /opt/homebrew/bin/brew
```

### Solution

#### Option 1: Reinstall opus for ARM64 (Recommended)

```bash
# Uninstall Intel version
brew uninstall opus

# Reinstall for ARM64
arch -arm64 brew install opus

# Or simply:
brew reinstall opus
```

#### Option 2: Install ARM64 Homebrew (if using Intel Homebrew)

If `which brew` shows `/usr/local/bin/brew`, you're using Intel Homebrew. Install ARM64 version:

```bash
# Install ARM64 Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
eval "$(/opt/homebrew/bin/brew shellenv)"

# Reload shell
source ~/.zshrc  # or source ~/.bash_profile

# Verify ARM Homebrew is active
which brew  # Should show /opt/homebrew/bin/brew

# Install opus
brew install opus
```

#### Option 3: Build opus from source (if brew doesn't work)

```bash
# Install dependencies
brew install autoconf automake libtool

# Download and build opus
cd /tmp
curl -L https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz -o opus.tar.gz
tar -xzf opus.tar.gz
cd opus-1.5.2

# Configure for ARM64
./configure --prefix=/opt/homebrew
make
sudo make install
```

### Verification

**1. Verify opus architecture:**
```bash
file /opt/homebrew/lib/libopus.dylib
# Should show: "Mach-O 64-bit dynamically linked shared library arm64"
```

**2. Test opuslib import:**
```bash
# Activate your conda environment
conda activate howdy310

# Test import
python -c "import opuslib; print('opuslib loaded successfully!')"
```

**3. Start HowdyTTS:**
```bash
python ./launch_howdy_shell.py --wireless
```

You should see:
```
INFO - Generated chunk 1/N: temp/audio/output_chunk_0.opus
```

Instead of:
```
WARNING - opuslib not available, falling back to PCM
```

## Issue: opuslib Installed in Wrong Python Environment

### Symptoms
```
WARNING - opuslib not available, falling back to PCM
```

Even though `pip install opuslib` succeeded.

### Root Cause
opuslib was installed to user site-packages (`~/Library/Python/3.13/...`) instead of the conda environment.

### Solution

**1. Verify installation location:**
```bash
conda activate howdy310
python -m pip show opuslib
```

**Good (conda environment):**
```
Location: /opt/anaconda3/envs/howdy310/lib/python3.10/site-packages
```

**Bad (user site-packages):**
```
Location: /Users/yourname/Library/Python/3.13/lib/python/site-packages
```

**2. Fix installation:**
```bash
# Uninstall from user site-packages
pip uninstall opuslib

# Install in conda environment using Python module
python -m pip install opuslib

# Verify correct location
python -m pip show opuslib
```

## New Machine Setup Checklist

When setting up HowdyTTS on a new machine (especially Apple Silicon):

### 1. Install ARM64 Homebrew (Apple Silicon only)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### 2. Install opus via Homebrew
```bash
brew install opus
```

### 3. Verify opus architecture
```bash
file /opt/homebrew/lib/libopus.dylib
# Should show: arm64 (Apple Silicon) or x86_64 (Intel)
```

### 4. Create conda environment
```bash
conda create -n howdy310 python=3.10
conda activate howdy310
```

### 5. Install Python dependencies
```bash
cd HowdyTTS
pip install -r requirements.txt
```

### 6. Verify opuslib installation
```bash
python -m pip show opuslib
# Location should be in conda environment, not user site-packages

python -c "import opuslib; print('Success!')"
```

### 7. Test HowdyTTS startup
```bash
python ./launch_howdy_shell.py --wireless
```

## Additional Resources

- **Opus Codec**: https://opus-codec.org/
- **opuslib Python bindings**: https://github.com/onbeep/opuslib
- **Homebrew for Apple Silicon**: https://docs.brew.sh/Installation
- **HowdyTTS Requirements**: `requirements.txt`

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `incompatible architecture (have 'x86_64', need 'arm64')` | Wrong opus architecture | Reinstall opus with ARM Homebrew |
| `opuslib not available, falling back to PCM` | opuslib not in conda env | Install with `python -m pip install opuslib` |
| `ModuleNotFoundError: No module named 'opuslib'` | opuslib not installed | Run `pip install opuslib` |
| `ctypes.CDLL: Library not loaded` | libopus not installed | Run `brew install opus` |

## Need Help?

If you encounter issues not covered here:
1. Check that you're using the correct Python environment: `python --version` (should be 3.10.x)
2. Verify conda environment is active: `conda info --envs` (should show `*` next to howdy310)
3. Check system architecture: `uname -m`
4. Review full error logs when starting HowdyTTS