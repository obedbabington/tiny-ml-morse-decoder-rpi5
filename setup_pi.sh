#!/bin/bash
# =============================================================================
# MorseAI Raspberry Pi Setup Script
# =============================================================================
# This script automates the installation of all dependencies on a fresh
# Raspberry Pi OS image (supports both 32-bit and 64-bit).
#
# Usage: chmod +x setup_pi.sh && ./setup_pi.sh
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "  MorseAI Raspberry Pi 5 - Setup Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on Raspberry Pi
if [[ ! -f /proc/device-tree/model ]]; then
    print_warning "Cannot detect Raspberry Pi model. Proceeding anyway..."
else
    MODEL=$(cat /proc/device-tree/model)
    echo "Detected: $MODEL"
fi

# Check architecture (supports both 32-bit and 64-bit)
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    print_status "64-bit architecture detected ($ARCH)"
    IS_64BIT=true
elif [[ "$ARCH" == "armv7l" ]]; then
    print_status "32-bit architecture detected ($ARCH)"
    IS_64BIT=false
else
    print_warning "Unusual architecture detected: $ARCH"
    print_warning "This script is designed for Raspberry Pi (armv7l or aarch64)"
    print_warning "Proceeding anyway, but TFLite runtime installation may fail..."
    IS_64BIT=false
fi

echo ""
echo "Step 1: Updating system packages..."
echo "----------------------------------------------"
sudo apt update && sudo apt upgrade -y
print_status "System packages updated"

echo ""
echo "Step 2: Installing Python and pip..."
echo "----------------------------------------------"
sudo apt install -y python3 python3-pip python3-venv python3-dev
print_status "Python3 and pip installed"

echo ""
echo "Step 3: Installing GPIO libraries..."
echo "----------------------------------------------"
sudo apt install -y python3-gpiozero python3-rpi.gpio
print_status "GPIO libraries installed"

echo ""
echo "Step 4: Creating Python virtual environment..."
echo "----------------------------------------------"
VENV_DIR="$HOME/morseai_venv"
if [[ -d "$VENV_DIR" ]]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR" --system-site-packages
    print_status "Virtual environment created at $VENV_DIR"
fi

echo ""
echo "Step 5: Installing Python dependencies..."
echo "----------------------------------------------"
source "$VENV_DIR/bin/activate"

# Install numpy first (required by tflite-runtime)
pip install --upgrade pip
pip install "numpy>=1.24.0,<2.0.0"
print_status "NumPy installed"

# Install TFLite runtime (architecture-specific)
if [[ "$IS_64BIT" == true ]]; then
    # 64-bit: Use pip to install tflite-runtime wheel
    print_status "Installing TFLite runtime for 64-bit (aarch64)..."
    pip install tflite-runtime
    print_status "TensorFlow Lite runtime installed"
else
    # 32-bit: Try apt package first (more reliable on 32-bit OS)
    print_status "Installing TFLite runtime for 32-bit (armv7l)..."
    if sudo apt install -y python3-tflite-runtime 2>/dev/null; then
        print_status "TensorFlow Lite runtime installed via apt"
    else
        print_warning "apt package not available, trying pip..."
        # Fallback to pip (may need specific wheel URL)
        pip install tflite-runtime || {
            print_error "Failed to install tflite-runtime via pip"
            print_error "You may need to manually install a compatible wheel"
            print_error "Check: https://www.tensorflow.org/lite/guide/python"
            exit 1
        }
        print_status "TensorFlow Lite runtime installed via pip"
    fi
fi

# Install gpiozero (should already be available via system-site-packages)
pip install gpiozero
print_status "gpiozero installed"

deactivate

echo ""
echo "Step 6: Enabling GPIO permissions..."
echo "----------------------------------------------"
# Add user to gpio group for non-root GPIO access
if groups $USER | grep -q '\bgpio\b'; then
    print_status "User already in gpio group"
else
    sudo usermod -a -G gpio $USER
    print_status "Added user to gpio group (logout/login required)"
fi

echo ""
echo "Step 7: Verifying installation..."
echo "----------------------------------------------"
source "$VENV_DIR/bin/activate"

python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import tflite_runtime.interpreter as tflite; print('TFLite runtime: OK')"
python3 -c "from gpiozero import Button; print('gpiozero: OK')"

deactivate
print_status "All dependencies verified"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the virtual environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the MorseAI decoder:"
echo "  cd $(dirname "$0")"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 main.py"
echo ""
print_warning "If GPIO permissions were updated, please logout and login again."
echo ""
