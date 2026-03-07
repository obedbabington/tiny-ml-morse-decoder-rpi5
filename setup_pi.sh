#!/bin/bash
# =============================================================================
# MorseAI Raspberry Pi Setup Script
# =============================================================================
# This script automates the installation of all dependencies on a fresh
# Raspberry Pi OS image.
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

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    print_status "64-bit architecture detected ($ARCH)"
elif [[ "$ARCH" == "armv7l" ]]; then
    print_status "32-bit architecture detected ($ARCH)"
else
    print_warning "Unusual architecture detected: $ARCH"
    print_warning "This script is designed for Raspberry Pi (armv7l or aarch64)"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ "$PYTHON_MAJOR" == "3" && "$PYTHON_MINOR" -ge "13" ]]; then
    print_warning "Python $PYTHON_VERSION detected"
    print_warning "tflite-runtime may not have wheels for Python 3.13+"
    print_warning "If installation fails, consider using Python 3.11 via pyenv"
fi

echo ""
echo "Step 1: Updating system packages..."
echo "----------------------------------------------"
sudo apt update && sudo apt upgrade -y
print_status "System packages updated"

echo ""
echo "Step 2: Installing system dependencies..."
echo "----------------------------------------------"
# Install Python and build tools
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install OpenBLAS (required by NumPy)
sudo apt install -y libopenblas0 liblapack3 || \
    sudo apt install -y libopenblas0-openmp liblapack3 || \
    print_warning "OpenBLAS installation may have failed - NumPy may not work"

# Install GPIO libraries (for system-site-packages)
sudo apt install -y python3-gpiozero python3-rpi.gpio

print_status "System dependencies installed"

echo ""
echo "Step 3: Creating Python virtual environment..."
echo "----------------------------------------------"
VENV_DIR="$HOME/morseai_venv"
if [[ -d "$VENV_DIR" ]]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Remove existing venv and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR" --system-site-packages
        print_status "Virtual environment recreated at $VENV_DIR"
    else
        print_status "Using existing virtual environment"
    fi
else
    python3 -m venv "$VENV_DIR" --system-site-packages
    print_status "Virtual environment created at $VENV_DIR"
fi

echo ""
echo "Step 4: Installing Python dependencies from requirements.txt..."
echo "----------------------------------------------"
source "$VENV_DIR/bin/activate"

# Upgrade pip first
pip install --upgrade pip

# Check if requirements.txt exists
REQUIREMENTS_FILE="$(dirname "$0")/requirements.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    print_error "requirements.txt not found at $REQUIREMENTS_FILE"
    print_error "Please run this script from the MorseAI_RPi5 directory"
    exit 1
fi

# Install all dependencies from requirements.txt
print_status "Installing packages from requirements.txt..."
pip install -r "$REQUIREMENTS_FILE"

print_status "Python dependencies installed"

deactivate

echo ""
echo "Step 5: Enabling GPIO permissions..."
echo "----------------------------------------------"
# Add user to gpio group for non-root GPIO access
if groups $USER | grep -q '\bgpio\b'; then
    print_status "User already in gpio group"
    GPIO_PERMISSIONS_UPDATED=false
else
    sudo usermod -a -G gpio $USER
    print_status "Added user to gpio group (logout/login required)"
    GPIO_PERMISSIONS_UPDATED=true
fi

echo ""
echo "Step 6: Verifying installation..."
echo "----------------------------------------------"
source "$VENV_DIR/bin/activate"

# Verify critical imports
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || {
    print_error "NumPy import failed - check OpenBLAS installation"
    exit 1
}

python3 -c "import tflite_runtime.interpreter as tflite; print('TFLite runtime: OK')" || {
    print_error "TFLite runtime import failed"
    print_error "For Python 3.13+, you may need to use Python 3.11 via pyenv"
    exit 1
}

python3 -c "from gpiozero import Button; print('gpiozero: OK')" || {
    print_error "gpiozero import failed"
    exit 1
}

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
echo "  python main.py"
echo ""
if [[ "$GPIO_PERMISSIONS_UPDATED" == true ]]; then
    print_warning "GPIO permissions were updated - please logout and login again."
fi
echo ""
