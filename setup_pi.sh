#!/bin/bash
# =============================================================================
# MorseAI Raspberry Pi Setup Script
# =============================================================================
# Automates installation of dependencies on Raspberry Pi using Python 3.11
# (required for tflite-runtime compatibility)
#
# Usage: chmod +x setup_pi.sh && ./setup_pi.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

echo "=============================================="
echo "  MorseAI Raspberry Pi 5 - Setup Script"
echo "=============================================="
echo ""

# Check if running from correct directory
REQUIREMENTS_FILE="$(dirname "$0")/requirements.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    print_error "requirements.txt not found"
    print_error "Please run this script from the MorseAI_RPi5 directory"
    exit 1
fi

# Step 1: System packages
echo "Step 1: Updating system packages..."
echo "----------------------------------------------"
sudo apt update && sudo apt upgrade -y
print_status "System packages updated"

# Step 2: Install system dependencies
echo ""
echo "Step 2: Installing system dependencies..."
echo "----------------------------------------------"
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    libopenblas0 liblapack3 \
    python3-gpiozero python3-rpi.gpio \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev git

print_status "System dependencies installed"

# Step 3: Install pyenv and Python 3.11
echo ""
echo "Step 3: Setting up Python 3.11 via pyenv..."
echo "----------------------------------------------"
VENV_DIR="$HOME/morseai_venv"
PYTHON_VERSION="3.11.9"

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    print_status "Installing pyenv..."
    curl https://pyenv.run | bash
    
    # Add pyenv to PATH for this session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    # Add to shell profile for future sessions
    if ! grep -q 'pyenv init' ~/.bashrc; then
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    fi
    print_status "pyenv installed"
else
    print_status "pyenv already installed"
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Install Python 3.11.9 if not already installed
if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    print_status "Python ${PYTHON_VERSION} already installed"
else
    print_status "Installing Python ${PYTHON_VERSION} (this may take a while)..."
    pyenv install -s ${PYTHON_VERSION}
    print_status "Python ${PYTHON_VERSION} installed"
fi

# Step 4: Create virtual environment with Python 3.11
echo ""
echo "Step 4: Creating Python 3.11 virtual environment..."
echo "----------------------------------------------"
if [[ -d "$VENV_DIR" ]]; then
    print_warning "Virtual environment already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        print_status "Using existing virtual environment"
        source "$VENV_DIR/bin/activate"
        PYTHON_CMD="python"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    pyenv local ${PYTHON_VERSION}
    # Use pyenv's python executable directly
    PYTHON_EXEC="$PYENV_ROOT/versions/${PYTHON_VERSION}/bin/python"
    "$PYTHON_EXEC" -m venv "$VENV_DIR"
    print_status "Virtual environment created at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
PYTHON_CMD="python"

# Step 5: Install Python dependencies
echo ""
echo "Step 5: Installing Python dependencies..."
echo "----------------------------------------------"
pip install --upgrade pip

# Install packages from requirements.txt
print_status "Installing packages from requirements.txt..."
pip install -r "$REQUIREMENTS_FILE"

print_status "Python dependencies installed"
deactivate

# Step 6: GPIO permissions
echo ""
echo "Step 6: Configuring GPIO permissions..."
echo "----------------------------------------------"
if groups $USER | grep -q '\bgpio\b'; then
    print_status "User already in gpio group"
    GPIO_UPDATED=false
else
    sudo usermod -a -G gpio $USER
    print_status "Added user to gpio group"
    GPIO_UPDATED=true
fi

# Step 7: Verify installation
echo ""
echo "Step 7: Verifying installation..."
echo "----------------------------------------------"
source "$VENV_DIR/bin/activate"

python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || {
    print_error "NumPy verification failed"
    exit 1
}

python -c "import tflite_runtime.interpreter as tflite; print('TFLite runtime: OK')" || {
    print_error "TFLite runtime verification failed"
    exit 1
}

python -c "from gpiozero import Button; print('GPIO libraries: OK')" || {
    print_warning "GPIO libraries may not be accessible"
    print_warning "Logout/login may be required"
}

deactivate
print_status "All dependencies verified"

# Final summary
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo "Python version: ${PYTHON_VERSION}"
echo ""
echo "To activate:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the decoder:"
echo "  cd $(dirname "$0")"
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py"
echo ""

if [[ "$GPIO_UPDATED" == true ]]; then
    print_warning "GPIO permissions updated - please logout and login again"
    echo ""
    echo "After logging back in, test GPIO:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python -c \"from gpiozero import Button; b = Button(17); print('GPIO OK')\""
    echo ""
fi

print_status "Troubleshooting:"
echo "  • If capture.py doesn't respond: logout/login after GPIO changes"
echo "  • Check button wiring: GPIO 17 (Pin 11) ↔ button ↔ GND (Pin 9)"
echo "  • Test components: python capture.py  or  python inference.py"
echo ""
