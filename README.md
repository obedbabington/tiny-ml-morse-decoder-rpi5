# MorseAI - Raspberry Pi Edition

AI-powered Morse Code Decoder using TensorFlow Lite on Raspberry Pi.

This is a port of the STM32-based MorseAI project, optimized for Raspberry Pi (supports both 32-bit and 64-bit OS). It uses TensorFlow Lite for efficient neural network inference instead of the manual C-based implementation.

## Features

- **Real-time Morse code decoding** using a neural network
- **Lightweight TFLite runtime** (no full TensorFlow required)
- **Microsecond-precision timing** for accurate signal capture
- **Hardware debounce filtering** to handle contact bounce
- **Modular architecture** for easy customization

## Project Structure

```
MorseAI_RPi5/
├── main.py              # Main application entry point
├── capture.py           # GPIO signal capture module
├── inference.py         # TFLite inference engine
├── requirements.txt     # Python dependencies
├── setup_pi.sh          # Automated setup script
├── README.md            # This file
└── model/               # Model files (created after conversion)
    ├── morse_classifier.tflite
    └── normalization_config.npz
```

## Hardware Requirements

- Raspberry Pi 5, Pi 4, or Pi 3 (32-bit or 64-bit OS supported)
- Tactile push button (momentary switch)
- 2x jumper wires

## Hardware Setup Guide

### Wiring Diagram

Connect a tactile button to the Raspberry Pi 5 GPIO header:

| Button Pin | RPi5 Pin | RPi5 Function | Description |
|------------|----------|---------------|-------------|
| Terminal 1 | Pin 11   | GPIO 17       | Signal input |
| Terminal 2 | Pin 9    | GND           | Ground |

**Note:** No external pull-up resistor is needed. The code enables the Pi's internal pull-up resistor on GPIO 17.

### GPIO Pin Reference

```
                    Raspberry Pi 5 GPIO Header
                    ┌─────────────────────────┐
            3V3  1  │ ●  ●                    │  2  5V
          GPIO2  3  │ ●  ●                    │  4  5V
          GPIO3  5  │ ●  ●                    │  6  GND
          GPIO4  7  │ ●  ●                    │  8  GPIO14
    ───►  GND   9  │ ●  ●                    │ 10  GPIO15
   │    GPIO17 11  │ ●  ●                    │ 12  GPIO18
   │    GPIO27 13  │ ●  ●                    │ 14  GND
   │    GPIO22 15  │ ●  ●                    │ 16  GPIO23
   │       3V3 17  │ ●  ●                    │ 18  GPIO24
   │    GPIO10 19  │ ●  ●                    │ 20  GND
   │     GPIO9 21  │ ●  ●                    │ 22  GPIO25
   │    GPIO11 23  │ ●  ●                    │ 24  GPIO8
   │       GND 25  │ ●  ●                    │ 26  GPIO7
   │              ...                              ...
   │               └─────────────────────────┘
   │
   │    BUTTON          Connect button between:
   │    ┌─────┐         • Pin 11 (GPIO 17) ← Signal
   └────┤     ├─────────• Pin 9  (GND)     ← Ground
        └─────┘
```

### Internal Pull-Up Resistor Configuration

The STM32 version configured GPIO pull-up via the `GPIO_PUPDR` register:

```c
// STM32 Pull-up configuration (from morse_encode.c)
GPIOC->PUPDR &= ~GPIO_PUPDR_PUPDR13;
GPIOC->PUPDR |= GPIO_PUPDR_PUPDR13_0;  // Pull-up enabled
```

The equivalent Raspberry Pi configuration using `gpiozero`:

```python
# Raspberry Pi Pull-up configuration (from capture.py)
button = Button(
    17,              # GPIO pin (BCM numbering)
    pull_up=True,    # Enable internal pull-up resistor
    bounce_time=0.05 # 50ms debounce filter
)
```

**How it works:**
- With pull-up enabled, GPIO 17 reads HIGH (3.3V) when the button is not pressed
- Pressing the button connects GPIO 17 to GND, pulling it LOW
- The `gpiozero` library detects this falling edge as a "button pressed" event

### Button Selection

For best results, use a **tactile push button** (momentary switch) with:
- Low actuation force (for comfortable tapping)
- Clear tactile feedback
- 6mm x 6mm form factor works well

**Recommended:** OMRON B3F series or similar quality tactile switches.

## Software Setup

### Quick Setup (Recommended)

1. Clone this repository to your Raspberry Pi:
   ```bash
   git clone https://github.com/your-username/MorseAI_RPi5.git
   cd MorseAI_RPi5
   ```

2. Run the automated setup script:
   ```bash
   chmod +x setup_pi.sh
   ./setup_pi.sh
   ```

3. Copy the model files (see [Model Conversion](#model-conversion) below)

4. Run the decoder:
   ```bash
   source ~/morseai_venv/bin/activate
   python3 main.py
   ```

### Manual Setup

1. Install system dependencies:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3 python3-pip python3-venv python3-gpiozero
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv ~/morseai_venv --system-site-packages
   source ~/morseai_venv/bin/activate
   ```

3. Install Python packages:
   ```bash
   pip install numpy tflite-runtime gpiozero
   ```

4. Add user to GPIO group (logout/login required):
   ```bash
   sudo usermod -a -G gpio $USER
   ```

## Model Conversion

Before running the decoder, you need to convert the Keras model to TFLite format.

### Step 1: Add Conversion Code to Notebook

Add this cell to your `MorseAI_Workshop_Notebook.ipynb` after training:

```python
# =============================================================================
# TFLite Conversion for Raspberry Pi 5
# =============================================================================

import numpy as np

def convert_to_tflite(classifier, output_dir="./model_export"):
    """
    Convert the trained Keras model to TFLite format.
    
    Args:
        classifier: Trained MorseCodeClassifier instance.
        output_dir: Directory to save the converted model.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Convert Keras model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier.model)
    
    # Optimize for size and latency (suitable for RPi5)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    model_path = os.path.join(output_dir, "morse_classifier.tflite")
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved: {model_path}")
    print(f"   Size: {len(tflite_model) / 1024:.2f} KB")
    
    # 2. Save normalization parameters
    config_path = os.path.join(output_dir, "normalization_config.npz")
    np.savez(
        config_path,
        mean=classifier.scaler.mean_.astype(np.float32),
        scale=classifier.scaler.scale_.astype(np.float32)
    )
    print(f"✅ Normalization config saved: {config_path}")
    print(f"   Mean: {classifier.scaler.mean_}")
    print(f"   Scale: {classifier.scaler.scale_}")
    
    # 3. Verify the conversion
    print("\n📊 Verifying TFLite model...")
    import tflite_runtime.interpreter as tflite
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    
    print("\n✅ Conversion complete!")
    print(f"\nCopy these files to your Raspberry Pi:")
    print(f"  {model_path}")
    print(f"  {config_path}")
    
    return model_path, config_path


# Run the conversion
convert_to_tflite(classifier)
```

### Step 2: Copy Files to Raspberry Pi

After running the conversion, copy the generated files to your Pi:

```bash
# From your development machine
scp model_export/morse_classifier.tflite pi@raspberrypi:~/MorseAI_RPi5/model/
scp model_export/normalization_config.npz pi@raspberrypi:~/MorseAI_RPi5/model/
```

Or use a USB drive:
1. Copy `model_export/` folder to USB
2. Mount USB on Pi: `sudo mount /dev/sda1 /mnt`
3. Copy files: `cp /mnt/model_export/* ~/MorseAI_RPi5/model/`

## Usage

### Basic Usage

```bash
# Activate virtual environment
source ~/morseai_venv/bin/activate

# Run the decoder
python3 main.py
```

### Command Line Options

```bash
python3 main.py [OPTIONS]

Options:
  -g, --gpio PIN      GPIO pin for button (default: 17)
  -m, --model PATH    Path to TFLite model file
  -c, --config PATH   Path to normalization config file
  -d, --debug         Enable debug output
  -v, --version       Show version number
  -h, --help          Show help message
```

### Examples

```bash
# Use a different GPIO pin
python3 main.py --gpio 27

# Use a custom model
python3 main.py --model /path/to/custom_model.tflite

# Enable debug mode for troubleshooting
python3 main.py --debug
```

### How to Enter Morse Code

1. **Press and hold** the button to create a signal
2. **Release** to end the signal
3. **Short press** (~100ms) = dot (·)
4. **Long press** (~300ms+) = dash (−)
5. **Wait 2 seconds** after releasing to submit the character
6. The AI will predict and display the decoded letter

### Morse Code Reference (A-J)

| Letter | Morse Code | Timing Pattern |
|--------|------------|----------------|
| A | ·− | Short, Long |
| B | −··· | Long, Short, Short, Short |
| C | −·−· | Long, Short, Long, Short |
| D | −·· | Long, Short, Short |
| E | · | Short |
| F | ··−· | Short, Short, Long, Short |
| G | −−· | Long, Long, Short |
| H | ···· | Short, Short, Short, Short |
| I | ·· | Short, Short |
| J | ·−−− | Short, Long, Long, Long |

## Troubleshooting

### GPIO Permission Error

```
RuntimeError: Cannot determine SOC peripheral base address
```

**Solution:** Add user to gpio group and logout/login:
```bash
sudo usermod -a -G gpio $USER
# Then logout and login again
```

### Model Not Found Error

```
FileNotFoundError: Model not found: model/morse_classifier.tflite
```

**Solution:** Run the TFLite conversion in your notebook and copy the files.

### Button Not Responding

1. Check wiring connections
2. Verify GPIO pin number (BCM numbering, not physical pin)
3. Test with: `python3 -c "from gpiozero import Button; b = Button(17); print(b.is_pressed)"`

### Low Accuracy

- Ensure consistent timing when entering Morse code
- Retrain the model with more training data
- Check that normalization parameters match the training data

### TFLite Runtime Installation Issues (32-bit)

If you're on 32-bit Raspberry Pi OS and `tflite-runtime` fails to install:

1. **Try the apt package first:**
   ```bash
   sudo apt install -y python3-tflite-runtime
   ```

2. **If apt package doesn't exist, manually download wheel:**
   ```bash
   # Find the correct wheel for your Python version and architecture
   # Example for Python 3.9 on armv7l:
   wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp39-cp39-linux_armv7l.whl
   pip install tflite_runtime-2.14.0-cp39-cp39-linux_armv7l.whl
   ```

3. **Verify installation:**
   ```bash
   python3 -c "import tflite_runtime.interpreter as tflite; print('OK')"
   ```

## Performance

| Metric | Value |
|--------|-------|
| Model Size | ~15 KB |
| Inference Time | <1 ms |
| Input Latency | <50 ms |
| Accuracy | >95% (on training distribution) |

## Architecture

The system consists of three main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   capture.py    │────▶│    main.py      │────▶│  inference.py   │
│                 │     │                 │     │                 │
│ • GPIO Input    │     │ • Integration   │     │ • TFLite Model  │
│ • Timing        │     │ • CLI Interface │     │ • Normalization │
│ • Debounce      │     │ • Display       │     │ • Prediction    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
    [Button]               [Terminal]              [.tflite]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is part of the Ashesi Research Assistantship program.

## Acknowledgments

- Original STM32 implementation by the TinyML research team
- TensorFlow Lite team for the lightweight runtime
- gpiozero library maintainers
