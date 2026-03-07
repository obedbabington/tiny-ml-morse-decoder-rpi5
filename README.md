# Morse Code Decoder with AI - Raspberry Pi Edition

This project was originally developed for an **Arm workshop** on the STM32 microcontroller. This Raspberry Pi port uses TensorFlow Lite for efficient neural network inference, making it easier to deploy and experiment with. The original STM32 implementation can be found at: [tiny-ml-morse-decoder](https://github.com/obedbabington/tiny-ml-morse-decoder)

**Tested on:** Raspberry Pi 5 with 64-bit OS (also supports 32-bit)

## Demo

[![MorseAI Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*Watch the system in action: real-time Morse code decoding on Raspberry Pi*

## Project Structure

```
MorseAI_RPi5/
├── main.py              # Main application entry point
├── capture.py           # GPIO signal capture module
├── inference.py         # TFLite inference engine
├── requirements.txt     # Python dependencies
├── setup_pi.sh          # Automated setup script
├── training/            # Model development files
│   ├── MorseAI_RPi_Notebook.ipynb
│   └── morse_code_data.csv
└── model/               # Model files (created after conversion)
    ├── morse_classifier.tflite
    └── normalization_config.npz
```

## Hardware Requirements

- **Raspberry Pi:** Pi 5, Pi 4, or Pi 3 (tested on Pi 5)
- **Button:** Tactile push button (momentary switch)
- **Wires:** 2x jumper wires

## Hardware Setup Guide

### Wiring Diagram

Connect a tactile button to the Raspberry Pi 5 GPIO header:

| Button Pin | RPi5 Pin | RPi5 Function | Description |
|------------|----------|---------------|-------------|
| Terminal 1 | Pin 11   | GPIO 17       | Signal input |
| Terminal 2 | Pin 9    | GND           | Ground |

**Note:** No external pull-up resistor is needed. The code enables the Pi's internal pull-up resistor on GPIO 17.


## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/obedbabington/tiny-ml-morse-decoder-rpi5.git
cd tiny-ml-morse-decoder-rpi5
chmod +x setup_pi.sh
./setup_pi.sh
```

The setup script will:
- Install system dependencies
- Create a Python virtual environment
- Install required packages (including TFLite runtime)
- Configure GPIO permissions

### 2. Run the Decoder

```bash
source ~/morseai_venv/bin/activate
python main.py
```

### Testing Components Independently

You can test `capture.py` and `inference.py` separately from `main.py` for debugging:

**Test GPIO signal capture:**
```bash
python capture.py
```
This runs a standalone test that prints raw timing signals as you press the button. Useful for verifying hardware connections and timing accuracy.

**Test model inference:**
```bash
python inference.py
```
This runs the inference engine with predefined test cases, showing predictions and confidence scores. Useful for verifying the model loads correctly and produces expected outputs.

## Software Setup

### Automated Setup

The `setup_pi.sh` script handles everything automatically.

## Model Training & Conversion

The model training notebook is located in `training/MorseAI_RPi_Notebook.ipynb`. The notebook includes all steps from data loading to TFLite conversion.

The notebook will generate:
- `model_export/morse_classifier.tflite` - The neural network model
- `model_export/normalization_config.npz` - Input normalization parameters

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
python main.py --gpio 27

# Use a custom model
python main.py --model /path/to/custom_model.tflite

# Enable debug mode for troubleshooting
python main.py --debug
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


## Architecture

The system consists of three main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   capture.py    │────>│    main.py      │<────│  inference.py   │
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

## Acknowledgments

- Original STM32 implementation: [tiny-ml-morse-decoder](https://github.com/obedbabington/tiny-ml-morse-decoder) (Arm workshop project @Ashesi University)


