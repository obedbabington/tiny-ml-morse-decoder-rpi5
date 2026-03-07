#!/usr/bin/env python3
"""
MorseAI - Morse Code Neural Network Decoder for Raspberry Pi 5

Main entry point that connects GPIO signal capture with TensorFlow Lite inference
to decode Morse code in real-time.

Usage:
    python3 main.py [--gpio PIN] [--model PATH] [--debug]

Hardware:
    - Tactile button connected to GPIO 17 (default) and GND
    - Uses internal pull-up resistor (no external resistors needed)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

from capture import MorseCapture, GPIO_PIN, TIMEOUT_SECONDS, BOUNCE_TIME
from inference import MorseInference, InferenceResult, display_result

# Configuration
VERSION = "1.0.0"
DEFAULT_MODEL_DIR = Path(__file__).parent / "model"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "morse_classifier.tflite"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_DIR / "normalization_config.npz"


class MorseAIDecoder:
    """
    Main application that integrates GPIO capture with neural network inference.
    
    Connects the hardware button input (capture.py) with the TensorFlow Lite
    inference engine (inference.py) to decode Morse code in real-time.
    """
    
    def __init__(
        self,
        gpio_pin: int = GPIO_PIN,
        model_path: str = None,
        config_path: str = None,
        debug: bool = False
    ):
        """
        Initialize the MorseAI decoder.
        
        Args:
            gpio_pin: GPIO pin for the button (BCM numbering).
            model_path: Path to the TFLite model file (defaults to model/morse_classifier.tflite).
            config_path: Path to normalization config file (defaults to model/normalization_config.npz).
            debug: Enable debug output.
        """
        self.gpio_pin = gpio_pin
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.config_path = config_path or str(DEFAULT_CONFIG_PATH)
        self.debug = debug
        
        self._inference_engine = None
        self._capture = None
        self._running = False
        self._decoded_message = []
    
    def _log(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def _on_character_complete(self, signals: List[float]) -> None:
        """
        Callback when a character capture is complete.
        
        This is called by the capture module when the timeout triggers
        (2 seconds of no button activity). We run inference on the captured
        timing signals and display the predicted letter.
        
        Args:
            signals: List of 4 timing values in microseconds.
        """
        self._log(f"Raw signals: {signals}")
        
        # Run neural network inference on the captured signals
        result = self._inference_engine.predict(signals)
        
        # Display the prediction result
        print()
        display_result(result)
        
        # Append to message if it's a valid letter (not unclassified)
        if result.letter != 'U':
            self._decoded_message.append(result.letter)
            print(f"Message so far: {''.join(self._decoded_message)}")
        else:
            print("(Unclassified signal - not added to message)")
        
        print()
        print("Ready for next character... (Press button to continue)")
    
    def start(self) -> None:
        """Start the MorseAI decoder."""
        print("=" * 60)
        print("  MorseAI - Morse Code AI Decoder")
        print(f"  Version: {VERSION}")
        print("=" * 60)
        print()
        
        # Load the TensorFlow Lite model and normalization parameters
        print("Loading neural network model...")
        try:
            self._inference_engine = MorseInference(
                model_path=self.model_path,
                config_path=self.config_path
            )
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            print("\nPlease ensure the model files are in place:")
            print(f"  Model: {self.model_path}")
            print(f"  Config: {self.config_path}")

            sys.exit(1)
        
        print()
        print("Initializing GPIO...")
        print(f"  Button GPIO: {self.gpio_pin} (BCM)")
        print(f"  Pull-up: Internal (enabled)")
        print(f"  Bounce filter: {BOUNCE_TIME * 1000:.0f}ms")
        print(f"  Timeout: {TIMEOUT_SECONDS}s")
        print()
        
        # Initialize GPIO capture with callback for character completion
        self._capture = MorseCapture(
            gpio_pin=self.gpio_pin,
            on_character_complete=self._on_character_complete,
            bounce_time=BOUNCE_TIME
        )
        
        self._running = True
        
        print("-" * 60)
        print("  READY TO DECODE!")
        print("-" * 60)
        print()
        print("Instructions:")
        print("  1. Press and hold the button to create a signal")
        print("  2. Short press = dot (·), Long press = dash (−)")
        print("  3. Release for 2 seconds to submit the character")
        print("  4. Press Ctrl+C to exit")
        print()
        print("Waiting for input...")
        print()
        
        # Main event loop - wait for keyboard interrupt
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()
            self.stop()
    
    def stop(self) -> None:
        """Stop the decoder and clean up resources."""
        self._running = False
        
        if self._capture:
            self._capture.close()
        
        print()
        print("=" * 60)
        print("  Session Summary")
        print("=" * 60)
        
        if self._decoded_message:
            final_message = ''.join(self._decoded_message)
            print(f"  Decoded message: {final_message}")
            print(f"  Characters: {len(self._decoded_message)}")
        else:
            print("  No characters decoded in this session.")
        
        print()
        print("Goodbye!")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MorseAI - Neural Network-powered Morse Code Decoder for Raspberry Pi 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use defaults (GPIO 17)
  python main.py --gpio 27          # Use GPIO 27 instead
  python main.py --debug            # Enable debug output
  python main.py --model custom.tflite  # Use custom model

For more information, see the README.md file.
        """
    )
    
    parser.add_argument(
        '--gpio', '-g',
        type=int,
        default=GPIO_PIN,
        help=f'GPIO pin for button (BCM numbering, default: {GPIO_PIN})'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to TFLite model file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to normalization config file (.npz)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'MorseAI v{VERSION}'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    decoder = MorseAIDecoder(
        gpio_pin=args.gpio,
        model_path=args.model,
        config_path=args.config,
        debug=args.debug
    )
    
    decoder.start()


if __name__ == "__main__":
    main()
