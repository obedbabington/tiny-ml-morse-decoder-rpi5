"""
Morse Code Signal Capture Module for Raspberry Pi 5

Captures timing signals from a physical button to decode Morse code.
Uses GPIO interrupts and high-precision timing to measure button press durations.

Hardware: Tactile button connected to GPIO 17 (BCM) with internal pull-up resistor.
"""

import time
import threading
from typing import List, Callable, Optional
from gpiozero import Button

# Configuration constants
MAX_SIGNALS = 4           # Maximum signals per character (Morse uses up to 4)
TIMEOUT_SECONDS = 2.0     # Timeout to detect end of character
GPIO_PIN = 17             # GPIO pin for button (BCM numbering)
BOUNCE_TIME = 0.05        # Debounce time to filter contact bounce (50ms)


class MorseCapture:
    """
    Captures Morse code timing signals from a GPIO button.
    
    Measures button press durations in microseconds and triggers a callback
    when a character is complete (detected via timeout after last release).
    """
    
    def __init__(
        self,
        gpio_pin: int = GPIO_PIN,
        on_character_complete: Optional[Callable[[List[float]], None]] = None,
        bounce_time: float = BOUNCE_TIME
    ):
        """
        Initialize the Morse capture system.
        
        Args:
            gpio_pin: BCM GPIO pin number for the button.
            on_character_complete: Callback function called when character is complete.
                                  Receives a list of 4 timing values (microseconds).
            bounce_time: Debounce time in seconds to filter mechanical contact bounce.
        """
        self.gpio_pin = gpio_pin
        self.on_character_complete = on_character_complete
        self.bounce_time = bounce_time
        
        # Thread-safe storage for captured signal durations
        self._signals: List[float] = []
        self._lock = threading.RLock()  # Re-entrant lock for nested calls
        
        # Timing state
        self._press_start_time: Optional[float] = None
        self._timeout_timer: Optional[threading.Timer] = None
        
        # Initialize GPIO button with internal pull-up resistor
        # With pull-up enabled, GPIO reads HIGH when button is open,
        # and LOW when button connects GPIO to GND
        self._button = Button(
            gpio_pin,
            pull_up=True,           # Use internal pull-up resistor (no external resistor needed)
            bounce_time=bounce_time  # Filter out mechanical contact bounce
        )
        
        # Register event handlers for button press/release
        # gpiozero calls these from a background thread when GPIO state changes
        self._button.when_pressed = self._on_button_pressed
        self._button.when_released = self._on_button_released
        
        self._running = True
    
    def _get_time_us(self) -> float:
        """Get current time in microseconds using high-precision timer."""
        # perf_counter_ns() provides nanosecond precision, convert to microseconds
        return time.perf_counter_ns() / 1000.0
    
    def _on_button_pressed(self) -> None:
        """Handle button press event - start timing the press duration."""
        if not self._running:
            return
        
        # Cancel any pending timeout (user is still entering signals)
        self._cancel_timeout()
        
        # Record the start time for this press
        self._press_start_time = self._get_time_us()
    
    def _on_button_released(self) -> None:
        """Handle button release event - record duration and start timeout timer."""
        if not self._running or self._press_start_time is None:
            return
        
        # Calculate how long the button was pressed (in microseconds)
        release_time = self._get_time_us()
        duration_us = release_time - self._press_start_time
        self._press_start_time = None
        
        # Store the signal duration if we haven't exceeded the maximum
        with self._lock:
            if len(self._signals) < MAX_SIGNALS:
                self._signals.append(duration_us)
        
        # Start timeout timer - if no more presses occur within TIMEOUT_SECONDS,
        # we'll assume the character is complete
        self._start_timeout()
    
    def _start_timeout(self) -> None:
        """Start the timeout timer to detect end of character."""
        # Cancel any existing timeout (user may have pressed again)
        self._cancel_timeout()
        
        # Create a timer that will fire after TIMEOUT_SECONDS
        # The timer runs in a daemon thread, so it won't prevent program exit
        self._timeout_timer = threading.Timer(
            TIMEOUT_SECONDS,
            self._on_timeout
        )
        self._timeout_timer.daemon = True
        self._timeout_timer.start()
    
    def _cancel_timeout(self) -> None:
        """Cancel any pending timeout timer."""
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None
    
    def _on_timeout(self) -> None:
        """
        Handle timeout event - character is complete.
        
        This is called when no button activity occurs for TIMEOUT_SECONDS.
        We package up the captured signals and trigger the callback.
        """
        if not self._running:
            return
        
        # Get the captured signals (padded to MAX_SIGNALS length)
        with self._lock:
            signals = self.get_padded_signals()
            self._signals = []  # Reset for next character
        
        # Trigger the callback if one was registered
        if self.on_character_complete is not None:
            self.on_character_complete(signals)
    
    def get_padded_signals(self) -> List[float]:
        """
        Get captured signals, padded to MAX_SIGNALS length.
        
        Returns a list of exactly MAX_SIGNALS float values (microseconds).
        Shorter sequences are padded with 0.0.
        """
        with self._lock:
            signals = self._signals.copy()
        
        # Pad with zeros if fewer than MAX_SIGNALS signals were captured
        while len(signals) < MAX_SIGNALS:
            signals.append(0.0)
        
        return signals[:MAX_SIGNALS]
    
    def close(self) -> None:
        """Clean up resources and stop capture."""
        self._running = False
        self._cancel_timeout()
        self._button.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  MorseAI Signal Capture Test")
    print("=" * 50)
    print(f"  GPIO Pin: {GPIO_PIN}")
    print(f"  Timeout: {TIMEOUT_SECONDS}s")
    print(f"  Bounce Time: {BOUNCE_TIME * 1000:.0f}ms")
    print("=" * 50)
    print()
    
    def on_char_complete(signals: List[float]) -> None:
        """Callback to display captured signals."""
        print(f"\n[Character Complete] Signals (µs): {signals}")
        print(f"  Signal 1: {signals[0]:,.1f} µs")
        print(f"  Signal 2: {signals[1]:,.1f} µs")
        print(f"  Signal 3: {signals[2]:,.1f} µs")
        print(f"  Signal 4: {signals[3]:,.1f} µs")
        print("\nReady for next character...")
    
    print("Press Ctrl+C to exit.\n")
    
    try:
        with MorseCapture(on_character_complete=on_char_complete) as capture:
            print("Capture initialized. Press the button to enter Morse code.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nCapture stopped.")
