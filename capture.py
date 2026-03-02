"""
Morse Code Signal Capture Module for Raspberry Pi 5

This module handles physical button input and captures Morse code timing signals.
Ported from the STM32 morse_encode.c implementation.

Hardware: Tactile button connected to GPIO 17 with internal pull-up resistor.
Timing: Microsecond precision using time.perf_counter_ns().
"""

import time
import threading
from typing import List, Callable, Optional
from gpiozero import Button

# =============================================================================
# Configuration Constants (Ported from morse_encode.h)
# =============================================================================

MAX_SIGNALS = 4           # Maximum number of signals (presses) per character
TIMEOUT_SECONDS = 2.0     # 2.0 second timeout to signify "End of Character"
GPIO_PIN = 17             # GPIO pin for the button (BCM numbering)
BOUNCE_TIME = 0.05        # 50ms debounce time to handle contact bounce


class MorseCapture:
    """
    Captures Morse code timing signals from a physical button.
    
    This class replicates the interrupt-driven timing logic from the STM32
    implementation using Python's threading and gpiozero.
    
    Attributes:
        gpio_pin (int): BCM GPIO pin number for the button.
        on_character_complete (Callable): Callback when character capture is done.
        signals (List[float]): Current captured signal durations in microseconds.
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
            gpio_pin: BCM GPIO pin number (default: 17).
            on_character_complete: Callback function when character is complete.
            bounce_time: Debounce time in seconds to filter contact bounce.
        """
        self.gpio_pin = gpio_pin
        self.on_character_complete = on_character_complete
        self.bounce_time = bounce_time
        
        # Signal storage
        self._signals: List[float] = []
        self._lock = threading.Lock()
        
        # Timing state
        self._press_start_time: Optional[float] = None
        self._timeout_timer: Optional[threading.Timer] = None
        
        # Initialize button with internal pull-up resistor
        # active_low logic: button connects GPIO to GND when pressed
        self._button = Button(
            gpio_pin,
            pull_up=True,           # Enable internal pull-up resistor
            bounce_time=bounce_time  # Hardware debounce filter
        )
        
        # Attach interrupt handlers
        self._button.when_pressed = self._on_button_pressed
        self._button.when_released = self._on_button_released
        
        self._running = True
    
    def _get_time_us(self) -> float:
        """Get current time in microseconds (high precision)."""
        return time.perf_counter_ns() / 1000.0
    
    def _on_button_pressed(self) -> None:
        """
        Handle button press event.
        
        Equivalent to EXTI4_15_IRQHandler when GPIOC->IDR & GPIO_IDR_13 is LOW.
        Starts timing the press duration.
        """
        if not self._running:
            return
            
        # Cancel any pending timeout
        self._cancel_timeout()
        
        # Record press start time
        self._press_start_time = self._get_time_us()
    
    def _on_button_released(self) -> None:
        """
        Handle button release event.
        
        Equivalent to EXTI4_15_IRQHandler when button is released.
        Records the press duration and starts the timeout timer.
        """
        if not self._running or self._press_start_time is None:
            return
        
        # Calculate press duration in microseconds
        release_time = self._get_time_us()
        duration_us = release_time - self._press_start_time
        self._press_start_time = None
        
        # Store signal if we haven't exceeded MAX_SIGNALS
        with self._lock:
            if len(self._signals) < MAX_SIGNALS:
                self._signals.append(duration_us)
        
        # Start timeout timer for end-of-character detection
        self._start_timeout()
    
    def _start_timeout(self) -> None:
        """
        Start the timeout timer.
        
        Equivalent to TIM2 configuration after button release:
        TIM2->ARR = TIMEOUT_US - 1
        """
        self._cancel_timeout()
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
        Handle timeout event (end of character).
        
        Equivalent to TIM2_IRQHandler setting g_sendFlag = 1.
        Triggers the callback with the captured signals.
        """
        if not self._running:
            return
            
        with self._lock:
            signals = self.get_padded_signals()
            self._signals = []  # Reset for next character
        
        # Trigger callback if registered
        if self.on_character_complete is not None:
            self.on_character_complete(signals)
    
    def get_padded_signals(self) -> List[float]:
        """
        Get the captured signals, padded to MAX_SIGNALS length.
        
        Returns:
            List of exactly MAX_SIGNALS float values (microseconds).
            Shorter sequences are padded with 0.0.
        """
        with self._lock:
            signals = self._signals.copy()
        
        # Pad with zeros if fewer than MAX_SIGNALS
        while len(signals) < MAX_SIGNALS:
            signals.append(0.0)
        
        return signals[:MAX_SIGNALS]
    
    def get_raw_signals(self) -> List[float]:
        """
        Get the raw captured signals without padding.
        
        Returns:
            List of captured signal durations in microseconds.
        """
        with self._lock:
            return self._signals.copy()
    
    def reset(self) -> None:
        """
        Reset the capture state.
        
        Equivalent to resetState() in morse_encode.c.
        """
        self._cancel_timeout()
        with self._lock:
            self._signals = []
            self._press_start_time = None
    
    def close(self) -> None:
        """Clean up resources and stop capture."""
        self._running = False
        self._cancel_timeout()
        self._button.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def capture_single_character(
    gpio_pin: int = GPIO_PIN,
    timeout: float = TIMEOUT_SECONDS,
    bounce_time: float = BOUNCE_TIME
) -> List[float]:
    """
    Capture a single Morse code character (blocking).
    
    This is a convenience function for simple single-character capture.
    
    Args:
        gpio_pin: BCM GPIO pin number.
        timeout: Timeout in seconds to end character capture.
        bounce_time: Debounce time in seconds.
    
    Returns:
        List of 4 float values representing signal durations in microseconds.
    """
    result: List[float] = []
    capture_done = threading.Event()
    
    def on_complete(signals: List[float]) -> None:
        nonlocal result
        result = signals
        capture_done.set()
    
    with MorseCapture(
        gpio_pin=gpio_pin,
        on_character_complete=on_complete,
        bounce_time=bounce_time
    ) as capture:
        print(f"Ready to capture... Press the button to enter Morse code.")
        print(f"(Release for {timeout}s to finish character)")
        capture_done.wait()
    
    return result


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
