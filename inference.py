"""
TensorFlow Lite Inference Engine for Morse Code Classification

This module handles neural network inference using TensorFlow Lite runtime.
Designed to be lightweight and optimized for Raspberry Pi 5.

The model expects 4 timing values (in microseconds) and outputs
probabilities for 11 classes (A-J + Unclassified).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Use tflite-runtime (lightweight) instead of full TensorFlow
import tflite_runtime.interpreter as tflite


# =============================================================================
# Configuration Constants
# =============================================================================

# Model architecture (from morse_decode.h)
INPUT_SIZE = 4
OUTPUT_SIZE = 11  # 10 letters (A-J) + 1 unclassified

# Character mapping
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'U']  # U = Unclassified

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "morse_classifier.tflite"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "model" / "normalization_config.npz"


@dataclass
class InferenceResult:
    """
    Result of a Morse code inference.
    
    Attributes:
        letter: Predicted character (A-J or 'U' for unclassified).
        confidence: Probability score (0.0 to 1.0).
        inference_time_us: Inference duration in microseconds.
        all_probabilities: Raw probability distribution for all classes.
    """
    letter: str
    confidence: float
    inference_time_us: float
    all_probabilities: np.ndarray


class MorseInference:
    """
    TensorFlow Lite inference engine for Morse code classification.
    
    This class loads a .tflite model and normalization parameters,
    then provides real-time inference for Morse code timing signals.
    
    Equivalent to the run_inference() function in morse_decode.c.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        input_mean: Optional[List[float]] = None,
        input_scale: Optional[List[float]] = None,
        num_threads: int = 4
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the .tflite model file.
            config_path: Path to normalization config (.npz file).
            input_mean: Manual normalization mean values (overrides config).
            input_scale: Manual normalization scale values (overrides config).
            num_threads: Number of CPU threads for inference.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.num_threads = num_threads
        
        # Load normalization parameters
        self._load_normalization_params(input_mean, input_scale)
        
        # Load TFLite model
        self._load_model()
    
    def _load_normalization_params(
        self,
        input_mean: Optional[List[float]] = None,
        input_scale: Optional[List[float]] = None
    ) -> None:
        """
        Load normalization parameters (mean and scale).
        
        Priority:
        1. Manual parameters passed to constructor
        2. Saved .npz config file
        3. Default values (zeros mean, ones scale - no normalization)
        
        Args:
            input_mean: Optional manual mean values.
            input_scale: Optional manual scale values.
        """
        if input_mean is not None and input_scale is not None:
            # Use manually provided parameters
            self.input_mean = np.array(input_mean, dtype=np.float32)
            self.input_scale = np.array(input_scale, dtype=np.float32)
            print(f"[Inference] Using manual normalization parameters")
            
        elif self.config_path.exists():
            # Load from config file
            config = np.load(str(self.config_path))
            self.input_mean = config['mean'].astype(np.float32)
            self.input_scale = config['scale'].astype(np.float32)
            print(f"[Inference] Loaded normalization from: {self.config_path}")
            
        else:
            # Default: no normalization (identity transform)
            print(f"[Inference] Warning: No normalization config found.")
            print(f"[Inference] Using default (no normalization)")
            self.input_mean = np.zeros(INPUT_SIZE, dtype=np.float32)
            self.input_scale = np.ones(INPUT_SIZE, dtype=np.float32)
        
        # Validate dimensions
        assert len(self.input_mean) == INPUT_SIZE, \
            f"Mean size mismatch: expected {INPUT_SIZE}, got {len(self.input_mean)}"
        assert len(self.input_scale) == INPUT_SIZE, \
            f"Scale size mismatch: expected {INPUT_SIZE}, got {len(self.input_scale)}"
    
    def _load_model(self) -> None:
        """Load and initialize the TFLite model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please run the model conversion script first."
            )
        
        # Initialize TFLite interpreter
        self.interpreter = tflite.Interpreter(
            model_path=str(self.model_path),
            num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Validate model architecture
        input_shape = self.input_details[0]['shape']
        output_shape = self.output_details[0]['shape']
        
        print(f"[Inference] Model loaded: {self.model_path.name}")
        print(f"[Inference] Input shape: {input_shape}")
        print(f"[Inference] Output shape: {output_shape}")
        print(f"[Inference] Threads: {self.num_threads}")
    
    def normalize(self, signals: np.ndarray) -> np.ndarray:
        """
        Normalize input signals using StandardScaler transform.
        
        Equivalent to normalize_input() in morse_decode.c:
            normalized[i] = (input[i] - input_mean[i]) / input_scale[i]
        
        Args:
            signals: Raw timing signals in microseconds.
        
        Returns:
            Normalized signals as float32 array.
        """
        signals = np.array(signals, dtype=np.float32)
        return (signals - self.input_mean) / self.input_scale
    
    def predict_letter(
        self,
        signals: List[float],
        return_all_probs: bool = False
    ) -> Tuple[str, float]:
        """
        Predict the Morse code letter from timing signals.
        
        Args:
            signals: List of 4 timing values in microseconds.
            return_all_probs: If True, include full probability distribution.
        
        Returns:
            Tuple of (predicted_letter, confidence_score).
            If return_all_probs is True, also returns probability array.
        """
        result = self.predict(signals)
        
        if return_all_probs:
            return result.letter, result.confidence, result.all_probabilities
        return result.letter, result.confidence
    
    def predict(self, signals: List[float]) -> InferenceResult:
        """
        Run full inference and return detailed results.
        
        This is the main inference function, equivalent to run_inference()
        in morse_decode.c.
        
        Args:
            signals: List of 4 timing values in microseconds.
        
        Returns:
            InferenceResult with letter, confidence, timing, and probabilities.
        """
        import time
        
        # Ensure correct input size
        signals = list(signals)
        while len(signals) < INPUT_SIZE:
            signals.append(0.0)
        signals = signals[:INPUT_SIZE]
        
        # Start timing
        start_time = time.perf_counter_ns()
        
        # Normalize input
        normalized = self.normalize(signals)
        
        # Prepare input tensor (batch size = 1)
        input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output probabilities (softmax already applied in model)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output_data[0]
        
        # Stop timing
        inference_time_us = (time.perf_counter_ns() - start_time) / 1000.0
        
        # Find best prediction
        max_idx = np.argmax(probabilities)
        max_prob = probabilities[max_idx]
        
        # Map index to letter
        if max_idx < len(LABELS):
            letter = LABELS[max_idx]
        else:
            letter = 'U'  # Unclassified
        
        return InferenceResult(
            letter=letter,
            confidence=float(max_prob),
            inference_time_us=inference_time_us,
            all_probabilities=probabilities
        )
    
    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current normalization parameters.
        
        Returns:
            Tuple of (mean, scale) arrays.
        """
        return self.input_mean.copy(), self.input_scale.copy()


def display_result(result: InferenceResult) -> None:
    """
    Display inference result in a formatted way.
    
    Equivalent to displayResult() in morse_decode.c.
    
    Args:
        result: InferenceResult from prediction.
    """
    print(f"Predicted Letter: {result.letter}")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    print(f"Inference Speed: {result.inference_time_us:.1f} µs")
    print("-" * 40)


# =============================================================================
# Utility Functions
# =============================================================================

def create_normalization_config(
    mean: List[float],
    scale: List[float],
    output_path: str
) -> None:
    """
    Create a normalization config file from mean and scale values.
    
    This function helps convert the values from network_parameters.h
    into a format usable by this inference engine.
    
    Args:
        mean: List of mean values (from StandardScaler).
        scale: List of scale values (from StandardScaler).
        output_path: Path to save the .npz config file.
    """
    np.savez(
        output_path,
        mean=np.array(mean, dtype=np.float32),
        scale=np.array(scale, dtype=np.float32)
    )
    print(f"[Config] Saved normalization config to: {output_path}")


# =============================================================================
# Standalone Testing (with mock data)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("  MorseAI Inference Engine Test")
    print("=" * 50)
    print()
    
    # Check if model exists
    if not DEFAULT_MODEL_PATH.exists():
        print(f"[Error] Model not found: {DEFAULT_MODEL_PATH}")
        print()
        print("To create the model, run the conversion snippet in your")
        print("Jupyter notebook and copy the files to the 'model/' directory.")
        print()
        print("Required files:")
        print(f"  - {DEFAULT_MODEL_PATH}")
        print(f"  - {DEFAULT_CONFIG_PATH}")
        sys.exit(1)
    
    try:
        # Initialize inference engine
        engine = MorseInference()
        
        print()
        print("Test Predictions:")
        print("-" * 50)
        
        # Test signals (example timing values in microseconds)
        test_cases = [
            [0.0, 0.0, 0.0, 0.0],          # Should be 'E' (single dot)
            [100000.0, 0.0, 0.0, 0.0],     # Short signal
            [500000.0, 0.0, 0.0, 0.0],     # Longer signal
            [100000.0, 300000.0, 0.0, 0.0], # Two signals
        ]
        
        for signals in test_cases:
            result = engine.predict(signals)
            print(f"\nInput: {signals}")
            display_result(result)
        
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
