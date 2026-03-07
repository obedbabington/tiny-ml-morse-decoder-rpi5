"""
TensorFlow Lite Inference Engine for Morse Code Classification

Handles neural network inference using TensorFlow Lite runtime, optimized
for edge devices like Raspberry Pi.

The model expects 4 timing values (in microseconds) and outputs
probabilities for 11 classes (A-J + Unclassified).
"""

import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Use tflite-runtime (lightweight) instead of full TensorFlow
# This reduces memory footprint and startup time on embedded devices
import tflite_runtime.interpreter as tflite

# Model architecture constants
INPUT_SIZE = 4   # 4 timing values per Morse code character
OUTPUT_SIZE = 11  # 10 letters (A-J) + 1 unclassified

# Character mapping - output classes from the neural network
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'U']  # U = Unclassified

# Default model paths
DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "morse_classifier.tflite"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "model" / "normalization_config.npz"


@dataclass
class InferenceResult:
    """
    Result of a Morse code inference prediction.
    
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
    
    Loads a .tflite model and normalization parameters, then provides
    real-time inference for Morse code timing signals.
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
            num_threads: Number of CPU threads for inference (default: 4).
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.num_threads = num_threads
        
        # Load normalization parameters (required for input preprocessing)
        self._load_normalization_params(input_mean, input_scale)
        
        # Load and initialize the TFLite model
        self._load_model()
    
    def _load_normalization_params(
        self,
        input_mean: Optional[List[float]] = None,
        input_scale: Optional[List[float]] = None
    ) -> None:
        """
        Load normalization parameters (mean and scale).
        
        Normalization ensures input data matches the distribution the model
        was trained on. Priority: manual params > config file > defaults.
        
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
            # Load from saved config file (created during model conversion)
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
        
        # Validate dimensions match expected input size
        assert len(self.input_mean) == INPUT_SIZE, \
            f"Mean size mismatch: expected {INPUT_SIZE}, got {len(self.input_mean)}"
        assert len(self.input_scale) == INPUT_SIZE, \
            f"Scale size mismatch: expected {INPUT_SIZE}, got {len(self.input_scale)}"
    
    def _load_model(self) -> None:
        """Load and initialize the TFLite model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please ensure the model files are in place."
            )
        
        # Initialize TFLite interpreter with thread configuration
        # The interpreter manages model execution and memory allocation
        self.interpreter = tflite.Interpreter(
            model_path=str(self.model_path),
            num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output tensor details for validation
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Validate model architecture matches expectations
        input_shape = self.input_details[0]['shape']
        output_shape = self.output_details[0]['shape']
        
        print(f"[Inference] Model loaded: {self.model_path.name}")
        print(f"[Inference] Input shape: {input_shape}")
        print(f"[Inference] Output shape: {output_shape}")
        print(f"[Inference] Threads: {self.num_threads}")
    
    def normalize(self, signals: np.ndarray) -> np.ndarray:
        """
        Normalize input signals using StandardScaler transform.
        
        Formula: normalized[i] = (input[i] - mean[i]) / scale[i]
        This ensures input data matches the training distribution.
        
        Args:
            signals: Raw timing signals in microseconds.
        
        Returns:
            Normalized signals as float32 array.
        """
        signals = np.array(signals, dtype=np.float32)
        return (signals - self.input_mean) / self.input_scale
    
    def predict(self, signals: List[float]) -> InferenceResult:
        """
        Run inference on timing signals and return prediction result.
        
        This is the main inference function. It normalizes the input,
        runs the neural network, and returns the predicted letter with
        confidence score and timing information.
        
        Args:
            signals: List of 4 timing values in microseconds.
        
        Returns:
            InferenceResult with letter, confidence, timing, and probabilities.
        """
        # Ensure input is exactly 4 elements
        signals = list(signals)
        if len(signals) != INPUT_SIZE:
            raise ValueError(
                f"Expected {INPUT_SIZE} timing values, got {len(signals)}. "
                f"Signals should be padded to {INPUT_SIZE} elements."
            )
        
        # Measure inference time for performance monitoring
        start_time = time.perf_counter_ns()
        
        # Normalize input to match training distribution
        normalized = self.normalize(signals)
        
        # Prepare input tensor (add batch dimension: shape becomes [1, 4])
        input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
        
        # Run inference: set input, invoke model, get output
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output probabilities (softmax already applied in model)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output_data[0]
        
        # Calculate inference time in microseconds
        inference_time_us = (time.perf_counter_ns() - start_time) / 1000.0
        
        # Find the class with highest probability
        max_idx = np.argmax(probabilities)
        max_prob = probabilities[max_idx]
        
        # Map index to letter label
        if max_idx < len(LABELS):
            letter = LABELS[max_idx]
        else:
            letter = 'U'  # Unclassified (safety fallback)
        
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
    
    Args:
        result: InferenceResult from prediction.
    """
    print(f"Predicted Letter: {result.letter}")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    print(f"Inference Speed: {result.inference_time_us:.1f} µs")
    print("-" * 40)


# =============================================================================
# Standalone Testing
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
        
        # Test cases with known Morse code patterns (timing in microseconds)
        # Format: [signal1, signal2, signal3, signal4]
        test_cases = [
            # E: · (single dot)
            ([150000.0, 0.0, 0.0, 0.0], 'E'),
            # I: ·· (two dots)
            ([120000.0, 130000.0, 0.0, 0.0], 'I'),
            # A: ·− (dot, dash)
            ([140000.0, 350000.0, 0.0, 0.0], 'A'),
            # H: ···· (four dots)
            ([110000.0, 120000.0, 130000.0, 140000.0], 'H'),
            # D: −·· (dash, dot, dot)
            ([320000.0, 150000.0, 160000.0, 0.0], 'D'),
            # B: −··· (dash, dot, dot, dot)
            ([340000.0, 140000.0, 150000.0, 160000.0], 'B'),
        ]
        
        for signals, expected_label in test_cases:
            result = engine.predict(signals)
            match_indicator = "✓" if result.letter == expected_label else "✗"
            print(f"\nInput: {signals}")
            print(f"Expected: {expected_label} {match_indicator}")
            display_result(result)
        
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
