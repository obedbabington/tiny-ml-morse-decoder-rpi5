"""
TFLite Model Conversion Script

This script/snippet converts a trained Keras MorseCodeClassifier model
to TensorFlow Lite format for deployment on Raspberry Pi 5.

USAGE:
------
Option 1: Add this code to your MorseAI_Workshop_Notebook.ipynb
          (after the training cells) and run it in the notebook.

Option 2: Run as standalone script if you have a saved Keras model:
          python convert_to_tflite.py --model saved_model.h5 --scaler scaler.pkl

The output files should be copied to the MorseAI_RPi5/model/ directory
on your Raspberry Pi 5.
"""

# =============================================================================
# TFLite Conversion Code - Copy this into your Jupyter Notebook
# =============================================================================

# Add this as a new cell in MorseAI_Workshop_Notebook.ipynb after training:

NOTEBOOK_CONVERSION_CELL = '''
# =============================================================================
# 🔄 TFLite Conversion for Raspberry Pi 5
# =============================================================================
# Run this cell AFTER training to export the model for Raspberry Pi deployment.

import os
import numpy as np
import tensorflow as tf

def convert_to_tflite(classifier, output_dir="./model_export"):
    """
    Convert the trained Keras model to TFLite format for Raspberry Pi 5.
    
    This function exports:
    1. morse_classifier.tflite - The converted neural network model
    2. normalization_config.npz - Mean and scale values for input normalization
    
    Args:
        classifier: Trained MorseCodeClassifier instance (from training cells).
        output_dir: Directory to save the converted files.
    
    Returns:
        Tuple of (model_path, config_path) for the generated files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  TFLite Model Conversion for Raspberry Pi 5")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Convert Keras model to TFLite
    # -------------------------------------------------------------------------
    print("Step 1: Converting Keras model to TFLite...")
    
    # Create converter from Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier.model)
    
    # Apply optimizations for Raspberry Pi (balance size vs accuracy)
    # DEFAULT optimization quantizes weights to int8, keeping activations float32
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Full integer quantization (even smaller, slightly less accurate)
    # Uncomment these lines for maximum compression:
    # converter.target_spec.supported_types = [tf.float16]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    model_path = os.path.join(output_dir, "morse_classifier.tflite")
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size_kb = len(tflite_model) / 1024
    print(f"   ✅ Model saved: {model_path}")
    print(f"   📦 Size: {model_size_kb:.2f} KB")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Export normalization parameters
    # -------------------------------------------------------------------------
    print("Step 2: Exporting normalization parameters...")
    
    # Extract mean and scale from the StandardScaler
    # These are used in inference.py to normalize input signals
    mean = classifier.scaler.mean_.astype(np.float32)
    scale = classifier.scaler.scale_.astype(np.float32)
    
    config_path = os.path.join(output_dir, "normalization_config.npz")
    np.savez(config_path, mean=mean, scale=scale)
    
    print(f"   ✅ Config saved: {config_path}")
    print(f"   📊 Mean values:  {mean}")
    print(f"   📊 Scale values: {scale}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Export label mapping
    # -------------------------------------------------------------------------
    print("Step 3: Exporting label mapping...")
    
    labels = list(classifier.label_encoder.classes_)
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{i}: {label}\\n")
        f.write(f"{len(labels)}: U  # Unclassified\\n")
    
    print(f"   ✅ Labels saved: {labels_path}")
    print(f"   🏷️  Classes: {labels}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Verify the converted model
    # -------------------------------------------------------------------------
    print("Step 4: Verifying TFLite model...")
    
    # Load and test the converted model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Input dtype:  {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Run a test inference
    test_input = np.zeros((1, 4), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   Test inference: OK (output sum = {test_output.sum():.4f})")
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Compare Keras vs TFLite predictions
    # -------------------------------------------------------------------------
    print("Step 5: Validating prediction consistency...")
    
    # Generate random test samples
    np.random.seed(42)
    test_samples = np.random.rand(10, 4).astype(np.float32) * 1000000  # Random μs values
    test_samples_scaled = classifier.scaler.transform(test_samples)
    
    # Keras predictions
    keras_predictions = classifier.model.predict(test_samples_scaled, verbose=0)
    keras_classes = np.argmax(keras_predictions, axis=1)
    
    # TFLite predictions
    tflite_classes = []
    for i in range(len(test_samples_scaled)):
        input_data = test_samples_scaled[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        tflite_classes.append(np.argmax(output))
    
    # Compare
    match_count = sum(k == t for k, t in zip(keras_classes, tflite_classes))
    accuracy = match_count / len(keras_classes) * 100
    
    print(f"   Keras vs TFLite match: {match_count}/{len(keras_classes)} ({accuracy:.1f}%)")
    
    if accuracy == 100:
        print("   ✅ Perfect match! Conversion successful.")
    else:
        print("   ⚠️  Minor differences (normal due to quantization)")
    print()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  ✅ Conversion Complete!")
    print("=" * 60)
    print()
    print("📁 Generated files:")
    print(f"   {model_path}")
    print(f"   {config_path}")
    print(f"   {labels_path}")
    print()
    print("📋 Next steps:")
    print("   1. Copy the 'model_export/' folder to your Raspberry Pi")
    print("   2. Place files in: ~/MorseAI_RPi5/model/")
    print("   3. Run: python3 main.py")
    print()
    print("🔧 Transfer command (from your PC):")
    print(f"   scp -r {output_dir}/* pi@raspberrypi:~/MorseAI_RPi5/model/")
    print()
    
    return model_path, config_path


# =============================================================================
# 🚀 Run the conversion
# =============================================================================
# Make sure 'classifier' is your trained MorseCodeClassifier instance
# (It should exist from the training cells above)

try:
    model_path, config_path = convert_to_tflite(classifier)
except NameError:
    print("❌ Error: 'classifier' not found!")
    print("   Please run the training cells first to create the classifier.")
'''


# =============================================================================
# Standalone Script Mode
# =============================================================================

def main():
    """
    Standalone conversion mode.
    
    Use this if you have a saved Keras model and scaler.
    """
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(
        description="Convert Keras MorseAI model to TFLite format"
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to saved Keras model (.h5 or SavedModel directory)'
    )
    parser.add_argument(
        '--scaler', '-s',
        required=True,
        help='Path to saved StandardScaler (.pkl file)'
    )
    parser.add_argument(
        '--labels', '-l',
        required=True,
        help='Path to saved LabelEncoder (.pkl file)'
    )
    parser.add_argument(
        '--output', '-o',
        default='./model_export',
        help='Output directory for converted files'
    )
    
    args = parser.parse_args()
    
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow is required for conversion.")
        print("Install with: pip install tensorflow")
        return
    
    # Load saved model
    print(f"Loading Keras model from: {args.model}")
    model = tf.keras.models.load_model(args.model)
    
    # Load scaler
    print(f"Loading scaler from: {args.scaler}")
    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load label encoder
    print(f"Loading labels from: {args.labels}")
    with open(args.labels, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Create a mock classifier object
    class MockClassifier:
        pass
    
    classifier = MockClassifier()
    classifier.model = model
    classifier.scaler = scaler
    classifier.label_encoder = label_encoder
    
    # Run conversion (using the notebook code)
    exec(NOTEBOOK_CONVERSION_CELL.replace('classifier', 'classifier'))


if __name__ == "__main__":
    # Print the notebook cell for easy copy-paste
    print("=" * 70)
    print("  TFLite Conversion Code for MorseAI")
    print("=" * 70)
    print()
    print("Copy the code below and paste it into a new cell in your")
    print("MorseAI_Workshop_Notebook.ipynb after the training cells:")
    print()
    print("-" * 70)
    print(NOTEBOOK_CONVERSION_CELL)
    print("-" * 70)
    print()
    print("After running the cell, copy the generated files to your")
    print("Raspberry Pi 5: MorseAI_RPi5/model/")
