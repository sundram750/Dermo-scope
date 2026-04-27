"""
convert_to_tflite.py
--------------------
One-time script to convert the trained Keras .h5 model to TFLite format.

Run this once locally (where full TensorFlow is installed):
    python model_training/convert_to_tflite.py

This generates:  model_training/saved_model.tflite
"""

import pathlib
import tensorflow as tf

BASE_DIR  = pathlib.Path(__file__).resolve().parent
H5_PATH   = BASE_DIR / "saved_model.h5"
TFLITE_PATH = BASE_DIR / "saved_model.tflite"

if not H5_PATH.exists():
    raise FileNotFoundError(f"Model not found at {H5_PATH}. Train the model first.")

print(f"Loading model from: {H5_PATH}")
model = tf.keras.models.load_model(str(H5_PATH))

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # enables dynamic-range quantization
tflite_model = converter.convert()

TFLITE_PATH.write_bytes(tflite_model)
size_mb = TFLITE_PATH.stat().st_size / (1024 * 1024)
print(f"✅  Saved TFLite model to: {TFLITE_PATH}")
print(f"    Size: {size_mb:.2f} MB  (was {H5_PATH.stat().st_size / (1024*1024):.2f} MB)")
