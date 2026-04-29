"""
utils.py
--------
Helper utilities for the Dermo-Scope Streamlit application.
Uses TFLite Runtime for inference – lightweight and cloud-compatible.

Provides:
  - load_classification_model()  – Cached TFLite model loader
  - process_image()              – End-to-end inference pipeline
  - pil_to_array()               – PIL → numpy helper
"""

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# TFLite Runtime Import (cloud-safe)
# ─────────────────────────────────────────────
# Inference backend – try in order of preference:
#   1. ai_edge_litert  (Google's official successor, Python 3.11 Windows ✓)
#   2. tflite_runtime  (older standalone package)
#   3. tensorflow.lite (bundled inside full TF)
# ─────────────────────────────────────────────
try:
    import ai_edge_litert.interpreter as tflite
    _TFLITE_BACKEND = "ai_edge_litert"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        _TFLITE_BACKEND = "tflite_runtime"
    except ImportError:
        try:
            import tensorflow.lite.python.interpreter as tflite
            _TFLITE_BACKEND = "tensorflow_lite"
        except ImportError:
            tflite = None
            _TFLITE_BACKEND = None

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "akiec": {
        "full_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "risk":      "High",
        "color":     (220, 53, 69),
        "description": "A pre-cancerous patch of thick, scaly, or crusty skin. It forms when skin is damaged by UV radiation.",
        "recommendation": "Consult a dermatologist immediately for evaluation and potential biopsy or removal.",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "risk":      "High",
        "color":     (220, 53, 69),
        "description": "The most common form of skin cancer. It usually appears as a slightly transparent bump on the skin.",
        "recommendation": "Requires professional medical attention. BCCs are highly treatable when caught early.",
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "risk":      "Low",
        "color":     (40, 167, 69),
        "description": "A common non-cancerous skin growth that usually appears as a brown, black or light tan growth.",
        "recommendation": "Generally harmless and requires no treatment unless it becomes irritated or bleeds.",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "risk":      "Low",
        "color":     (40, 167, 69),
        "description": "Common, benign skin growths that tend to be firm and often have a slightly darker color at the edges.",
        "recommendation": "Usually harmless. Monitor for any rapid changes in size, shape, or color.",
    },
    "mel": {
        "full_name": "Melanoma",
        "risk":      "High",
        "color":     (220, 53, 69),
        "description": "The most serious type of skin cancer. It develops in the cells (melanocytes) that produce melanin.",
        "recommendation": "URGENT: Consult a dermatologist immediately. Early detection is critical for successful treatment.",
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "risk":      "Low",
        "color":     (40, 167, 69),
        "description": "Common moles. They are benign neoplasms of pigment-forming cells.",
        "recommendation": "Generally harmless. Perform regular self-checks using the ABCDEs of melanoma.",
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "risk":      "Low",
        "color":     (40, 167, 69),
        "description": "A relatively common abnormality of the skin and underlying tissues, more commonly known as a birthmark.",
        "recommendation": "Usually benign. If it bleeds frequently or changes rapidly, consult a doctor.",
    },
}

HIGH_RISK_CLASSES = {"mel", "bcc", "akiec"}
# Alphabetical order – must match the order used in flow_from_directory
CLASS_NAMES = sorted(CLASS_INFO.keys())   # ['akiec','bcc','bkl','df','mel','nv','vasc']

# ─────────────────────────────────────────────
# Model Loading (Cached) – TFLite Interpreter
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Dermo-Scope model ...")
def load_classification_model(model_path: str):
    """
    Load a .tflite model via TFLite Interpreter with Streamlit caching.

    Parameters
    ----------
    model_path : str
        Absolute path to saved_model.tflite

    Returns
    -------
    tflite.Interpreter (allocated and ready for inference)
    """
    if tflite is None:
        raise RuntimeError("No inference backend found. Install tflite-runtime.")
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter

# ─────────────────────────────────────────────
# Image Pre-processing
# ─────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a raw RGB numpy frame for TFLite inference.

    Returns
    -------
    np.ndarray  shape (1, 224, 224, 3)  dtype float32  range [0, 1]
    """
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def pil_to_array(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array."""
    return np.array(pil_image.convert("RGB"))

# ─────────────────────────────────────────────
# Overlay Heatmap (kept for future use)
# ─────────────────────────────────────────────
def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a heatmap (uint8) on top of the original image."""
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    orig_resized = cv2.resize(original_rgb, IMG_SIZE)
    return cv2.addWeighted(orig_resized, 1 - alpha, colored_rgb, alpha, 0)

# ─────────────────────────────────────────────
# Main Inference Pipeline (TFLite)
# ─────────────────────────────────────────────
def process_image(
    interpreter,
    image_rgb: np.ndarray,
    enable_gradcam: bool = False,   # TFLite does not support Grad-CAM
) -> dict:
    """
    Full end-to-end inference using TFLite Interpreter.

    Parameters
    ----------
    interpreter   : tflite.Interpreter  (pre-loaded, allocated)
    image_rgb     : np.ndarray           RGB image (H, W, 3)
    enable_gradcam: bool                 Ignored – TFLite does not support gradients

    Returns
    -------
    dict with keys: predicted_class, full_name, confidence, risk,
                    risk_color, top3, all_probs, gradcam_overlay
    """
    img_array = preprocess_frame(image_rgb)

    # ── TFLite inference ──
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # (7,)

    top_idx   = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(predictions[top_idx])

    # All predictions sorted by confidence (descending)
    all_indices = np.argsort(predictions)[::-1]
    all_probs   = [(CLASS_NAMES[i], float(predictions[i])) for i in all_indices]
    top3        = all_probs[:3]

    info = CLASS_INFO[top_class]

    return {
        "predicted_class": top_class,
        "full_name":       info["full_name"],
        "confidence":      confidence,
        "risk":            info["risk"],
        "risk_color":      info["color"],
        "top3":            top3,
        "all_probs":       all_probs,
        "gradcam_overlay": None,   # Not supported with TFLite
    }


