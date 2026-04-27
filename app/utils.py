"""
utils.py
--------
Helper utilities for the Dermo-Scope Streamlit application.

Provides:
  - load_classification_model()   – Cached model loader
  - generate_gradcam()            – Grad-CAM heatmap generator
  - overlay_heatmap()             – Blend heatmap onto original image
  - process_image()               – End-to-end inference pipeline
"""

import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "akiec": {
        "full_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "risk":      "High",
        "color":     (220, 53, 69),       # Red
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
        "color":     (40, 167, 69),       # Green
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
# Model Loading (Cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Dermo-Scope model …")
def load_classification_model(model_path: str):
    """
    Load saved Keras model with Streamlit caching so it is only loaded once.

    Parameters
    ----------
    model_path : str
        Absolute path to saved_model.h5

    Returns
    -------
    tf.keras.Model
    """
    model = tf.keras.models.load_model(model_path)
    return model

# ─────────────────────────────────────────────
# Image Pre-processing
# ─────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Prepare a raw BGR/RGB numpy frame for model inference.

    Parameters
    ----------
    frame : np.ndarray
        Raw image in BGR (from OpenCV) or RGB (from PIL).

    Returns
    -------
    np.ndarray  shape (1, 224, 224, 3)  dtype float32  range [0,1]
    """
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def pil_to_array(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array."""
    return np.array(pil_image.convert("RGB"))

# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────
def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str = "Conv_1",
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a given class index.

    Parameters
    ----------
    model              : tf.keras.Model – the full classifier
    img_array          : np.ndarray     – shape (1, 224, 224, 3) pre-processed
    class_idx          : int            – predicted class index
    last_conv_layer_name : str          – name of the last conv layer in MobileNetV2

    Returns
    -------
    np.ndarray  uint8 heatmap (224, 224)  range [0, 255]
    """
    # Build a model that outputs: (last_conv_output, final_predictions)
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ],
        )
    except ValueError:
        # Fallback: search for any Conv layer near the end
        conv_layers = [
            layer.name for layer in model.layers
            if "conv" in layer.name.lower() and len(layer.output_shape) == 4
        ]
        if not conv_layers:
            return np.zeros((*IMG_SIZE,), dtype=np.uint8)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_layers[-1]).output, model.output],
        )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, class_idx]

    # Gradients of target class w.r.t. last conv feature map
    grads = tape.gradient(loss, conv_outputs)          # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]                     # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to input image size
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    return heatmap

# ─────────────────────────────────────────────
# Overlay Heatmap
# ─────────────────────────────────────────────
def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap (red colormap) on top of the original image.

    Parameters
    ----------
    original_rgb : np.ndarray  RGB image (H, W, 3)
    heatmap      : np.ndarray  uint8 heatmap (H, W)
    alpha        : float       transparency of heatmap overlay

    Returns
    -------
    np.ndarray  RGB image with heatmap overlay
    """
    # Apply red colormap
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)   # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Resize original to match (224x224 for heatmaps)
    orig_resized = cv2.resize(original_rgb, IMG_SIZE)
    superimposed = cv2.addWeighted(orig_resized, 1 - alpha, colored_rgb, alpha, 0)
    return superimposed

# ─────────────────────────────────────────────
# Main Inference Pipeline
# ─────────────────────────────────────────────
def process_image(
    model: tf.keras.Model,
    image_rgb: np.ndarray,
    enable_gradcam: bool = True,
) -> dict:
    """
    Full end-to-end inference: preprocessing → prediction → risk evaluation → Grad-CAM.

    Parameters
    ----------
    model         : tf.keras.Model
    image_rgb     : np.ndarray    RGB image (H, W, 3)
    enable_gradcam: bool          Whether to generate Grad-CAM heatmap

    Returns
    -------
    dict with keys:
        predicted_class  : str
        confidence       : float
        risk             : str   ("High" | "Low")
        risk_color       : tuple (R, G, B)
        top3             : list of (class_name, confidence) tuples
        gradcam_overlay  : np.ndarray | None
        full_name        : str
    """
    img_array = preprocess_frame(image_rgb)
    predictions = model.predict(img_array, verbose=0)[0]   # (7,)

    top_idx = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(predictions[top_idx])

    # All predictions sorted by confidence (descending)
    all_indices = np.argsort(predictions)[::-1]
    all_probs = [(CLASS_NAMES[i], float(predictions[i])) for i in all_indices]

    # Top-3 predictions (kept for backward compat)
    top3 = all_probs[:3]

    info = CLASS_INFO[top_class]
    risk = info["risk"]
    risk_color = info["color"]

    # Grad-CAM (generated for HIGH risk classes or if explicitly enabled)
    gradcam_overlay = None
    if enable_gradcam:
        heatmap = generate_gradcam(model, img_array, top_idx)
        gradcam_overlay = overlay_heatmap(image_rgb, heatmap)

    return {
        "predicted_class": top_class,
        "full_name":       info["full_name"],
        "confidence":      confidence,
        "risk":            risk,
        "risk_color":      risk_color,
        "top3":            top3,
        "all_probs":       all_probs,
        "gradcam_overlay": gradcam_overlay,
    }

# ─────────────────────────────────────────────
# WebRTC Frame Annotation
# ─────────────────────────────────────────────
def annotate_frame(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
    """
    Annotate a BGR webcam frame with prediction label and risk indicator.

    Parameters
    ----------
    frame_bgr : np.ndarray   BGR frame from WebRTC
    result    : dict         output from process_image()

    Returns
    -------
    np.ndarray  annotated BGR frame
    """
    h, w = frame_bgr.shape[:2]
    label = f"{result['predicted_class'].upper()} ({result['confidence']*100:.1f}%)"
    risk  = result["risk"]

    # Risk colour in BGR
    color_rgb = result["risk_color"]
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

    # Semi-transparent background bar at the bottom
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)

    # Prediction text
    cv2.putText(
        frame_bgr, label,
        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    # Risk indicator
    risk_label = f"Risk: {risk}"
    cv2.putText(
        frame_bgr, risk_label,
        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2,
    )
    # Coloured border
    cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), color_bgr, 4)

    return frame_bgr
