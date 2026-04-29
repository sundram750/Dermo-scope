"""
app.py
------
Dermo-Scope – Real-Time Skin Disease Detection
Streamlit web application with:
    • Native webcam capture via st.camera_input (zero extra dependencies, no lag)
    • Image upload inference
    • All 7 class probabilities with animated bars
    • Confidence score displayed prominently
    • Grad-CAM toggle (shown when model supports it)

Run:
    streamlit run app/app.py
"""

import time
import numpy as np
import cv2
import streamlit as st
from pathlib import Path
from PIL import Image

# ── Streamlit page config (MUST be first st call) ──
st.set_page_config(
    page_title="Dermo-Scope | Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ──
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_components import (
    render_metric_cards,
    render_probability_bars,
    render_dynamic_insights,
    get_downloadable_report,
)
from utils import (
    load_classification_model,
    process_image,
    pil_to_array,
    CLASS_INFO,
    CLASS_NAMES,
    HIGH_RISK_CLASSES,
)

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_training" / "saved_model.tflite"

# ────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────
with open(BASE_DIR / "app" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────
st.markdown("""
<div class="ds-header">
    <h1>🔬 Dermo-Scope</h1>
    <div class="ds-subtitle">
        Real-Time Skin Disease Detection &nbsp;·&nbsp;
        MobileNetV2 + TFLite &nbsp;·&nbsp;
        HAM10000 Dataset
    </div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    input_mode = st.radio(
        "Input Mode",
        ["📷 Live Camera", "🖼️ Upload Image"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🚦 Risk Legend")
    st.markdown("""
    <div class="risk-legend-item">
        <div class="risk-dot" style="background:#fc8181;"></div>
        <span><b>High Risk</b> — mel, bcc, akiec</span>
    </div>
    <div class="risk-legend-item">
        <div class="risk-dot" style="background:#68d391;"></div>
        <span><b>Low Risk</b> — nv, bkl, df, vasc</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 All Classes")
    for cls, info in CLASS_INFO.items():
        dot_color = "#fc8181" if info["risk"] == "High" else "#68d391"
        st.markdown(
            f'<div class="risk-legend-item">'
            f'<div class="risk-dot" style="background:{dot_color};"></div>'
            f'<span><b>{cls.upper()}</b> – {info["full_name"]}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🔍 ABCDEs of Melanoma")
    st.caption("Early detection markers:")
    st.markdown("""
    - **A**symmetry: One half doesn't match.
    - **B**order: Ragged or blurred edges.
    - **C**olor: Uneven shades of brown/black.
    - **D**iameter: Larger than 6 mm.
    - **E**volving: Changes over time.
    """)

    st.markdown("---")
    st.caption("Dermo-Scope v2.0 · HAM10000 Dataset")
    st.caption("⚠️ Educational use only – not a medical tool.")

# ────────────────────────────────────────────────────────────
# Load Model
# ────────────────────────────────────────────────────────────
model_loaded = False
model        = None

if MODEL_PATH.exists():
    try:
        model        = load_classification_model(str(MODEL_PATH))
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning(
        f"⚠️ **Model not found** at `{MODEL_PATH}`.\n\n"
        "Run `python model_training/02_train_model.py` first.\n\n"
        "The UI is in **demo mode** – predictions are random placeholders."
    )

# ────────────────────────────────────────────────────────────
# Demo-mode random prediction (when model not loaded)
# ────────────────────────────────────────────────────────────
def demo_prediction(image_rgb: np.ndarray) -> dict:
    """Return a random plausible prediction dict for UI demonstration."""
    import random
    cls  = random.choice(CLASS_NAMES)
    conf = float(np.random.uniform(0.55, 0.92))

    others   = [c for c in CLASS_NAMES if c != cls]
    remaining = 1.0 - conf
    splits   = sorted(np.random.uniform(0, 1, len(others) - 1).tolist())
    splits   = [0.0] + splits + [1.0]
    scores   = [remaining * (splits[i + 1] - splits[i]) for i in range(len(others))]

    all_probs = sorted(
        [(cls, conf)] + list(zip(others, scores)),
        key=lambda x: x[1], reverse=True,
    )
    info = CLASS_INFO[cls]
    return {
        "predicted_class": cls,
        "full_name":       info["full_name"],
        "confidence":      conf,
        "risk":            info["risk"],
        "risk_color":      info["color"],
        "top3":            all_probs[:3],
        "all_probs":       all_probs,
        "gradcam_overlay": None,
    }

# ────────────────────────────────────────────────────────────
# Shared result renderer
# ────────────────────────────────────────────────────────────
def render_results(result: dict, key_suffix: str = ""):
    """Display the full analysis panel for a given result dict."""
    render_metric_cards(result)
    render_dynamic_insights(result)

    st.markdown('<div class="section-title">All Class Probabilities</div>', unsafe_allow_html=True)
    render_probability_bars(result)

    if result.get("gradcam_overlay") is not None:
        st.markdown('<div class="section-title">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        st.image(
            result["gradcam_overlay"],
            caption="Regions influencing the prediction (red = highest activation)",
            use_container_width=True,
        )
        if result["risk"] == "High":
            st.markdown(
                '<div class="gradcam-note">⚠️ High-risk lesion detected. '
                'Red areas highlight the regions the model focuses on. '
                'Please consult a dermatologist.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    report_md = get_downloadable_report(result)
    st.download_button(
        label="📄 Download Full Analysis Report",
        data=report_md,
        file_name="dermo_scope_report.md",
        mime="text/markdown",
        use_container_width=True,
        key=f"dl_report_{key_suffix}",
    )

# ════════════════════════════════════════════════════════════
# Mode A – Upload Image
# ════════════════════════════════════════════════════════════
if input_mode == "🖼️ Upload Image":
    st.markdown('<div class="section-title">Upload a Skin Lesion Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a dermoscopy or skin lesion image",
        type=["jpg", "jpeg", "png"],
        help="Supported: JPG, JPEG, PNG",
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        img_rgb = pil_to_array(pil_img)

        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
            st.image(img_rgb, caption="Uploaded image", use_container_width=True)

        with col_res:
            st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
            with st.spinner("Analyzing …"):
                result = (
                    process_image(model, img_rgb)
                    if model_loaded
                    else demo_prediction(img_rgb)
                )
            render_results(result, key_suffix="upload")
    else:
        st.info("👆 Upload a skin lesion image to begin analysis.")

# ════════════════════════════════════════════════════════════
# Mode B – Live Camera  (st.camera_input – no WebRTC, no lag)
# ════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-title">Live Camera Capture</div>', unsafe_allow_html=True)

    col_cam, col_res = st.columns([1, 1], gap="large")

    with col_cam:
        st.markdown(
            '<p style="color:#718096;font-size:0.85rem;margin-bottom:0.5rem;">'
            '📷 Point the camera at a skin lesion and click the shutter button. '
            'Each new photo is analysed instantly.</p>',
            unsafe_allow_html=True,
        )
        camera_photo = st.camera_input(
            label="Take a photo",
            key="cam_capture",
            help="Click the camera icon to capture. Results appear instantly on the right.",
        )

    with col_res:
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

        if camera_photo is not None:
            # Decode the JPEG bytes from st.camera_input
            file_bytes = np.frombuffer(camera_photo.getvalue(), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            with st.spinner("Analysing …"):
                result = (
                    process_image(model, img_rgb)
                    if model_loaded
                    else demo_prediction(img_rgb)
                )
            render_results(result, key_suffix="cam")
        else:
            st.markdown("""
            <div style="
                background: rgba(13,20,40,0.5);
                border: 1px dashed rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 2.5rem 2rem;
                text-align: center;
                color: #4a5568;
                font-size: 0.9rem;
            ">
                📷 Waiting for camera capture…<br>
                <span style="font-size:0.78rem;">
                    Click the shutter button on the left to capture a photo.
                </span>
            </div>
            """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#2d3748; font-size:0.78rem; padding:0.8rem 0;'>
        🔬 <b>Dermo-Scope v2.0</b> &nbsp;·&nbsp; TFLite + Streamlit &nbsp;·&nbsp;
        HAM10000 Dataset (ISIC Archive) &nbsp;·&nbsp;
        <i>Educational only – not a medical diagnostic tool.</i>
    </div>
    """,
    unsafe_allow_html=True,
)
