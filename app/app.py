"""
main.py
-------
Dermo-Scope – Real-Time Skin Disease Detection System
Streamlit web application with:
    • Live webcam analysis via WebRTC
    • Image upload inference
    • Grad-CAM explainability
    • Risk level indicators

Run:
    streamlit run app/main.py
"""

import time
import numpy as np
import cv2
import streamlit as st
from pathlib import Path
from PIL import Image
import av

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
    render_plotly_probabilities,
    render_dynamic_insights,
    get_downloadable_report
)
from utils import (
    load_classification_model,
    process_image,
    pil_to_array,
    annotate_frame,
    CLASS_INFO,
    CLASS_NAMES,
    HIGH_RISK_CLASSES,
)

# ────────────────────────────────────────────────────────────
# WebRTC import (optional – graceful degradation if not installed)
# ────────────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_training" / "saved_model.tflite"

# ────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────
with open(BASE_DIR / "app" / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Dermo-Scope</h1>
    <p>Real-Time Skin Disease Detection · Powered by MobileNetV2 + Grad-CAM Explainability</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    input_mode = st.radio(
        "Input Mode",
        ["📷 Live Webcam", "🖼️ Upload Image"],
        index=0,
    )

    st.markdown("---")
    enable_gradcam = st.toggle("🧠 Grad-CAM Explainability", value=True)

    st.markdown("---")
    st.markdown("### 🚦 Risk Legend")
    st.markdown("""
    <div class="risk-legend-item">
        <div class="risk-dot" style="background:#ef5350;"></div>
        <span><b>High Risk</b> — mel, bcc, akiec</span>
    </div>
    <div class="risk-legend-item">
        <div class="risk-dot" style="background:#66bb6a;"></div>
        <span><b>Low Risk</b> — nv, bkl, df, vasc</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Classes")
    for cls, info in CLASS_INFO.items():
        dot_color = "#ef5350" if info["risk"] == "High" else "#66bb6a"
        st.markdown(
            f'<div class="risk-legend-item">'
            f'<div class="risk-dot" style="background:{dot_color};"></div>'
            f'<span><b>{cls}</b> – {info["full_name"]}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🔍 The ABCDEs of Melanoma")
    st.caption("Early detection markers for irregular lesions:")
    st.markdown("""
    - **A**symmetry: One half doesn't match the other.
    - **B**order: Edges are ragged or blurred.
    - **C**olor: Uneven shades of brown, black, or red.
    - **D**iameter: Larger than 6mm (pencil eraser).
    - **E**volving: Changes in size, shape, or color.
    """)

    st.markdown("---")
    st.caption("Dermo-Scope v1.0 · HAM10000 Dataset")

# ────────────────────────────────────────────────────────────
# Load Model
# ────────────────────────────────────────────────────────────
model_loaded = False
model = None
if MODEL_PATH.exists():
    try:
        model = load_classification_model(str(MODEL_PATH))
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning(
        f"⚠️ **Model not found** at `{MODEL_PATH}`.\n\n"
        "Please run `python model_training/02_train_model.py` first to train and save the model.\n\n"
        "You can still explore the UI in **demo mode** (predictions will be random placeholders)."
    )

# ────────────────────────────────────────────────────────────
# Helper – Render Results Panel
# ────────────────────────────────────────────────────────────
def render_results(result: dict):
    """Display prediction metrics, probability bars, and Grad-CAM."""
    conf = result["confidence"]
    
    # Render main metric cards (Prediction and Risk Level)
    render_metric_cards(result)

    # Render dynamic insights box
    render_dynamic_insights(result)

    # Use tabs for detailed visualization
    tab1, tab2 = st.tabs(["📊 Interactive Probabilities", "🧠 Grad-CAM Heatmap"])
    
    with tab1:
        render_plotly_probabilities(result)
        # ── Low Confidence Warning ──
        if conf < 0.60:
            st.warning("⚠️ **Low Confidence (*{pct}%*):** The model is not highly confident. Please ensure the image is a clear, well-lit, close-up shot of the skin lesion.".format(pct=int(conf*100)))

    with tab2:
        if result.get("gradcam_overlay") is not None:
            st.image(
                result["gradcam_overlay"],
                caption="Grad-CAM: regions influencing the prediction (red = highest activation)",
                use_container_width=True,
            )
            if result["risk"] == "High":
                st.markdown(
                    '<div class="gradcam-note">⚠️ High-risk lesion detected. '
                    'Red areas highlight the regions the model focuses on most. '
                    'Please consult a dermatologist.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Grad-CAM is not enabled or not available for this prediction.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Download Report Button
    report_md = get_downloadable_report(result)
    st.download_button(
        label="📄 Download Full Analysis Report",
        data=report_md,
        file_name="dermo_scope_report.md",
        mime="text/markdown",
        use_container_width=True
    )

# ────────────────────────────────────────────────────────────
# Demo-mode random prediction (when model not loaded)
# ────────────────────────────────────────────────────────────
def demo_prediction(image_rgb: np.ndarray) -> dict:
    """Return a random prediction dict for UI demonstration."""
    import random
    cls = random.choice(CLASS_NAMES)
    conf = float(np.random.uniform(0.55, 0.92))
    others = [c for c in CLASS_NAMES if c != cls]
    random.shuffle(others)
    # Build all 7 class scores
    remaining = 1 - conf
    splits = sorted([float(np.random.uniform(0, 1)) for _ in range(len(others) - 1)])
    splits = [0] + splits + [1]
    other_scores = [remaining * (splits[i+1] - splits[i]) for i in range(len(others))]
    all_probs = sorted(
        [(cls, conf)] + [(others[i], other_scores[i]) for i in range(len(others))],
        key=lambda x: x[1], reverse=True
    )
    top3 = all_probs[:3]
    info = CLASS_INFO[cls]
    return {
        "predicted_class": cls,
        "full_name":       info["full_name"],
        "confidence":      conf,
        "risk":            info["risk"],
        "risk_color":      info["color"],
        "top3":            top3,
        "all_probs":       all_probs,
        "gradcam_overlay": None,
    }

# ────────────────────────────────────────────────────────────
# Mode A – Image Upload
# ────────────────────────────────────────────────────────────
if input_mode == "🖼️ Upload Image":
    st.markdown('<div class="section-title">Upload a Skin Lesion Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a dermoscopy or skin lesion image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
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
            with st.spinner("Analyzing image …"):
                if model_loaded:
                    result = process_image(model, img_rgb, enable_gradcam=enable_gradcam)
                else:
                    result = demo_prediction(img_rgb)
                time.sleep(0.3)   # slight delay for UX feel
            render_results(result)
    else:
        st.info("👆 Please upload a skin lesion image to begin analysis.")

# ────────────────────────────────────────────────────────────
# Mode B – Live Webcam (WebRTC)
# ────────────────────────────────────────────────────────────
else:
    if not WEBRTC_AVAILABLE:
        st.error(
            "❌ `streamlit-webrtc` is not installed. "
            "Run `pip install streamlit-webrtc` and restart the app."
        )
    else:
        st.markdown('<div class="section-title">Live Webcam Analysis</div>', unsafe_allow_html=True)

        # Shared state for live results
        if "live_result" not in st.session_state:
            st.session_state.live_result = None

        # ── WebRTC Video Processor ──
        class SkinAnalysisProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.enable_gradcam = enable_gradcam
                self._frame_count = 0
                self._result = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img_bgr = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Process every 5th frame to reduce latency
                self._frame_count += 1
                if self._frame_count % 5 == 0:
                    if self.model is not None:
                        self._result = process_image(
                            self.model, img_rgb, enable_gradcam=self.enable_gradcam
                        )
                    else:
                        self._result = demo_prediction(img_rgb)
                    # Push to session state (best-effort)
                    try:
                        st.session_state.live_result = self._result
                    except Exception:
                        pass

                # Annotate frame if result available
                if self._result:
                    img_bgr = annotate_frame(img_bgr, self._result)

                return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        RTC_CONFIG = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        col_cam, col_res = st.columns([1.2, 1], gap="large")

        with col_cam:
            ctx = webrtc_streamer(
                key="dermo-scope-webcam",
                video_processor_factory=SkinAnalysisProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with col_res:
            st.markdown('<div class="section-title">Live Analysis Results</div>', unsafe_allow_html=True)
            result_placeholder = st.empty()

            if ctx.state.playing:
                st.success("🟢 Webcam active – analyzing in real-time")
                # Continuously refresh results while webcam is active
                while ctx.state.playing:
                    live_result = st.session_state.get("live_result")
                    if live_result:
                        with result_placeholder.container():
                            render_metric_cards(live_result)
                            render_dynamic_insights(live_result)
                    time.sleep(0.5)
            else:
                result_placeholder.info(
                    "📷 Click **START** above to activate your webcam.\n\n"
                    "The system will analyze your skin in real-time and display predictions here."
                )

# ────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#4a5568; font-size:0.8rem; padding:1rem 0;'>
        🔬 <b>Dermo-Scope</b> · Built with TensorFlow &amp; Streamlit ·
        HAM10000 Dataset (ISIC Archive) ·
        <i>For educational purposes only – not a medical diagnostic tool.</i>
    </div>
    """,
    unsafe_allow_html=True,
)
