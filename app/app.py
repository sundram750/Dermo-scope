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
MODEL_PATH = BASE_DIR / "model_training" / "saved_model.h5"

# ────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    text-align: center;
}
.main-header h1 {
    color: #e94560;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: #a8b2d8;
    font-size: 1rem;
    margin: 0.5rem 0 0;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e2140, #252b50);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.06);
}
.metric-label {
    color: #7f8ccd;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.2;
}
.high-risk { color: #ef5350; }
.low-risk  { color: #66bb6a; }

/* ── Top-3 bars ── */
.prob-bar-wrapper { margin-bottom: 0.6rem; }
.prob-label       { font-size: 0.82rem; color: #c5cae9; margin-bottom: 2px; }
.prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 10px;
    border-radius: 6px;
    transition: width 0.5s ease;
}

/* ── Section headings ── */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1.2rem 0 0.8rem;
    padding-left: 6px;
    border-left: 3px solid #e94560;
}

/* ── Grad-CAM container ── */
.gradcam-note {
    background: rgba(233, 69, 96, 0.08);
    border: 1px solid rgba(233, 69, 96, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #ef9a9a;
    margin-top: 0.5rem;
}

/* ── Sidebar ── */
.risk-legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 0.85rem;
}
.risk-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* General dark override */
.stApp { background-color: #0d1117; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
    cls       = result["predicted_class"]
    conf      = result["confidence"]
    risk      = result["risk"]
    full_name = result["full_name"]
    top3      = result["top3"]
    risk_css  = "high-risk" if risk == "High" else "low-risk"
    risk_icon = "🔴" if risk == "High" else "🟢"

    # ── Metric row ──
    # ── Confidence gauge + metric cards ──
    conf_pct   = conf * 100
    # SVG arc parameters
    radius     = 52
    stroke_w   = 10
    circumference = 2 * 3.14159 * radius
    dash_offset   = circumference * (1 - conf / 1)   # filled portion
    gauge_color   = "#ef5350" if risk == "High" else "#66bb6a"

    gauge_html = f"""
    <div style="display:flex; align-items:center; justify-content:center; margin-bottom:0.8rem;">
      <!-- Circular confidence gauge -->
      <div style="text-align:center; margin-right:1.5rem;">
        <svg width="130" height="130" viewBox="0 0 130 130">
          <!-- Background track -->
          <circle cx="65" cy="65" r="{radius}" fill="none"
                  stroke="rgba(255,255,255,0.08)" stroke-width="{stroke_w}"/>
          <!-- Filled arc -->
          <circle cx="65" cy="65" r="{radius}" fill="none"
                  stroke="{gauge_color}" stroke-width="{stroke_w}"
                  stroke-dasharray="{circumference:.1f}"
                  stroke-dashoffset="{dash_offset:.1f}"
                  stroke-linecap="round"
                  transform="rotate(-90 65 65)"
                  style="transition: stroke-dashoffset 0.8s ease;"/>
          <!-- Centre text -->
          <text x="65" y="58" text-anchor="middle"
                font-size="22" font-weight="700" fill="{gauge_color}">{conf_pct:.1f}%</text>
          <text x="65" y="76" text-anchor="middle"
                font-size="11" fill="#7f8ccd" letter-spacing="1">CONFIDENCE</text>
        </svg>
      </div>

      <!-- Prediction + Risk vertical stack -->
      <div style="display:flex; flex-direction:column; gap:0.6rem; flex:1;">
        <div class="metric-card">
          <div class="metric-label">Prediction</div>
          <div class="metric-value {risk_css}">{cls.upper()}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Risk Level</div>
          <div class="metric-value {risk_css}">{risk_icon} {risk}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

    st.markdown(f"<p style='color:#7f8ccd; margin:0 0 0.6rem; font-size:0.9rem;'>🔬 <b>{full_name}</b></p>",
                unsafe_allow_html=True)

    # ── All Classes probabilities ──
    st.markdown('<div class="section-title">All Class Probabilities</div>', unsafe_allow_html=True)
    all_probs = result.get("all_probs", top3)  # fallback to top3 if missing
    for name, prob in all_probs:
        bar_color = "#ef5350" if name in HIGH_RISK_CLASSES else "#66bb6a"
        pct = round(prob * 100, 1)
        is_top = name == cls
        weight = "font-weight:700;" if is_top else ""
        border = f"border-left: 3px solid {bar_color};" if is_top else ""
        st.markdown(f"""
        <div class="prob-bar-wrapper" style="{border} padding-left:6px;">
            <div class="prob-label" style="{weight}">{name.upper()} – {CLASS_INFO[name]['full_name'][:38]}&nbsp;&nbsp;<b>{pct}%</b></div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Low Confidence Warning ──
    if conf < 0.60:
        st.warning("⚠️ **Low Confidence (*{pct}%*):** The model is not highly confident. Please ensure the image is a clear, well-lit, close-up shot of the skin lesion.".format(pct=int(conf*100)))

    # ── Grad-CAM ──
    if result.get("gradcam_overlay") is not None:
        st.markdown('<div class="section-title">🧠 Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        st.image(
            result["gradcam_overlay"],
            caption="Grad-CAM: regions influencing the prediction (red = highest activation)",
            use_container_width=True,
        )
        if risk == "High":
            st.markdown(
                '<div class="gradcam-note">⚠️ High-risk lesion detected. '
                'Red areas highlight the regions the model focuses on most. '
                'Please consult a dermatologist.</div>',
                unsafe_allow_html=True,
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
                            render_results(live_result)
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
