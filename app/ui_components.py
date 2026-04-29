# app/ui_components.py
# ─────────────────────────────────────────────────────────────
# Pure-Python / HTML+CSS UI components – zero external charting libs.
# Works without plotly, pandas, seaborn, or matplotlib.
# ─────────────────────────────────────────────────────────────

import streamlit as st
from utils import CLASS_INFO, HIGH_RISK_CLASSES


# ──────────────────────────────────────────────────────────────
# 1 · THREE METRIC CARDS  (Prediction | Confidence | Risk)
# ──────────────────────────────────────────────────────────────
def render_metric_cards(result: dict):
    """
    Render three side-by-side glassmorphism cards:
      • Predicted class (short code + full name)
      • Confidence percentage
      • Risk level with animated glow
    """
    cls        = result["predicted_class"]
    full_name  = result.get("full_name", CLASS_INFO[cls]["full_name"])
    conf_pct   = result["confidence"] * 100
    risk       = result["risk"]
    risk_icon  = "🔴" if risk == "High" else "🟢"
    risk_class = "high-risk-val" if risk == "High" else "low-risk-val"

    st.markdown(f"""
    <div class="cards-row">
        <div class="ds-card" style="animation-delay:0.05s">
            <div class="ds-card-label">Prediction</div>
            <div class="ds-card-value class-val">{cls.upper()}</div>
            <div class="ds-card-sub">{full_name}</div>
        </div>
        <div class="ds-card" style="animation-delay:0.12s">
            <div class="ds-card-label">Confidence</div>
            <div class="ds-card-value conf-val">{conf_pct:.1f}%</div>
            <div class="ds-card-sub">Model certainty</div>
        </div>
        <div class="ds-card" style="animation-delay:0.19s">
            <div class="ds-card-label">Risk Level</div>
            <div class="ds-card-value {risk_class}">{risk_icon} {risk}</div>
            <div class="ds-card-sub">{"Consult a doctor" if risk == "High" else "Monitor regularly"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 2 · ALL-CLASS ANIMATED PROBABILITY BARS  (pure HTML/CSS)
# ──────────────────────────────────────────────────────────────
def render_probability_bars(result: dict):
    """
    Render animated horizontal bars for ALL 7 skin-disease classes.
    The top prediction bar glows; high-risk classes use red, low-risk green.
    No external charting library required.
    """
    all_probs = result.get("all_probs", [])
    if not all_probs:
        st.info("No probability data available.")
        return

    top_class = result["predicted_class"]

    rows_html = ""
    for i, (cls, prob) in enumerate(all_probs):
        pct        = prob * 100
        is_top     = cls == top_class
        is_high    = cls in HIGH_RISK_CLASSES
        bar_class  = "high-bar" if is_high else "low-bar"
        label_cls  = "top-label" if is_top else ""
        pct_cls    = "top-pct"  if is_top else ""
        delay      = f"{0.05 * i:.2f}s"
        full_name  = CLASS_INFO[cls]["full_name"]

        # Top prediction: override bar colour to blue gradient
        if is_top:
            bar_style = (
                f"width:{pct:.1f}%; "
                "background: linear-gradient(90deg, #63b3ed, #4299e1); "
                "box-shadow: 0 0 12px rgba(99,179,237,0.5); "
                f"animation: barGrow 0.8s cubic-bezier(0.16,1,0.3,1) {delay} both; "
                "transform-origin: left;"
            )
        else:
            bar_style = (
                f"width:{pct:.1f}%; "
                f"animation: barGrow 0.6s cubic-bezier(0.16,1,0.3,1) {delay} both; "
                "transform-origin: left;"
            )

        rows_html += f"""
        <div class="prob-row">
            <span class="prob-label {label_cls}">{cls.upper()}</span>
            <div class="prob-track">
                <div class="prob-fill {bar_class}" style="{bar_style}"></div>
            </div>
            <span class="prob-pct {pct_cls}">{pct:.1f}%</span>
            <span class="prob-name">{full_name}</span>
        </div>"""

    st.markdown(f"""
    <div class="prob-section">
        <div class="prob-section-title">All Class Probabilities</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

    # Low-confidence warning
    if result["confidence"] < 0.60:
        st.markdown(
            f'<div class="low-conf-warn">⚠️ <b>Low Confidence ({result["confidence"]*100:.0f}%)</b> – '
            'Ensure the image is a clear, well-lit, close-up shot of the skin lesion.</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# 3 · DYNAMIC INSIGHT BOX
# ──────────────────────────────────────────────────────────────
def render_dynamic_insights(result: dict):
    """
    Educational insight card: description + recommendation + top-3 ranking.
    """
    cls    = result["predicted_class"]
    info   = CLASS_INFO[cls]
    risk   = info["risk"]
    icon   = "⚠️" if risk == "High" else "ℹ️"
    box_cls = "" if risk == "High" else " low"

    # Top-3 mini list
    top3       = result.get("top3", [])
    top3_html  = ""
    if top3:
        items = "".join(
            f'<span style="margin-right:1.2rem; font-size:0.8rem; color:#718096;">'
            f'<b style="color:#a0aec0">#{i+1}</b> {c.upper()} '
            f'<span style="color:#4a5568">({p*100:.0f}%)</span></span>'
            for i, (c, p) in enumerate(top3)
        )
        top3_html = f'<div style="margin-top:0.8rem">{items}</div>'

    st.markdown(f"""
    <div class="insight-box{box_cls}">
        <div class="insight-title">{icon} {info['full_name']}</div>
        <div class="insight-desc">{info.get('description', '')}</div>
        <div class="insight-rec"><b>Recommendation:</b> {info.get('recommendation', 'Consult a doctor if unsure.')}</div>
        {top3_html}
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 4 · DOWNLOADABLE REPORT
# ──────────────────────────────────────────────────────────────
def get_downloadable_report(result: dict) -> str:
    """Generate a plain-Markdown analysis report for download."""
    cls  = result["predicted_class"]
    info = CLASS_INFO[cls]

    all_probs_lines = "\n".join(
        f"| {c.upper()} | {CLASS_INFO[c]['full_name']} | {p*100:.2f}% | {CLASS_INFO[c]['risk']} |"
        for c, p in result.get("all_probs", [])
    )

    return f"""# Dermo-Scope Analysis Report

## Prediction Summary
| Field | Value |
|---|---|
| **Predicted Class** | {info['full_name']} ({cls.upper()}) |
| **Confidence** | {result['confidence'] * 100:.1f}% |
| **Risk Level** | {info['risk']} |

## Disease Information
{info.get('description', '')}

## Clinical Recommendation
{info.get('recommendation', '')}

## All Class Probabilities
| Class | Full Name | Probability | Risk |
|---|---|---|---|
{all_probs_lines}

---
*Disclaimer: Dermo-Scope is an educational tool, not a medical diagnostic device.
Always consult a qualified dermatologist for medical advice.*
"""
