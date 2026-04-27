# app/ui_components.py
import streamlit as st
import plotly.express as px
import pandas as pd
from utils import CLASS_INFO, HIGH_RISK_CLASSES

def render_metric_cards(result: dict):
    """Render the main metric cards (Prediction and Risk)."""
    cls = result["predicted_class"]
    risk = result["risk"]
    risk_css = "high-risk" if risk == "High" else "low-risk"
    risk_icon = "🔴" if risk == "High" else "🟢"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prediction</div>
            <div class="metric-value {risk_css}">{cls.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value {risk_css}">{risk_icon} {risk}</div>
        </div>
        """, unsafe_allow_html=True)

def render_plotly_probabilities(result: dict):
    """Render an interactive Plotly Bar chart for class probabilities."""
    all_probs = result.get("all_probs", [])
    if not all_probs:
        return

    # Prepare data
    df = pd.DataFrame(all_probs, columns=["Class", "Probability"])
    df["Percentage"] = df["Probability"] * 100
    df["Full Name"] = df["Class"].apply(lambda x: CLASS_INFO[x]["full_name"])
    
    # Define colors
    df["Color"] = df["Class"].apply(lambda x: "#ff5252" if x in HIGH_RISK_CLASSES else "#4caf50")

    fig = px.bar(
        df, 
        x="Percentage", 
        y="Class", 
        orientation='h',
        hover_data=["Full Name"],
        color="Class",
        color_discrete_sequence=df["Color"].tolist(),
        text=df["Percentage"].apply(lambda x: f"{x:.1f}%")
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, range=[0, 110], title=""),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        font=dict(color="#e2e8f0", family="Outfit, sans-serif")
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_dynamic_insights(result: dict):
    """Render the educational insight box based on the prediction."""
    cls = result["predicted_class"]
    info = CLASS_INFO[cls]
    risk = info["risk"]
    
    box_class = "insight-box" if risk == "High" else "insight-box low-risk-box"
    icon = "⚠️" if risk == "High" else "ℹ️"
    
    st.markdown(f"""
    <div class="{box_class}">
        <div class="insight-title">{icon} About {info['full_name']}</div>
        <div class="insight-desc">{info.get('description', 'No description available.')}</div>
        <div class="insight-rec"><b>Recommendation:</b> {info.get('recommendation', 'Consult a doctor if unsure.')}</div>
    </div>
    """, unsafe_allow_html=True)

def get_downloadable_report(result: dict) -> str:
    """Generate a Markdown string report of the result."""
    cls = result["predicted_class"]
    info = CLASS_INFO[cls]
    
    report = f"""# Dermo-Scope Analysis Report

**Prediction:** {info['full_name']} ({cls.upper()})
**Confidence:** {result['confidence'] * 100:.1f}%
**Risk Level:** {info['risk']}

## Disease Information
{info.get('description', '')}

## Recommendation
{info.get('recommendation', '')}

---
*Disclaimer: Dermo-Scope is an educational tool, not a medical diagnostic device. Always consult a dermatologist for medical advice.*
"""
    return report
