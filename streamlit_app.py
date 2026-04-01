import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import torch

# IMPORTANT: set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="Toxic Message Analyzer | Pro-Grade",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import existing logic (after config)
from backend.services.toxicity_service import toxicity_analyzer
from backend.services.rephraser_service import rephraser
from backend.db import SessionLocal, init_db
from backend.models import Message

# Custom UI Styling (Refactored for Stability)
CSS_THEME = """
<style>
    .main { background-color: #f8fafc; }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .stTextArea>div>div>textarea {
        border-radius: 15px;
        border: 1px solid #e2e8f0;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        background-color: white;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    .expert-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #bfdbfe;
        background-color: #eff6ff;
        color: #1e40af;
        font-style: italic;
    }
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }
</style>
"""
st.html(CSS_THEME) # Modern, stable way to inject CSS on 1.40.1

# Cache Models for Cloud Performance
@st.cache_resource
def load_models():
    toxicity_analyzer.load_model()
    return toxicity_analyzer, rephraser

tox_engine, rephrase_engine = load_models()
init_db()

# Sidebar Content
with st.sidebar:
    st.title("🛡️ Analyzer Logic")
    st.markdown("### Strategic Guide: 7 Tiers")
    
    tiers = [
        {"name": "Tier 1: Healthy", "range": "0-15%", "strategy": "Maintain professional clarity and support."},
        {"name": "Tier 2: Low-Risk", "range": "15-30%", "strategy": "Minor polishing for subtle phrasing refinement."},
        {"name": "Tier 3: Noticeable", "range": "30-45%", "strategy": "Diplomacy active. Neutralizing passive-aggression."},
        {"name": "Tier 4: Moderate", "range": "45-60%", "strategy": "Construction mode. Rebuilding direct critiques."},
        {"name": "Tier 5: High Risk", "range": "60-75%", "strategy": "Restorative shift. Transforming hostile intent."},
        {"name": "Tier 6: Severe", "range": "75-90%", "strategy": "Deep De-escalation and professional grace."},
        {"name": "Tier 7: Critical", "range": "90-100%", "strategy": "Full Masterful Restoration and cooperation."}
    ]
    
    for tier in tiers:
        with st.expander(f"{tier['name']}"):
            st.caption(f"Range: {tier['range']}")
            st.write(tier['strategy'])

    st.divider()
    st.caption("7-Tier Analytical Engine v2.8")

# Main Application Layout
st.title("Toxic Message Analyzer")
st.subheader("Pro-Grade 7-Tier Severity Engine & Restorative AI Strategy")

# Informatics Dashboard
db = SessionLocal()
history = db.query(Message).all()
db.close()

stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.metric("Total Scanned", len(history))
with stat_col2:
    high_risk_count = len([m for m in history if m.toxicity_score > 0.75])
    st.metric("High Risk Alerts", high_risk_count)
with stat_col3:
    avg_score = (sum([m.toxicity_score for m in history]) / len(history) * 100) if history else 0
    st.metric("Avg Severity", f"{avg_score:.1f}%")

# Main Input Interface
user_input = st.text_area("Analyze Feedback", placeholder="Paste professional feedback here to begin restorative rephrasing...", height=150)
if st.button("Execute Intelligence Analysis"):
    if user_input:
        with st.spinner("Calibrating Cloud Models..."):
            score, label, segmentation = tox_engine.predict(user_input)
            rephrased = rephrase_engine.rephrase(user_input, score=score)
            tier_data = rephrase_engine.get_tier_key(score)
            benchmark = rephrase_engine.examples[tier_data]["output"]
            
            # Persist Result
            db = SessionLocal()
            db.add(Message(
                user_input=user_input,
                toxicity_score=score,
                prediction_label=label,
                rephrased_output=rephrased,
                segmentation_scores=json.dumps(segmentation)
            ))
            db.commit()
            db.close()
            
            # Display Real-time Results
            r_col1, r_col2 = st.columns([1, 1.5])
            
            with r_col1:
                st.markdown("### Intensity Segmentation")
                score_color = "#ef4444" if score > 0.6 else "#4f46e5"
                st.html(f"<div style='text-align: center; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; background: white;'><strong>Toxicity Intensity</strong><h1 style='color: {score_color}; font-size: 3rem; margin: 0;'>{(score*100):.0f}%</h1></div>")
                
                radar_data = pd.DataFrame({
                    'Metric': [k.replace('_', ' ').upper() for k in segmentation.keys()],
                    'Value': [v * 100 for v in segmentation.values()]
                })
                fig = px.line_polar(radar_data, r='Value', theta='Metric', line_close=True)
                fig.update_traces(fill='toself', line_color='#4f46e5')
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)

            with r_col2:
                st.markdown("### Restorative AI Calibration")
                st.html(f"<div class='result-box'><strong>{rephrased}</strong></div>")
                
                st.markdown("### 📘 Expert Benchmark Standard")
                st.html(f"<div class='expert-box'>\"{benchmark}\"</div>")
                st.success("Calibration Successful: Tier-7 Expert Engine Active.")
    else:
        st.warning("Please enter some feedback for the engine to analyze.")

# Historical Analytics Suite
st.divider()
if st.checkbox("Toggle Historical Dashboard", value=False):
    st.markdown("## Multi-Dimension Data Insights")
    if history:
        h_df = pd.DataFrame([{"timestamp": m.timestamp, "score": m.toxicity_score, "label": m.prediction_label} for m in history])
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            # Global Sentiment
            pie_val = h_df['label'].value_counts()
            st.plotly_chart(px.pie(values=pie_val.values, names=pie_val.index, title="Global Sentiment Distribution", color_discrete_sequence=['#4f46e5', '#ef4444']), use_container_width=True)
            
            # Severity Trend
            h_df['date'] = pd.to_datetime(h_df['timestamp']).dt.date
            line_val = h_df.groupby('date')['score'].mean().reset_index()
            st.plotly_chart(px.line(line_val, x='date', y='score', title="Avg Severity Index Trend", markers=True).update_traces(line_color='#4f46e5'), use_container_width=True)
            
        with d_col2:
            # Hourly Volume
            h_df['hour'] = pd.to_datetime(h_df['timestamp']).dt.hour
            area_val = h_df.groupby(['hour', 'label']).size().reset_index(name='count')
            st.plotly_chart(px.area(area_val, x='hour', y='count', color='label', title="Layered Toxicity Volume", color_discrete_map={'Toxic': '#ef4444', 'Non-toxic': '#4f46e5'}), use_container_width=True)
            
            # Segment Prevalence (Averaged from historical records)
            try:
                hist_seg = pd.DataFrame([json.loads(m.segmentation_scores) for m in history]).mean().reset_index()
                hist_seg.columns = ['Metric', 'Avg']
                st.plotly_chart(px.bar(hist_seg, x='Metric', y='Avg', title="Segment Prevalence Hierarchy", color='Avg', color_continuous_scale='RdPu'), use_container_width=True)
            except Exception:
                st.info("Insufficient segmentation history for full analysis.")
    else:
        st.info("No historical scan records available.")

st.markdown("---")
st.caption("7-Tier Analytical Engine v2.8 | Intensity-Aware Calibration | Pro-Grade Visualization")
