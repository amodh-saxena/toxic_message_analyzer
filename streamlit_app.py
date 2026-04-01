import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import torch

# Import existing logic
from backend.services.toxicity_service import toxicity_analyzer
from backend.services.rephraser_service import rephraser
from backend.db import SessionLocal, init_db
from backend.models import Message

# Page Configuration
st.set_page_config(
    page_title="Toxic Message Analyzer | Pro-Grade",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetic
st.markdown("""
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
    }
    .expert-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #bfdbfe;
        background-color: #eff6ff;
        color: #1e40af;
        font-style: italic;
    }
    .severity-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_all_html=True)

# Cache Models
@st.cache_resource
def load_models():
    toxicity_analyzer.load_model()
    # Rephraser loads on init
    return toxicity_analyzer, rephraser

tox_engine, rephrase_engine = load_models()
init_db()

# Sidebar: Strategic Guide
with st.sidebar:
    st.title("🛡️ Analyzer Intelligence")
    st.markdown("### Strategic Guide: 7 Tiers")
    
    tiers = [
        {"name": "Tier 1: Healthy", "range": "0-15%", "strategy": "Everything looks great. The AI will preserve your professional tone while ensuring clarity."},
        {"name": "Tier 2: Low-Risk", "range: ": "15-30%", "strategy": "Minor polishing applied. Softening subtle edges for better recipient reception."},
        {"name": "Tier 3: Noticeable", "range": "30-45%", "strategy": "Diplomacy mode active. Identifying and neutralizing passive-aggressive indicators."},
        {"name": "Tier 4: Moderate", "range": "45-60%", "strategy": "Construction mode. Rebuilding direct critiques into constructive, project-focused feedback."},
        {"name": "Tier 5: High Risk", "range": "60-75%", "strategy": "Restorative shift. Transforming hostile intent into collaborative, supportive guidance."},
        {"name": "Tier 6: Severe", "range": "75-90%", "strategy": "Deep De-escalation. Implementing extreme professional grace for severe interpersonal friction."},
        {"name": "Tier 7: Critical", "range": "90-100%", "strategy": "Masterful Restoration. Full neutralization of extreme hostility into sincere cooperation."}
    ]
    
    for tier in tiers:
        with st.expander(f"{tier['name']} ({tier['range']})"):
            st.write(tier['strategy'])

    st.divider()
    st.caption("Powered by 7-Tier Analytical Engine v2.8")

# Main Header
st.title("Toxic Message Analyzer")
st.subheader("Pro-Grade 7-Tier Severity Engine & Restorative AI Strategy")

# Informatics Row
db = SessionLocal()
history = db.query(Message).all()
db.close()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Scanned", len(history))
with col2:
    high_risk = len([m for m in history if m.toxicity_score > 0.75])
    st.metric("High Risk Alerts", high_risk, delta=None, delta_color="inverse")
with col3:
    avg_tox = (sum([m.toxicity_score for m in history]) / len(history) * 100) if history else 0
    st.metric("Avg Severity", f"{avg_tox:.1f}%")

# Input Section
user_input = st.text_area("Input Feedback Analysis", placeholder="Paste your feedback here for restorative professional rephrasing...", height=150)
analyze_btn = st.button("Execute Analysis")

if analyze_btn and user_input:
    with st.spinner("Calibrating Tier-7 Intelligence..."):
        # 1. Prediction
        score, label, segmentation = tox_engine.predict(user_input)
        
        # 2. Rephrasing (Dynamic)
        rephrased = rephrase_engine.rephrase(user_input, score=score)
        
        # 3. Expert Benchmark (Using same logic as React App)
        tier_data = rephrase_engine.get_tier_key(score)
        benchmark = rephrase_engine.examples[tier_data]["output"]
        
        # 4. Save to DB
        db = SessionLocal()
        new_msg = Message(
            user_input=user_input,
            context="",
            toxicity_score=score,
            prediction_label=label,
            rephrased_output=rephrased,
            segmentation_scores=json.dumps(segmentation)
        )
        db.add(new_msg)
        db.commit()
        db.close()
        
        # Display Results
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.markdown("### Level Segmentation")
            color = "red" if score > 0.6 else "green"
            st.markdown(f"<div style='text-align: center; border: 1px solid #e2e8f0; padding: 15px; border-radius: 12px;'><strong>Toxicity Intensity</strong><h1 style='color: {color};'>{(score*100):.0f}%</h1></div>", unsafe_allow_all_html=True)
            
            # Radar Chart
            radar_df = pd.DataFrame({
                'Category': [k.replace('_', ' ').upper() for k in segmentation.keys()],
                'Score': [v * 100 for v in segmentation.values()]
            })
            fig = px.line_polar(radar_df, r='Score', theta='Category', line_close=True)
            fig.update_traces(fill='toself', line_color='#4f46e5')
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown("### Restorative AI Output")
            st.markdown(f"<div class='result-box'><strong>{rephrased}</strong></div>", unsafe_allow_all_html=True)
            
            st.markdown("### 📘 Expert Benchmark Rephrase")
            st.markdown(f"<div class='expert-box'>\"{benchmark}\"</div>", unsafe_allow_all_html=True)
            
            st.success("Analysis Complete: Calibrated via Tier-7 Expert Severity Engine.")

# Dashboard Section
st.divider()
if st.checkbox("Explore Detailed Analytics Dashboard", value=False):
    st.markdown("## Data Insights Dashboard")
    
    if history:
        df = pd.DataFrame([{
            "timestamp": m.timestamp,
            "score": m.score if hasattr(m, 'score') else m.toxicity_score, # Handle possible naming diff
            "label": m.prediction_label
        } for m in history])
        
        dash_col1, dash_col2 = st.columns(2)
        
        with dash_col1:
            # 1. Global Sentiment Profile (Pie)
            pie_data = df['label'].value_counts()
            fig_pie = px.pie(values=pie_data.values, names=pie_data.index, title="Global Sentiment Profile", color_discrete_sequence=['#4f46e5', '#ef4444'])
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # 2. Average Severity Index Trend (Line)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            line_data = df.groupby('date')['score'].mean().reset_index()
            fig_line = px.line(line_data, x='date', y='score', title="Avg Severity Index Trend", markers=True)
            fig_line.update_traces(line_color='#4f46e5')
            st.plotly_chart(fig_line, use_container_width=True)
            
        with dash_col2:
            # 3. Layered Toxicity Volume (Area)
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            area_data = df.groupby(['hour', 'label']).size().reset_index(name='count')
            fig_area = px.area(area_data, x='hour', y='count', color='label', title="Layered Toxicity Volume", color_discrete_map={'Toxic': '#ef4444', 'Non-toxic': '#4f46e5'})
            st.plotly_chart(fig_area, use_container_width=True)
            
            # 4. Segment Prevalence (Bar)
            # Re-calculating segment averages from JSON
            all_segs = []
            for m in history:
                try:
                    s = json.loads(m.segmentation_scores)
                    all_segs.append(s)
                except: continue
            
            if all_segs:
                seg_df = pd.DataFrame(all_segs).mean().reset_index()
                seg_df.columns = ['Segment', 'Avg Score']
                fig_bar = px.bar(seg_df, x='Segment', y='Avg Score', title="Segment Prevalence Hierarchy", color='Avg Score', color_continuous_scale='RdPu')
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No historical data available for dashboard visualization.")

st.markdown("---")
st.caption("7-Tier Analytical Engine v2.8 | Multi-Model Benchmark Verification | Powering Streamlit Cloud")
