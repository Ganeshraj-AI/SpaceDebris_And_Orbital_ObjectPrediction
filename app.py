"""
===============================================================================
STREAMLIT HIGH-PERFORMANCE INTERACTIVE DASHBOARD
===============================================================================
Academic Goal:
Provide a modern, visually appealing, interactive prediction interface
for classifying space debris and payload satellites using trained ML models.

Features:
  - Tab 1: Real-Time Orbit Classifier & Probability Gauge
  - Tab 2: CelesTrak SATCAT Dataset Analytics & Interactive Charts
  - Tab 3: Model Evaluation Benchmark & Project KPI Performance
  - Tab 4: Student Learning Hub & Inference Flow Explanation
===============================================================================
"""

import sys
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add src folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_space_object


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Space Debris & Orbital Object Classification Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Styling
st.markdown("""
<style>
    /* Global Container Styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #38bdf8, #a855f7, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        font-weight: 400;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Category Result Cards */
    .result-debris {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.2) 0%, rgba(159, 18, 57, 0.3) 100%);
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 1.5rem;
        color: #ffe4e6;
    }
    .result-payload {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 95, 70, 0.3) 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem;
        color: #d1fae5;
    }
    
    /* Badge Pills */
    .orbit-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-leo { background-color: #0284c7; color: white; }
    .badge-meo { background-color: #d97706; color: white; }
    .badge-geo { background-color: #7c3aed; color: white; }
    .badge-heo { background-color: #dc2626; color: white; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HERO HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛰️ Space Debris & Orbital Object Classifier</div>
    <div class="hero-subtitle">
        Academic Machine Learning Dashboard • Sourced from real NASA/ESA CelesTrak Tracking Data (70,580 Objects)
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & PRESETS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🧪 Quick Presets")
st.sidebar.markdown("Select a real cataloged space trajectory to auto-fill inputs:")

preset = st.sidebar.selectbox(
    "Choose Preset Orbit:",
    [
        "Custom Input",
        "Sample 1: LEO Debris Fragment (SL-1 R/B)",
        "Sample 2: Operational Satellite (Vanguard 1)",
        "Sample 3: ISS Space Station Orbit",
        "Sample 4: Inactive GEO Debris (Geostationary)"
    ]
)

# Preset Values Setup
def_apogee, def_perigee, def_period, def_inc, def_rcs = 938.0, 466.0, 96.19, 65.10, 0.0800

if preset == "Sample 1: LEO Debris Fragment (SL-1 R/B)":
    def_apogee, def_perigee, def_period, def_inc, def_rcs = 938.0, 466.0, 96.19, 65.10, 20.4200
elif preset == "Sample 2: Operational Satellite (Vanguard 1)":
    def_apogee, def_perigee, def_period, def_inc, def_rcs = 3960.0, 650.0, 134.20, 34.25, 0.1200
elif preset == "Sample 3: ISS Space Station Orbit":
    def_apogee, def_perigee, def_period, def_inc, def_rcs = 420.0, 415.0, 92.90, 51.64, 400.0000
elif preset == "Sample 4: Inactive GEO Debris (Geostationary)":
    def_apogee, def_perigee, def_period, def_inc, def_rcs = 35786.0, 35786.0, 1436.00, 14.50, 0.5000

st.sidebar.divider()
st.sidebar.markdown("### ℹ️ Project Metadata")
st.sidebar.info(
    "• **Model**: Random Forest / XGBoost\n"
    "• **Dataset**: CelesTrak SATCAT\n"
    "• **Recall Score**: 96.43%\n"
    "• **Accuracy**: 92.34%\n"
    "• **Inference Time**: < 5.7 ms"
)


# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Real-Time Classifier",
    "📊 Dataset Analytics",
    "🏆 Model Benchmarks & KPIs",
    "📚 Student Learning Hub"
])


# =============================================================================
# TAB 1: REAL-TIME CLASSIFIER
# =============================================================================
with tab1:
    st.markdown("### 1. Enter Orbital Characteristics")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        apogee = st.number_input(
            "Apogee Altitude (km)",
            min_value=0.0,
            max_value=500000.0,
            value=def_apogee,
            step=10.0,
            help="Furthest distance above Earth's surface in kilometers."
        )
        perigee = st.number_input(
            "Perigee Altitude (km)",
            min_value=0.0,
            max_value=500000.0,
            value=def_perigee,
            step=10.0,
            help="Closest distance above Earth's surface in kilometers."
        )

    with col_in2:
        period = st.number_input(
            "Orbital Period (minutes)",
            min_value=1.0,
            max_value=500000.0,
            value=def_period,
            step=1.0,
            help="Time taken to complete one full orbit around Earth."
        )
        inclination = st.number_input(
            "Orbital Inclination (degrees)",
            min_value=0.0,
            max_value=180.0,
            value=def_inc,
            step=0.1,
            help="Tilt angle relative to Earth's equator (0° to 180°)."
        )

    with col_in3:
        rcs_num = st.number_input(
            "Radar Cross Section - RCS (m²)",
            min_value=0.0001,
            max_value=1000.0,
            value=def_rcs,
            format="%.4f",
            step=0.01,
            help="Physical size signature detected by radar in square meters."
        )

    # -------------------------------------------------------------------------
    # REAL-TIME DERIVED KEPLERIAN FEATURES PREVIEW
    # -------------------------------------------------------------------------
    earth_radius = 6371.0
    mean_alt = (apogee + perigee) / 2.0
    eccentricity = (apogee - perigee) / (apogee + perigee + 2.0 * earth_radius)
    semi_major = mean_alt + earth_radius
    velocity_approx = np.sqrt(398600.4418 / semi_major)

    # Determine Orbit Regime Badge
    if mean_alt < 2000:
        regime_badge = '<span class="orbit-badge badge-leo">LEO (Low Earth Orbit)</span>'
    elif mean_alt < 35786:
        regime_badge = '<span class="orbit-badge badge-meo">MEO (Medium Earth Orbit)</span>'
    elif mean_alt <= 36000:
        regime_badge = '<span class="orbit-badge badge-geo">GEO (Geostationary Orbit)</span>'
    else:
        regime_badge = '<span class="orbit-badge badge-heo">HEO (High Earth Orbit)</span>'

    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.4); padding: 0.8rem 1.2rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); margin-top: 0.5rem; margin-bottom: 1.5rem;">
        ⚡ <b>Calculated Keplerian Physics Preview</b>: 
        {regime_badge} | 
        Mean Altitude: <b>{mean_alt:.1f} km</b> | 
        Eccentricity: <b>{eccentricity:.4f}</b> | 
        Velocity: <b>{velocity_approx:.2f} km/s</b>
    </div>
    """, unsafe_allow_html=True)

    # Predict Button
    btn_predict = st.button("🚀 Predict Target Category", type="primary", use_container_width=True)

    if btn_predict:
        try:
            # Perform Single-Sample Inference
            res = predict_space_object(
                period=period,
                inclination=inclination,
                apogee=apogee,
                perigee=perigee,
                rcs_num=rcs_num,
                model_path='models/final_model.pkl'
            )

            st.markdown("---")
            st.markdown("### 2. Prediction Results & Probabilities")

            res_col1, res_col2 = st.columns([1.3, 1])

            with res_col1:
                if res['prediction_class'] == 1:
                    st.markdown(f"""
                    <div class="result-debris">
                        <h3 style="margin-top:0; color:#f43f5e;">🚀 Target Class 1: {res['category_label']}</h3>
                        <p style="font-size:1.1rem; margin-bottom:0;">
                            The model predicts with <b>{res['confidence_percent']:.2f}% confidence</b> 
                            that this object is an uncontrolled <b>Space Debris fragment or Rocket Body</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-payload">
                        <h3 style="margin-top:0; color:#10b981;">🛰️ Target Class 0: {res['category_label']}</h3>
                        <p style="font-size:1.1rem; margin-bottom:0;">
                            The model predicts with <b>{res['confidence_percent']:.2f}% confidence</b> 
                            that this object is an active or inactive <b>Payload Satellite</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Plotly Probability Breakdown Bar Chart
                fig_bar = go.Figure(go.Bar(
                    x=[res['debris_probability'] * 100.0, res['payload_probability'] * 100.0],
                    y=['Debris / Rocket Body', 'Payload Satellite'],
                    orientation='h',
                    marker=dict(color=['#f43f5e', '#10b981']),
                    text=[f"{res['debris_probability']*100:.1f}%", f"{res['payload_probability']*100:.1f}%"],
                    textposition='auto'
                ))
                fig_bar.update_layout(
                    title="Probability Distribution Breakdown",
                    xaxis_title="Confidence Percentage (%)",
                    height=240,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1')
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with res_col2:
                # Plotly Interactive Gauge Chart for Debris Risk %
                gauge_color = "#f43f5e" if res['debris_probability'] > 0.5 else "#10b981"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res['debris_probability'] * 100.0,
                    number={'suffix': "%"},
                    title={'text': "Debris Probability Gauge", 'font': {'size': 18, 'color': '#cbd5e1'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "rgba(30, 41, 59, 0.5)",
                        'borderwidth': 1,
                        'bordercolor': "#475569",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.15)'},
                            {'range': [50, 100], 'color': 'rgba(244, 63, 94, 0.15)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    height=240,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1')
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Metrics Summary Cards
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Model Latency", f"{res['latency_ms']:.3f} ms")
                with m_col2:
                    st.metric("Target Code", f"Class {res['prediction_class']}")

        except Exception as e:
            st.error(f"Inference Error: {e}")
            st.warning("Please ensure 'models/final_model.pkl' exists by running `python src/train.py`.")


# =============================================================================
# TAB 2: DATASET ANALYTICS & EDA
# =============================================================================
with tab2:
    st.markdown("### 📊 CelesTrak SATCAT Dataset Exploration")
    
    try:
        df_raw = pd.read_csv('data/dataset.csv')
        df_valid = df_raw[df_raw['OBJECT_TYPE'].isin(['DEB', 'R/B', 'PAY'])].copy()
        df_valid['is_debris'] = df_valid['OBJECT_TYPE'].apply(lambda x: 1 if x in ['DEB', 'R/B'] else 0)

        # Overview Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Cataloged Objects", f"{len(df_raw):,}")
        k2.metric("Debris & Rocket Bodies", f"{(df_valid['is_debris']==1).sum():,} ({df_valid['is_debris'].mean()*100:.1f}%)")
        k3.metric("Payload Satellites", f"{(df_valid['is_debris']==0).sum():,} ({(1-df_valid['is_debris'].mean())*100:.1f}%)")
        k4.metric("Data Completeness KPI", "88.16%")

        st.divider()

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            # Pie Chart of Object Categories
            cat_counts = df_raw['OBJECT_TYPE'].value_counts().reset_index()
            cat_counts.columns = ['Object Type', 'Count']
            fig_pie = px.pie(
                cat_counts,
                values='Count',
                names='Object Type',
                title="Object Type Distribution in SATCAT",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'))
            st.plotly_chart(fig_pie, use_container_width=True)

        with eda_col2:
            # Scatter Plot Apogee vs Perigee
            df_sample = df_valid.dropna(subset=['APOGEE', 'PERIGEE']).sample(n=min(3000, len(df_valid)), random_state=42)
            df_sample['Category'] = df_sample['is_debris'].apply(lambda x: 'Debris / Rocket Body' if x==1 else 'Payload Satellite')
            
            fig_scatter = px.scatter(
                df_sample,
                x='PERIGEE',
                y='APOGEE',
                color='Category',
                title="Apogee vs Perigee Altitude (Sampled 3,000 Objects)",
                log_x=True,
                log_y=True,
                color_discrete_map={'Debris / Rocket Body': '#f43f5e', 'Payload Satellite': '#10b981'}
            )
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'))
            st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load dataset analytics: {e}")


# =============================================================================
# TAB 3: MODEL BENCHMARKS & KPIS
# =============================================================================
with tab3:
    st.markdown("### 🏆 Machine Learning Model Evaluation Benchmarks")
    st.caption("Performance measured on 13,673 unseen test space objects:")

    benchmark_data = pd.DataFrame([
        {'Model': 'Logistic Regression', 'Accuracy': 0.7359, 'Precision': 0.7325, 'Recall': 0.8850, 'F1 Score': 0.8016, 'ROC-AUC': 0.7220, 'Latency (ms)': 0.747},
        {'Model': 'Random Forest (Default)', 'Accuracy': 0.9291, 'Precision': 0.9219, 'Recall': 0.9640, 'F1 Score': 0.9425, 'ROC-AUC': 0.9794, 'Latency (ms)': 43.890},
        {'Model': 'XGBoost Classifier', 'Accuracy': 0.9236, 'Precision': 0.9162, 'Recall': 0.9613, 'F1 Score': 0.9382, 'ROC-AUC': 0.9788, 'Latency (ms)': 5.694},
        {'Model': 'Random Forest (Tuned)', 'Accuracy': 0.9234, 'Precision': 0.9134, 'Recall': 0.9643, 'F1 Score': 0.9382, 'ROC-AUC': 0.9776, 'Latency (ms)': 43.434}
    ])

    st.dataframe(benchmark_data.style.highlight_max(subset=['Accuracy', 'Recall', 'F1 Score', 'ROC-AUC'], color='#1e3a8a'), use_container_width=True)

    # Plotly Grouped Bar Chart of Model Metrics
    fig_comp = go.Figure()
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    colors = ['#38bdf8', '#818cf8', '#f43f5e', '#10b981', '#fbbf24']

    for idx, metric in enumerate(metrics_list):
        fig_comp.add_trace(go.Bar(
            name=metric,
            x=benchmark_data['Model'],
            y=benchmark_data[metric] * 100.0,
            marker_color=colors[idx]
        ))

    fig_comp.update_layout(
        barmode='group',
        title="Model Metric Comparison (% Score)",
        yaxis_title="Percentage (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        height=350
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### 🎯 5 Project Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Hazard Detection Rate", "96.43%", delta="Target ≥ 85%")
    kpi2.metric("Recall Score", "96.43%", delta="Target ≥ 85%")
    kpi3.metric("F1 Score", "93.82%", delta="Target ≥ 85%")
    kpi4.metric("Data Completeness", "88.16%", delta="Target ≥ 80%")
    kpi5.metric("Prediction Latency", "5.69 ms", delta="Target < 10 ms")


# =============================================================================
# TAB 4: STUDENT LEARNING HUB
# =============================================================================
with tab4:
    st.markdown("### 📚 Student Machine Learning Hub")
    st.markdown("""
    This project is built to demonstrate **how Machine Learning works step-by-step**:
    """)

    with st.expander("🔄 1. The Inference Pipeline (How Predictions Are Made)"):
        st.markdown("""
        ```text
        User Enters Raw Inputs (Apogee, Perigee, Period, Inc, RCS)
                            ↓
        Calculate Keplerian Features (Mean Altitude, Eccentricity, Velocity)
                            ↓
        Format 8-Feature Numerical Vector X
                            ↓
        Load Pre-Trained Weights from 'models/final_model.pkl' [No Retraining!]
                            ↓
        Pass Features through Pre-Computed Decision Tree Split Nodes
                            ↓
        Compute Class Probabilities (e.g. 92.3% Debris, 7.7% Payload)
                            ↓
        Display Results & Prediction Latency in Streamlit UI
        ```
        """)

    with st.expander("⚡ 2. Why We Don't Retrain on Prediction"):
        st.markdown("""
        - **Training Phase (Done ONCE)**: Fits decision tree split nodes on 54,000+ historical satellite tracks.
        - **Prediction Phase (Inference)**: Loads pre-saved split nodes from disk and evaluates a single vector in under **5 milliseconds**.
        """)

    with st.expander("⚠️ 3. Why Recall is Critical in Space Hazard Safety"):
        st.markdown("""
        - **False Negative (FN)**: Missing a piece of space debris and misclassifying it as a safe satellite. This represents an un-tracked hazard!
        - **False Positive (FP)**: A false alarm misclassifying a satellite as debris.
        - **Priority**: We optimize **Recall** (96.43%) to catch as many space debris hazards as possible.
        """)

st.caption("Space Debris & Orbital Object Classification Dashboard • Built for Learning & Research Purposes")
