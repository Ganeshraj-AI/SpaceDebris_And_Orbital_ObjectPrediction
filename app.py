"""
===============================================================================
STREAMLIT SIMPLE PRODUCT INTERFACE
===============================================================================
Academic Goal:
Provide a simple, interactive prediction interface for testing trained models.

LEARNING CONCEPT:
-----------------
WHAT:  Streamlit Web UI for model inference.
WHY:   Allows users to interact with ML models without writing code.
HOW:   Streamlit captures user inputs, passes them to src/predict.py, loads the
       saved model from disk (models/final_model.pkl), and renders predictions.

IMPORTANT LEARNING NOTE:
------------------------
When you click [ Predict ], the model is NOT retrained!
The trained model parameters were saved into 'models/final_model.pkl'.
The app simply loads those pre-computed parameters to evaluate new inputs.
===============================================================================
"""

import sys
import os
import time
import streamlit as st
import pandas as pd
import numpy as np

# Add src folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_space_object


# Streamlit Page Configuration
st.set_page_config(
    page_title="Space Debris & Orbital Object Classification",
    page_icon="🛰️",
    layout="wide"
)

# Header Section
st.title("🛰️ Space Debris & Orbital Object Classification Interface")
st.caption("Academic ML Project — NASA/ESA CelesTrak Satellite Catalog (SATCAT) Data")

st.info(
    "💡 **Academic Clarification**: This is an **Object-Type Classification System** "
    "(Space Debris / Rocket Body vs Payload Satellite) based on orbital features. "
    "It is built for learning purposes and is **not** a direct collision-risk or operational collision avoidance system."
)

st.divider()

# Sidebar: Preset Educational Samples
st.sidebar.header("🧪 Preset Orbital Objects")
st.sidebar.markdown("Select a sample real-world orbit to auto-fill input parameters:")

preset = st.sidebar.selectbox(
    "Choose Preset Orbit:",
    [
        "Custom Input",
        "Sample 1: LEO Debris Fragment (SL-1 R/B)",
        "Sample 2: Operational Satellite (Vanguard 1)",
        "Sample 3: Geostationary Orbit Object (GEO Inactive)"
    ]
)

# Default parameter values
def_apogee = 938.0
def_perigee = 466.0
def_period = 96.19
def_inc = 65.10
def_rcs = 0.0800

if preset == "Sample 1: LEO Debris Fragment (SL-1 R/B)":
    def_apogee, def_perigee, def_period, def_inc, def_rcs = 938.0, 466.0, 96.19, 65.10, 20.4200
elif preset == "Sample 2: Operational Satellite (Vanguard 1)":
    def_apogee, def_perigee, def_period, def_inc, def_rcs = 3960.0, 650.0, 134.20, 34.25, 0.1200
elif preset == "Sample 3: Geostationary Orbit Object (GEO Inactive)":
    def_apogee, def_perigee, def_period, def_inc, def_rcs = 35786.0, 35786.0, 1436.00, 14.50, 0.5000

# Input Layout
st.subheader("1. Enter Orbital Characteristics")
col1, col2, col3 = st.columns(3)

with col1:
    apogee = st.number_input(
        "Apogee Altitude (km)",
        min_value=0.0,
        max_value=500000.0,
        value=def_apogee,
        step=10.0,
        help="Highest altitude of orbit above Earth's surface in kilometers."
    )
    perigee = st.number_input(
        "Perigee Altitude (km)",
        min_value=0.0,
        max_value=500000.0,
        value=def_perigee,
        step=10.0,
        help="Lowest altitude of orbit above Earth's surface in kilometers."
    )

with col2:
    period = st.number_input(
        "Orbital Period (minutes)",
        min_value=1.0,
        max_value=500000.0,
        value=def_period,
        step=1.0,
        help="Time required to complete one full revolution around Earth."
    )
    inclination = st.number_input(
        "Orbital Inclination (degrees)",
        min_value=0.0,
        max_value=180.0,
        value=def_inc,
        step=0.1,
        help="Tilt angle of orbit relative to Earth's equator (0° to 180°)."
    )

with col3:
    rcs_num = st.number_input(
        "Radar Cross Section - RCS (m²)",
        min_value=0.0001,
        max_value=1000.0,
        value=def_rcs,
        format="%.4f",
        step=0.01,
        help="Physical size signature detected by radar in square meters."
    )

# Engineered Features Preview
earth_radius = 6371.0
mean_alt = (apogee + perigee) / 2.0
eccentricity = (apogee - perigee) / (apogee + perigee + 2.0 * earth_radius)
semi_major = mean_alt + earth_radius
velocity_approx = np.sqrt(398600.4418 / semi_major)

st.caption(
    f"💡 **Calculated Derived Features Preview**: "
    f"Mean Altitude = **{mean_alt:.1f} km** | "
    f"Eccentricity = **{eccentricity:.4f}** | "
    f"Approx Velocity = **{velocity_approx:.2f} km/s**"
)

st.divider()

# Predict Button
if st.button("🚀 Predict Target Category", type="primary", use_container_width=True):
    try:
        # Perform Inference
        result = predict_space_object(
            period=period,
            inclination=inclination,
            apogee=apogee,
            perigee=perigee,
            rcs_num=rcs_num,
            model_path='models/final_model.pkl'
        )

        st.subheader("2. Prediction Result & Confidence")
        
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if result['prediction_class'] == 1:
                st.warning(f"### 🚀 Target Class 1: {result['category_label']}")
                st.markdown(
                    f"The model predicts with **{result['confidence_percent']:.2f}% confidence** "
                    f"that this object is **Space Debris or Rocket Body Junk**."
                )
            else:
                st.success(f"### 🛰️ Target Class 0: {result['category_label']}")
                st.markdown(
                    f"The model predicts with **{result['confidence_percent']:.2f}% confidence** "
                    f"that this object is a **Payload Satellite**."
                )

            # Probabilities Breakdown
            st.markdown("#### Probability Distribution:")
            prob_df = pd.DataFrame({
                "Category": ["Space Debris / Rocket Body (Class 1)", "Payload Satellite (Class 0)"],
                "Probability (%)": [result['debris_probability'] * 100.0, result['payload_probability'] * 100.0]
            })
            st.bar_chart(prob_df.set_index("Category"), height=200)

        with res_col2:
            st.metric(
                label="Prediction Confidence",
                value=f"{result['confidence_percent']:.2f}%"
            )
            st.metric(
                label="Inference Latency (KPI)",
                value=f"{result['latency_ms']:.3f} ms"
            )
            st.info(
                "⚡ **Inference Speed**: Prediction generated instantly using saved weights from `models/final_model.pkl`."
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("Please ensure 'models/final_model.pkl' exists by executing `python src/train.py`.")

st.divider()

# Educational Prediction Flow Diagram
with st.expander("📚 How Prediction Works (Step-by-Step Inference Flow)"):
    st.markdown("""
    ```text
    User Inputs (Apogee, Perigee, Period, Inclination, RCS)
                    ↓
    Calculate Derived Features (Mean Altitude, Eccentricity, Velocity)
                    ↓
    Format 8-Feature Vector X
                    ↓
    Load Saved Trained Model ('models/final_model.pkl') [No Retraining!]
                    ↓
    Pass Features through Decision Trees (Majority Voting)
                    ↓
    Compute Probability Score (e.g., 92.28% Debris)
                    ↓
    Display Final Class Output & Prediction Latency
    ```
    """)
    st.markdown("""
    - **Training vs. Inference**: Training fits tree split nodes on 54,000+ historical satellite tracks. Inference takes less than 1 millisecond by evaluating a single vector through the pre-computed tree nodes!
    """)

st.caption("Academic ML Project — Space Debris & Orbital Object Classification | Built for Learning Purposes Only")
