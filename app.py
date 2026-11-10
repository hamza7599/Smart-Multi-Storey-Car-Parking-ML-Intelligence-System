import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Smart Parking ML Dashboard", page_icon="🚗", layout="wide")

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load('parking_ml_pipeline.joblib')
    except Exception:
        return None

pipeline = load_pipeline()

st.title("🚗 Smart Multi-Storey Parking ML Intelligence System")
st.markdown("Predict parking customer clusters and optimize operations in real-time.")

with st.sidebar:
    st.header("Input Parking Parameters")
    car_type = st.selectbox("Car Type", ['Sedan', 'SUV', 'Hatchback', 'Electric', 'Truck'])
    entry_hour = st.slider("Entry Hour", 0, 23, 12)
    parking_duration_hrs = st.number_input("Parking Duration (Hours)", min_value=0.1, max_value=24.0, value=2.0)
    hourly_rate = st.number_input("Hourly Rate ($)", min_value=1.0, max_value=20.0, value=5.0)
    is_vip = st.selectbox("VIP Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    floor_name = st.selectbox("Floor Location", ['Ground Floor - Zone A', 'First Floor - Zone B', 'Second Floor - Zone C', 'Rooftop - Zone D'])
    shift = st.selectbox("Guard Shift", ['Morning', 'Evening', 'Night', 'Rotating'])

if st.button("Predict Parking Segment", type="primary"):
    if pipeline is None:
        st.error("Model pipeline artifact not found! Please run the training script first to generate 'parking_ml_pipeline.joblib'.")
    else:
        # Corrected dictionary and list brackets closing
        input_data = pd.DataFrame([{
            'entry_hour': entry_hour,
            'parking_duration_hrs': parking_duration_hrs,
            'hourly_rate': hourly_rate,
            'total_fee': parking_duration_hrs * hourly_rate,
            'is_vip': is_vip,
            'car_type_SUV': 1 if car_type == 'SUV' else 0,
            'car_type_Hatchback': 1 if car_type == 'Hatchback' else 0,
            'car_type_Electric': 1 if car_type == 'Electric' else 0,
            'car_type_Truck': 1 if car_type == 'Truck' else 0,
            'floor_name_First Floor - Zone B': 1 if floor_name == 'First Floor - Zone B' else 0,
            'floor_name_Second Floor - Zone C': 1 if floor_name == 'Second Floor - Zone C' else 0,
            'floor_name_Rooftop - Zone D': 1 if floor_name == 'Rooftop - Zone D' else 0,
            'shift_Evening': 1 if shift == 'Evening' else 0,
            'shift_Night': 1 if shift == 'Night' else 0,
            'shift_Rotating': 1 if shift == 'Rotating' else 0
        }])

        # Ensure all training columns exist
        for col in pipeline['feature_columns']:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[pipeline['feature_columns']]

        # Scale numerical features
        numerical_cols = ['entry_hour', 'parking_duration_hrs', 'hourly_rate', 'total_fee']
        input_data[numerical_cols] = pipeline['scaler'].transform(input_data[numerical_cols])

        # Predict
        prediction = pipeline['model'].predict(input_data)[0]

        st.success(f"### Predicted Parking Cluster Group: **Cluster {prediction}**")
        if prediction == 0:
            st.info("ℹ️ **Insight:** Short-term parker with standard fee profile. Ideal for rapid turnover zones.")
        else:
            st.warning("⚠️ **Insight:** Long-term parker or high-revenue profile. Recommended for reserved VIP floors.")