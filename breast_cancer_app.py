import streamlit as st
import numpy as np
import pickle
import time

st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    page_icon="🎀",
    layout="wide"
)

# Load model
model = pickle.load(open("breast_cancer_model.pkl", "rb"))

st.title("🎀 Breast Cancer Risk Prediction System")
st.markdown("### Early Detection for Better Medical Consultation")

st.sidebar.header("Enter Tumor Measurements")

# IMPORTANT: Feature order must match training
features = []

feature_names = [
    'radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean',
    'concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se',
    'smoothness_se','compactness_se','concavity_se',
    'concave points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst',
    'area_worst','smoothness_worst','compactness_worst',
    'concavity_worst','concave points_worst',
    'symmetry_worst','fractal_dimension_worst'
]

for feature in feature_names:
    value = st.sidebar.number_input(feature, value=0.0)
    features.append(value)

input_data = np.array([features])

if st.sidebar.button("Predict Cancer"):

    with st.spinner("Analyzing Tumor Data..."):
        time.sleep(2)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠ Malignant Tumor Detected (Risk: {round(probability*100,2)}%)")

        st.markdown("## 👩‍⚕ Doctor Consultation Recommended")
        st.markdown("""
        - Consult an Oncologist immediately  
        - Schedule biopsy confirmation  
        - Perform MRI / CT scan  
        - Start early treatment planning  
        """)

    else:
        st.success(f"✅ Benign Tumor (Risk: {round(probability*100,2)}%)")

        st.markdown("## 🩺 Regular Monitoring Suggested")
        st.markdown("""
        - Maintain yearly screening  
        - Follow healthy lifestyle  
        - Regular clinical check-ups  
        """)

st.markdown("---")
st.markdown("Developed for Early Breast Cancer Risk Assessment")












