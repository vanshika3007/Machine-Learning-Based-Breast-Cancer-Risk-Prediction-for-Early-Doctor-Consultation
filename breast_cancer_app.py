import streamlit as st
import numpy as np
import pickle
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    page_icon="🎀",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #fff5f7;
}
.stButton>button {
    background-color: #e63946;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("breast_cancer_model.pkl", "rb"))

# ---------------- MAIN TITLE ----------------
st.title(" AI-POWERED BREAST CANCER RISK PREDICTION SYSTEM FOR CLINICAL DECISION SUPPORT")
st.markdown("### AI-Powered Clinical Decision Support System")
st.image("https://media.giphy.com/media/efjT7lfJI3Oko/giphy.gif", use_column_width=250)

st.markdown("""
This system uses advanced Machine Learning algorithms to predict whether a breast tumor is **Benign or Malignant** 
based on medical measurements. Early detection improves survival rate and enables timely doctor consultation.

The model was trained on tumor feature data including mean, standard error, and worst values of 
radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.
""")

st.markdown("---")

# ---------------- MODEL DETAILS SECTION ----------------
st.header("📊 Models Evaluated")
st.image("https://media1.giphy.com/media/iJhw5xU0oc4FeNOi0C/source.gif", width=250)

st.markdown("""
The following Machine Learning models were trained and compared:

- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- Neural Network (MLP)  
- XGBoost  
- LightGBM  

Model performance was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
""")

st.success("🏆 Best Performing Model: Support Vector Machine (SVM)")
st.markdown("Selected based on highest ROC-AUC and Recall, which are critical for minimizing false negatives in medical diagnosis.")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.sidebar.header("📝 Enter Tumor Measurements")
st.sidebar.video('https://cdnl.iconscout.com/lottie/premium/preview-watermark/woman-doing-breast-cancer-awareness-animation-gif-download-10656636.mp4')

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

features = []

for feature in feature_names:
    value = st.sidebar.number_input(feature, value=0.0, format="%.5f")
    features.append(value)

input_data = np.array([features])

# ---------------- PREDICTION ----------------
st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWhyMHpqaXl1YmhmazJqeDF0bTVtc2NuMWs2aDc5d3FzdHZmNnFycCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/lhhagrA5kanz3BuzRr/giphy.webp", use_column_width=50)

if st.sidebar.button("🔍 Predict Cancer Risk"):

    with st.spinner("Analyzing Tumor Data..."):
        time.sleep(2)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

    st.header("🔎 Prediction Result")
    st.progress(int(probability * 100))

    if prediction[0] == 1:
        st.error("⚠ Malignant Tumor Detected")
        st.metric("Cancer Risk Probability", f"{round(probability*100,2)} %")
        st.image("https://media2.giphy.com/media/v1.Y2lkPTZjMDliOTUyOWxkbDR1eGNyZ3QxYzlseTAwNWpuaHF5b2NmbzZodG1xZHk0NTJwaCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/zbcGiWRNxfpIF0tLiH/200w.gif", width=50)

        st.markdown("## 👩‍⚕ Doctor Consultation: ✅ YES")
        st.markdown("""
        ✔ Immediate oncologist consultation  
        ✔ Biopsy confirmation  
        ✔ MRI / CT Scan  
        ✔ Begin treatment planning  
        """)

    else:
        st.success("✅ Benign Tumor Detected")
        st.metric("Cancer Risk Probability", f"{round(probability*100,2)} %")
        st.image("https://media3.giphy.com/media/Ierkq3VyOpH0cCvwPQ/source.gif", use_column_width=50)
        
        st.markdown("## 🩺 Doctor Consultation: ❌ NO (Routine Monitoring)")
        st.markdown("""
        ✔ Maintain yearly screening  
        ✔ Healthy lifestyle  
        ✔ Regular clinical check-ups  
        """)

st.markdown("---")
st.markdown("PRESENTED BY VANSHIKA BHARDWAJ AND NIYATI SINGH")









