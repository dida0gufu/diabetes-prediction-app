import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Diabetes AI", layout="centered")

# 1. Model Load Gochuu
@st.cache_resource
def load_my_model():
    return load_model('diabetes_prediction_model.h5')

model = load_my_model()

# 2. Interface (UI)
st.title("🩺 Diabetes Prediction AI")
st.write("Odeeffannoo kee galchuun carraa dhibee sukkaaraa qabaachuu kee madaali.")

# Input Fields
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 0, 120, 25)
    glucose = st.number_input("Blood Glucose Level", 0, 300, 100)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

with col2:
    hba1c = st.number_input("HbA1c Level", 0.0, 15.0, 5.5)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])

# 3. Prediction Logic
if st.button("Predict Now"):
    # Input data (Tartiiba model-ichi barbaaduun)
    # Hubachiisa: Scaler fi Feature engineering asitti dabalama
    features = np.array([[age, hypertension, heart_disease, bmi, hba1c, glucose]])
    
    prediction = model.predict(features)[0][0]
    
    st.markdown("---")
    if prediction > 0.5:
        st.error(f"Bu'aa: **DIABETIC** (Confidence: {prediction*100:.1f}%)")
    else:
        st.success(f"Bu'aa: **NON-DIABETIC** (Confidence: {(1-prediction)*100:.1f}%)")
