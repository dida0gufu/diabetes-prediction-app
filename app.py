import streamlit as st
import pandas as pd
import numpy as np
import os

# Disable TensorFlow GPU to avoid CUDA errors on Streamlit Cloud
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(page_title="Diabetes AI Predictor", layout="centered")

# 1. Load the Trained Model
@st.cache_resource
def load_my_model():
    # Ensure this filename matches your file in GitHub exactly
    return load_model('diabetes_prediction_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")

# 2. User Interface (UI)
st.title("🩺 Diabetes Prediction AI")
st.write("Enter your health metrics below to assess the probability of diabetes.")

# Input Fields organized in columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male (1)" if x==1 else "Female (0)")
    age = st.number_input("Age", 0, 120, 25)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes (1)" if x==1 else "No (0)")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes (1)" if x==1 else "No (0)")

with col2:
    smoking_history = st.selectbox("Smoking History", [0, 1], format_func=lambda x: "Ever Smoked (1)" if x==1 else "Never Smoked (0)")
    bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
    hba1c = st.number_input("HbA1c Level", 0.0, 15.0, 5.5)
    glucose = st.number_input("Blood Glucose Level", 0, 300, 100)

# 3. Prediction Logic
if st.button("Predict Now"):
    # The model expects exactly 8 features:
    # [Gender, Age, Hypertension, Heart_Disease, Smoking_History, BMI, HbA1c, Glucose]
    features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]])
    
    # Generate Prediction
    prediction = model.predict(features)[0][0]
    
    st.markdown("---")
    if prediction > 0.5:
        st.error(f"Result: **DIABETIC** (Confidence: {prediction*100:.1f}%)")
        st.info("Recommendation: Please consult a healthcare professional for further testing.")
    else:
        st.success(f"Result: **NON-DIABETIC** (Confidence: {(1-prediction)*100:.1f}%)")
        st.info("Great news! Your results suggest a low risk of diabetes.")
