import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Page Config
st.set_page_config(page_title="Diabetes AI", layout="centered")

# 1. Model Load Gochuu
@st.cache_resource
def load_my_model():
    # Model-ichi kalleettiidhaan akka dubbisamu
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
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])

# 3. Prediction Logic
if st.button("Predict Now"):
    # Tartiiba Model-ichi barbaadu (Total 9 features):
    # [Gender, Age, Hypertension, Heart_Disease, Smoking, BMI, HbA1c, Glucose, Residence]
    # Odeeffannoo nuti hin qabneef (Gender, Smoking, Residence) 0 galchina.
    features = np.array([[0, age, hypertension, heart_disease, 0, bmi, hba1c, glucose, 0]])
    
    # Madaallii (Prediction) gochuu
    prediction = model.predict(features)[0][0]
    
    st.markdown("---")
    if prediction > 0.5:
        st.error(f"Bu'aa: **DIABETIC** (Confidence: {prediction*100:.1f}%)")
        st.info("Gorsa: Maaloo ogeessa fayyaa dubbisuun qorannoo dabalataa godhaa.")
    else:
        st.success(f"Bu'aa: **NON-DIABETIC** (Confidence: {(1-prediction)*100:.1f}%)")
        st.info("Baga gammadde! Bu'aan kee dhibee sukkaaraa irra bilisa akka taate agarsiisa.")
