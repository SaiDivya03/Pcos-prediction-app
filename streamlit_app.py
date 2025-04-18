
import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and column names
model, scaler, feature_names = joblib.load("../models/pcos_model.pkl")

# Streamlit app layout
st.set_page_config(page_title="PCOS Prediction App", layout="centered")
st.title("💡 PCOS Detection")
st.write("Enter the following values to check PCOS prediction")

# Dynamic form based on feature names
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", format="%.2f")
    user_input.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("🔴 Prediction: Likely PCOS")
    else:
        st.success("🟢 Prediction: Not likely PCOS")
