import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("BigMart_XGBoost_Model.joblib")

st.title("BigMart Sales Prediction")

# Input form for features
features_input = st.text_input("Enter 4 feature values (comma-separated):")

if st.button("Predict"):
    try:
        features_list = [float(x) for x in features_input.split(",")]
        
        if len(features_list) != 4:
            st.error(f"Feature shape mismatch! Expected 4 features, but got {len(features_list)}.")
        else:
            features_array = np.array(features_list).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            st.success(f"Predicted Sales: {prediction:.2f}")

    except ValueError:
        st.error("Invalid input! Please enter numeric values separated by commas.")
