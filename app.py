import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("BigMart_XGBoost_Model.joblib")

# Streamlit UI
st.title("BigMart Sales Prediction")
st.write("Enter the feature values to predict sales:")

# Example: Adjust input fields as per your model's expected features
feature_count = 10  # Change this based on the number of features in your model
features = []

for i in range(feature_count):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

# Prediction button
if st.button("Predict"):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    st.success(f"Predicted Sales: {prediction:.2f}")