import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("BigMart_XGBoost_Model.joblib")

st.title("BigMart Sales Prediction")

# Define feature names (replace with actual feature names)
feature_names = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]

# Create input fields for each feature
features_list = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", step=0.01)
    features_list.append(value)

if st.button("Predict"):
    # Ensure all values are provided
    if any(v is None for v in features_list):
        st.error("Please enter values for all features.")
    else:
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features_list).reshape(1, -1)
        
        # Predict sales
        prediction = model.predict(features_array)[0]
        st.success(f"Predicted Sales: {prediction:.2f}")
