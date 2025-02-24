import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("BigMart_XGBoost_Model.joblib")

# Define product categories for better understanding
ITEM_TYPES = {
    "FD": "Food",
    "DR": "Drinks",
    "NC": "Non-Consumable"
}

# Streamlit App
st.title("🏪 BigMart Sales Prediction")

# User Input Section
st.header("Enter Product & Store Details")

# Product Details
item_identifier = st.text_input("Item Identifier (e.g., FDX001)")
item_type = ITEM_TYPES.get(item_identifier[:2], "Unknown")
item_weight = st.number_input("Item Weight (kg)", min_value=0.1, step=0.1)
item_visibility = st.number_input("Item Visibility (0-1)", min_value=0.0, max_value=1.0, step=0.01)
item_mrp = st.number_input("Item MRP (₹)", min_value=1.0, step=0.5)

# Store Details
outlet_identifier = st.text_input("Outlet Identifier (e.g., OUT027)")
outlet_type = st.selectbox("Outlet Type", ["Supermarket", "Grocery Store"])
outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "Large"])
outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1980, max_value=2024, step=1)

# Prediction Button
if st.button("Predict Sales"):
    # Prepare input features (ensure it matches the model's training format)
    features = np.array([item_weight, item_visibility, item_mrp, outlet_establishment_year]).reshape(1, -1)
    prediction = model.predict(features)[0]
    
    # Display Results
    st.subheader("📊 Predicted Sales")
    st.metric(label="Expected Revenue", value=f"₹{prediction:.2f}")
    
    # Additional Insights
    avg_sales = 4000  # Hypothetical average sales for comparison
    performance = "🔥 Above Average" if prediction > avg_sales else "📉 Below Average"
    
    st.subheader("🔍 Product Summary")
    st.write(f"**Item Name:** {item_identifier} ({item_type})")
    st.write(f"**Item Weight:** {item_weight} kg")
    st.write(f"**Item MRP:** ₹{item_mrp}")
    
    st.subheader("🏢 Store Details")
    st.write(f"**Outlet ID:** {outlet_identifier}")
    st.write(f"**Outlet Type:** {outlet_type}")
    st.write(f"**Outlet Size:** {outlet_size}")
    st.write(f"**Outlet Location Type:** {outlet_location_type}")
    st.write(f"**Established Year:** {outlet_establishment_year}")
    
    st.subheader("📈 Performance Insights")
    st.write(f"**Sales Performance:** {performance}")
