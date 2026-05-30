import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = load_model("housing_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("🏠 California Housing Prediction System")

st.markdown("""
Predict the housing category using demographic and housing information.

### How to Use
Enter the property details below and click **Predict**.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("ℹ️ Feature Information")

st.sidebar.info("""
**Longitude:** -124 to -114

**Latitude:** 32 to 42

**Housing Median Age:** 1 to 52 years

**Median Income:** Average income in the area

**Rooms/Bedrooms/Population/Households:** Census information
""")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📋 Enter Housing Details")

col1, col2 = st.columns(2)

with col1:

    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-110.0,
        value=-118.0,
        help="Example: -118.24"
    )

    latitude = st.number_input(
        "Latitude",
        min_value=30.0,
        max_value=45.0,
        value=34.0,
        help="Example: 34.05"
    )

    housing_median_age = st.slider(
        "Housing Median Age",
        min_value=1,
        max_value=60,
        value=25
    )

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        value=5.0,
        help="Example: 3.5 to 8.0"
    )

with col2:

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        value=3000,
        help="Example: 1500-6000"
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        value=500,
        help="Example: 200-1000"
    )

    population = st.number_input(
        "Population",
        min_value=1,
        value=1500,
        help="Example: 500-5000"
    )

    households = st.number_input(
        "Households",
        min_value=1,
        value=400,
        help="Example: 100-1000"
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict Housing Category", use_container_width=True):

    rooms_per_household = total_rooms / households

    bedrooms_per_room = total_bedrooms / total_rooms

    population_per_household = population / households

    data = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        rooms_per_household,
        bedrooms_per_room,
        population_per_household
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    pred_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(
        f"Predicted Class: {pred_class}"
    )

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

    # Class Meaning
    if pred_class == 0:
        st.info("🏡 Lower Housing Value Category")

    elif pred_class == 1:
        st.info("🏠 Higher Housing Value Category")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built using TensorFlow, Scikit-Learn and Streamlit")
