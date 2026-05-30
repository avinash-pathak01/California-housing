import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("housing_ann_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("California Housing Prediction")

longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")

if st.button("Predict"):

    rooms_per_household = total_rooms / households if households != 0 else 0

    bedrooms_per_room = total_bedrooms / total_rooms if total_rooms != 0 else 0

    population_per_household = population / households if households != 0 else 0

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

    st.success(f"Predicted Class: {pred_class}")
