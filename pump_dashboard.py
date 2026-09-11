import streamlit as st
import pandas as pd
import joblib

# load everything we saved from the notebook
model = joblib.load("pump_rf_model.pkl")
scaler = joblib.load("pump_scaler.pkl")
label_encoder = joblib.load("pump_label_encoder.pkl")
top_features = joblib.load("pump_top_features.pkl")

st.set_page_config(page_title="Pump Failure Detection", page_icon=":gear:")

st.title("Industrial Pump Failure Detection")
st.write("Adjust the sensor readings on the left and check the predicted pump health status.")

st.sidebar.header("Sensor Readings")

# build one slider per top sensor - min/max/default just based on typical ranges
sensor_inputs = {}
for feature in top_features:
    sensor_inputs[feature] = st.sidebar.slider(
        feature, min_value=0.0, max_value=1000.0, value=200.0
    )

if st.button("Predict Pump Status"):
    input_df = pd.DataFrame([sensor_inputs])[top_features]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_status = label_encoder.inverse_transform(prediction)[0]

    if predicted_status == "NORMAL":
        st.success(f"Pump Status: {predicted_status}")
    elif predicted_status == "RECOVERING":
        st.warning(f"Pump Status: {predicted_status}")
    else:
        st.error(f"Pump Status: {predicted_status}")

st.write("---")
st.caption("Model: Random Forest | Project: Industrial Pump Failure Detection Using ML")
