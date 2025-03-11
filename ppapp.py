import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Independent Variables
independent_vars = [
    "PIPE OD (INCH)", "PIPE WT (MM)", "C(%)", "Mn(%)", "Si(%)", "Cr(%)", "Ni(%)",
    "Nb(%)", "Ti(%)", "V(%)", "Mo(%)", "Cu(%)", "Al(%)", "N(%)", "B(%)", "CE(PCM)",
    "BEND ANGLE (°)", "BEND RADIUS"
]

dependent_vars = [
    "OD TEMPERATURE INTRADOS (°C)", "WATER FLOW RATE (M3 PER HR)", "SPEED (MM PER MIN)",
    "BEND FREQUENCY (HZ)", "INDUCTION POWER (KW)",
    "WATER PRESSURE (KG PER CM2)", "AIR PRESSURE (KG PER CM2)"
]

# Load trained models
model_dir = "saved_models"
models = {}
for target in dependent_vars:
    model_path = os.path.join(model_dir, f"rf_model_{target}.pkl")
    if os.path.exists(model_path):
        models[target] = joblib.load(model_path)

# Load data to apply scaling
file_path = "Book2.xlsx"
df = pd.read_excel(file_path)
df.fillna(df.mean(), inplace=True)
scaler = RobustScaler()
scaler.fit(df[independent_vars])

# Streamlit UI
st.set_page_config(page_title="Random Forest Regression Predictor", layout="wide")
st.title("🌟 Random Forest Regression Predictor 🌟")
st.write("Enter values for independent variables to predict dependent variables.")
st.markdown("---")

# Create input fields in two columns
col1, col2 = st.columns(2)
user_input = {}

for i, var in enumerate(independent_vars):
    if i % 2 == 0:
        user_input[var] = col1.text_input(f"{var}", value="", placeholder="Enter value")
    else:
        user_input[var] = col2.text_input(f"{var}", value="", placeholder="Enter value")

# Convert input to DataFrame and apply scaling
if st.button("🚀 Predict"):
    try:
        # Convert user input to numeric values
        input_values = {var: float(user_input[var]) for var in user_input if user_input[var] != ""}

        if len(input_values) < len(independent_vars):
            st.error("Please enter valid numerical values for all inputs.")
        else:
            input_df = pd.DataFrame([input_values])
            scaled_input = scaler.transform(input_df)
            predictions = {target: model.predict(scaled_input)[0] for target, model in models.items()}

            # Display predictions in table format
            st.subheader("🔍 Predicted Values:")
            results_df = pd.DataFrame(predictions.items(), columns=["Dependent Variable", "Predicted Value"])
            st.dataframe(
                results_df.style.format({"Predicted Value": "{:.2f}"}).set_properties(**{'text-align': 'center'}))

    except ValueError:
        st.error("Please enter only numerical values.")

