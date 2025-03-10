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

# API 5L Constraints with Geometrical Considerations
constraints = {
    "PIPE OD (INCH)": (4, 60),
    "PIPE WT (MM)": (6.4, 45),
    "C(%)": (0.02, 0.28),
    "Mn(%)": (0.30, 1.60),
    "Si(%)": (0.02, 0.45),
    "Cr(%)": (0.01, 0.50),
    "Ni(%)": (0.01, 0.50),
    "Nb(%)": (0.001, 0.10),
    "Ti(%)": (0.001, 0.06),
    "V(%)": (0.002, 0.10),
    "Mo(%)": (0.002, 0.35),
    "Cu(%)": (0.01, 0.50),
    "Al(%)": (0.001, 0.060),
    "N(%)": (0.001, 0.012),
    "B(%)": (0.0005, 0.0015),
    "CE(PCM)": (0.10, 0.50),
    "BEND ANGLE (°)": (10, 90),
    "BEND RADIUS": (3, 30)  # Minimum 3D bend radius constraint
}

# Additional Geometrical Constraints
geometrical_constraints = {
    "Wall Thinning Limit (%)": 12,
    "Max Ovality (%)": 5,
    "Springback Effect Consideration": "Higher for X70, X80, X100",
    "Thin-wall Wrinkling Risk": "D/t > 50 requires internal mandrel",
    "Thick-wall Bending Challenge": "D/t < 15 needs high induction power"
}

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

# Display Geometrical Constraints
st.subheader("📏 Geometrical Constraints")
st.json(geometrical_constraints)
st.markdown("---")

# Create input fields in two columns
col1, col2 = st.columns(2)
user_input = {}
for i, var in enumerate(independent_vars):
    min_val, max_val = constraints.get(var, (None, None))
    if i % 2 == 0:
        user_input[var] = col1.number_input(f"{var}", min_value=min_val, max_value=max_val, value=None, step=0.01,
                                            format="%.4f")
    else:
        user_input[var] = col2.number_input(f"{var}", min_value=min_val, max_value=max_val, value=None, step=0.01,
                                            format="%.4f")

# Convert input to DataFrame and apply scaling
if st.button("🚀 Predict"):
    if None in user_input.values():
        st.error("Please enter valid numerical values for all inputs.")
    else:
        input_df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(input_df)
        predictions = {target: model.predict(scaled_input)[0] for target, model in models.items()}

        # Display predictions in table format
        st.subheader("🔍 Predicted Values:")
        results_df = pd.DataFrame(predictions.items(), columns=["Dependent Variable", "Predicted Value"])
        st.dataframe(results_df.style.format({"Predicted Value": "{:.2f}"}).set_properties(**{'text-align': 'center'}))