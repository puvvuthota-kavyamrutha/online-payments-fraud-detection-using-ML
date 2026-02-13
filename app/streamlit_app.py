import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Online Payments Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Online Payments Fraud Detection System")
st.markdown("Detect fraudulent online transactions using Machine Learning.")

# -----------------------------------
# Load Model
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
    feature_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_path)

    return model, feature_columns

try:
    model, feature_columns = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("📊 About")
st.sidebar.info(
    """
    Random Forest model trained to detect fraudulent online transactions.
    """
)

# ===================================
# 🔹 INPUT FORM
# ===================================
st.subheader("Enter Transaction Details")

step = st.text_input("Step")
amount = st.text_input("Amount")
oldbalanceOrg = st.text_input("Old Balance Origin")
newbalanceOrig = st.text_input("New Balance Origin")
oldbalanceDest = st.text_input("Old Balance Destination")
newbalanceDest = st.text_input("New Balance Destination")

transaction_type = st.selectbox(
    "Select Transaction Type",
    ["Select Type", "PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
)

if st.button("Check Transaction"):

    # Validate input
    if (
        step == "" or
        amount == "" or
        oldbalanceOrg == "" or
        newbalanceOrig == "" or
        oldbalanceDest == "" or
        newbalanceDest == "" or
        transaction_type == "Select Type"
    ):
        st.warning("⚠️ Please fill all fields before submitting.")
    else:
        try:
            # Type encoding (must match training encoding)
            type_mapping = {
                "PAYMENT": 0,
                "TRANSFER": 1,
                "CASH_OUT": 2,
                "DEBIT": 3
            }

            input_data = {
                "step": float(step),
                "amount": float(amount),
                "oldbalanceOrg": float(oldbalanceOrg),
                "newbalanceOrig": float(newbalanceOrig),
                "oldbalanceDest": float(oldbalanceDest),
                "newbalanceDest": float(newbalanceDest),
                "type": type_mapping[transaction_type]
            }

            # Maintain correct feature order
            input_list = []
            for col in feature_columns:
                input_list.append(float(input_data.get(col, 0)))

            input_array = np.array([input_list])

            # Get probability of fraud
            probability = model.predict_proba(input_array)[0][1]

            # Prediction threshold
            prediction = 1 if probability >= 0.5 else 0

            # Risk classification
            if probability >= 0.8:
                risk_level = "High Risk"
            elif probability >= 0.5:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"

            st.markdown("---")

            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

            st.write(f"Fraud Probability: **{round(float(probability), 4)}**")
            st.write(f"Risk Level: **{risk_level}**")

        except Exception as e:
            st.error(f"Invalid input: {e}")
