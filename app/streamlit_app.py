"""
Interactive Streamlit Dashboard for Customer Churn Prediction.
Provides a recruiter-friendly UI with sliders and dropdowns to interactively test
the model in real-time. Can be deployed directly to Hugging Face Spaces or run locally.
"""

import streamlit as st
import pandas as pd
from app.model_loader import model_manager
from src.config import settings

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction — MLOps Live Demo")
st.markdown(
    "Predict subscription cancellation risk using a low-latency Gradient Boosted Tree "
    "pipeline with automated feature engineering."
)

# Load model pipeline into memory
if not model_manager.is_loaded:
    try:
        model_manager.load_model()
    except Exception as e:
        st.error(f"Failed to load model pipeline: {e}. Please ensure model training is complete.")
        st.stop()

# Layout: Two columns (Inputs on left, Prediction card on right)
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Customer Profile & Contract Details")

    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=4, step=1)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.50, step=0.5)
        # Default total charges calculated from tenure * monthly charges
        default_total = round(tenure * monthly_charges, 2)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(default_total), step=10.0)

    with c2:
        contract_type = st.selectbox("Contract Term", ["Month-to-month", "One year", "Two year"], index=0)
        internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"], index=0)
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=0,
        )
        tech_support = st.selectbox("Tech Support Subscription", ["No", "Yes", "No internet service"], index=0)

with col2:
    st.subheader("Real-Time Churn Risk Assessment")

    customer_dict = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "tech_support": tech_support,
    }

    if st.button("Evaluate Customer Risk", type="primary", use_container_width=True):
        with st.spinner("Scoring customer through pipeline..."):
            result = model_manager.predict_single_cached(**customer_dict)

        prob = result["churn_probability"]
        risk = result["risk_level"]
        label = result["churn_label"]

        # Risk indicator color
        if risk == "High":
            st.error(f"### High Risk: {prob:.1%} Churn Probability")
            st.warning("⚠️ **Retention Alert**: This customer exhibits strong cancellation signals. Immediate proactive retention discounts or technical support outreach recommended.")
        elif risk == "Medium":
            st.warning(f"### Moderate Risk: {prob:.1%} Churn Probability")
            st.info("ℹ️ Customer shows borderline loyalty. Monitor usage patterns.")
        else:
            st.success(f"### Low Risk: {prob:.1%} Churn Probability")
            st.info("✅ Customer is well-retained and exhibits high contract stability.")

        # Display raw JSON inference payload
        with st.expander("Inspect Raw Prediction JSON (FastAPI Equivalent)"):
            st.json(result)

st.markdown("---")
st.caption(f"Backend Service: FastAPI + LightGBM | Pipeline Version: {settings.VERSION} | Hosted on Free Cloud Tier")
