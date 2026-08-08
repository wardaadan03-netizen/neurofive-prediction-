import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model, scaler and features
# -----------------------------
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/churn_scaler.pkl")
feature_names = joblib.load("models/churn_features.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")
st.write(
    "Enter the customer's information below to predict whether "
    "the customer is likely to churn."
)


# -----------------------------
# Customer Information
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# -----------------------------
# Contract and Billing
# -----------------------------

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)


# -----------------------------
# Create model input
# -----------------------------

input_data = pd.DataFrame({
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],

    "gender_Male": [1 if gender == "Male" else 0],
    "Partner_Yes": [1 if partner == "Yes" else 0],
    "Dependents_Yes": [1 if dependents == "Yes" else 0],
    "PhoneService_Yes": [1 if phone_service == "Yes" else 0],

    "MultipleLines_No phone service": [
        1 if multiple_lines == "No phone service" else 0
    ],
    "MultipleLines_Yes": [
        1 if multiple_lines == "Yes" else 0
    ],

    "InternetService_Fiber optic": [
        1 if internet_service == "Fiber optic" else 0
    ],
    "InternetService_No": [
        1 if internet_service == "No" else 0
    ],

    "OnlineSecurity_No internet service": [
        1 if online_security == "No internet service" else 0
    ],
    "OnlineSecurity_Yes": [
        1 if online_security == "Yes" else 0
    ],

    "OnlineBackup_No internet service": [
        1 if online_backup == "No internet service" else 0
    ],
    "OnlineBackup_Yes": [
        1 if online_backup == "Yes" else 0
    ],

    "DeviceProtection_No internet service": [
        1 if device_protection == "No internet service" else 0
    ],
    "DeviceProtection_Yes": [
        1 if device_protection == "Yes" else 0
    ],

    "TechSupport_No internet service": [
        1 if tech_support == "No internet service" else 0
    ],
    "TechSupport_Yes": [
        1 if tech_support == "Yes" else 0
    ],

    "StreamingTV_No internet service": [
        1 if streaming_tv == "No internet service" else 0
    ],
    "StreamingTV_Yes": [
        1 if streaming_tv == "Yes" else 0
    ],

    "StreamingMovies_No internet service": [
        1 if streaming_movies == "No internet service" else 0
    ],
    "StreamingMovies_Yes": [
        1 if streaming_movies == "Yes" else 0
    ],

    "Contract_One year": [
        1 if contract == "One year" else 0
    ],
    "Contract_Two year": [
        1 if contract == "Two year" else 0
    ],

    "PaperlessBilling_Yes": [
        1 if paperless_billing == "Yes" else 0
    ],

    "PaymentMethod_Credit card (automatic)": [
        1 if payment_method == "Credit card (automatic)" else 0
    ],
    "PaymentMethod_Electronic check": [
        1 if payment_method == "Electronic check" else 0
    ],
    "PaymentMethod_Mailed check": [
        1 if payment_method == "Mailed check" else 0
    ]
})


# Make absolutely sure columns match training order
input_data = input_data.reindex(
    columns=feature_names,
    fill_value=0
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Churn", use_container_width=True):

    # Scale input using the scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Probability if available
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0][1]
    else:
        probability = None

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ The customer is likely to churn.")

        if probability is not None:
            st.write(
                f"Estimated churn probability: **{probability:.2%}**"
            )

    else:
        st.success("✅ The customer is unlikely to churn.")

        if probability is not None:
            st.write(
                f"Estimated churn probability: **{probability:.2%}**"
            )