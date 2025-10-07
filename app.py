import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load model and encoders
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    return model, scaler, encoders, target_encoder

model, scaler, encoders, target_encoder = load_artifacts()

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_data(df, encoders, scaler):
    df = df.copy()
    cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
    num_cols = ['tenure','MonthlyCharges','TotalCharges']

    # Encode categorical variables
    for col in cat_cols:
        if col in df.columns:
            le = encoders.get(col)
            if le:
                df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else le.classes_[0])

    # Scale numeric columns
    df[num_cols] = scaler.transform(df[num_cols])

    # Handle missing values
    df = df.fillna(0)

    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on their service details.")

menu = ["Single Prediction", "Batch Prediction"]
choice = st.sidebar.selectbox("Select Mode", menu)

# -----------------------------
# Single Prediction
# -----------------------------
if choice == "Single Prediction":
    st.subheader("Enter Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 100, 10)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

    input_dict = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_dict])
    processed_df = preprocess_data(input_df, encoders, scaler)

    if st.button("Predict"):
        try:
            prediction = model.predict(processed_df)
            result = target_encoder.inverse_transform(prediction)[0]
            st.success(f"**Prediction:** {result}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# -----------------------------
# Batch Prediction
# -----------------------------
else:
    st.subheader("Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            processed_df = preprocess_data(df, encoders, scaler)
            predictions = model.predict(processed_df)
            df['Churn_Prediction'] = target_encoder.inverse_transform(predictions)
            st.write("Predictions Completed")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", data=csv, file_name="churn_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
