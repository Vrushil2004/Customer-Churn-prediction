import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

st.title('Customer Churn Prediction')

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# File upload
uploaded_file = st.file_uploader('Upload Customer Data (CSV)', type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Preprocess
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    
    df['TenureCategory'] = pd.cut(df['tenure'], bins=[0, 12, 36, np.inf], labels=[0, 1, 2])
    df['TotalServices'] = df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                             'StreamingTV', 'StreamingMovies']].sum(axis=1)
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Predict
    predictions = model.predict(df)
    df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
    
    st.write('Predictions:')
    st.dataframe(df)
    
    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Predictions', csv, 'predictions.csv')

# Input form
st.subheader('Predict for One Customer')
with st.form('predict_form'):
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure (months)', min_value=0)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
    total_charges = st.number_input('Total Charges', min_value=0.0)
    
    submit = st.form_submit_button('Predict')
    if submit:
        data = pd.DataFrame({
            'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner], 'Dependents': [dependents],
            'tenure': [tenure], 'PhoneService': [phone_service], 'MultipleLines': [multiple_lines],
            'InternetService': [internet_service], 'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup], 'DeviceProtection': [device_protection],
            'TechSupport': [tech_support], 'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies], 'Contract': [contract],
            'PaperlessBilling': [paperless_billing], 'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
        })
        
        # Preprocess
        for col in cat_cols:
            data[col] = LabelEncoder().fit_transform(data[col])
        data['TenureCategory'] = pd.cut(data['tenure'], bins=[0, 12, 36, np.inf], labels=[0, 1, 2])
        data['TotalServices'] = data[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                      'StreamingTV', 'StreamingMovies']].sum(axis=1)
        data[num_cols] = scaler.transform(data[num_cols])
        
        # Predict
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]
        st.write(f'Predicted Churn: {"Yes" if pred == 1 else "No"} (Probability: {prob:.2f})'