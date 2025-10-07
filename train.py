import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('data/Teleco-Customer-Churn.csv')

# Clean and handle missing data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# -----------------------------
# Encode categorical variables
# -----------------------------
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders for the app
joblib.dump(encoders, 'encoders.pkl')

# -----------------------------
# Encode target variable
# -----------------------------
target_le = LabelEncoder()
df['Churn'] = target_le.fit_transform(df['Churn'])
joblib.dump(target_le, 'target_encoder.pkl')

# -----------------------------
# Split features & target
# -----------------------------
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Scale numeric features
# -----------------------------
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# -----------------------------
# Train Logistic Regression model
# -----------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, 'model.pkl')
print("\n Saved files: model.pkl, scaler.pkl, encoders.pkl, target_encoder.pkl")
