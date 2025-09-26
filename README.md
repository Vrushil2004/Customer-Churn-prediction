# Customer Churn Prediction

This project predicts customer churn for a telecom company using a simple machine learning pipeline. It uses the Telco-Customer-Churn dataset to train Logistic Regression and Random Forest models. Feature engineering is applied to customer data (e.g., tenure, services, charges) to improve accuracy. Models are compared based on accuracy, precision, and recall. A Streamlit dashboard allows users to upload data and predict churn, supporting business decisions.

## Highlights
- Designed ML pipeline to predict customer churn using classification models (Logistic Regression, Random Forest).
- Worked on feature engineering from customer behavior data to improve accuracy.
- Goal: Compare models & deploy using Streamlit dashboard for business decision support.

## Setup
1. Clone the repo: `git clone https://github.com/yourusername/Simple-Churn-Prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run `python train.py` to train models.
4. Run `streamlit run app.py` to launch the dashboard.

## Dataset
- Located in `data/Telco-Customer-Churn.csv`.
- Features: gender, tenure, services, charges, etc.
- Target: Churn (Yes/No).

## Files
- `train.py`: Loads data, preprocesses, engineers features, trains Logistic Regression and Random Forest, compares models, and saves the best.
- `app.py`: Streamlit app for uploading data and predicting churn.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Excludes unnecessary files.
- `LICENSE`: MIT License.

## Usage
- Train models: `python train.py`
- Use the Streamlit app to upload a CSV or input customer data for predictions.

## License
MIT License.
