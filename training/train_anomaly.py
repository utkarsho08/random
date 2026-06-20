import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import TRAIN_DATA_PATH, MODELS_DIR

def train_and_save_anomaly_model():
    data_path = TRAIN_DATA_PATH
    model_path = os.path.join(MODELS_DIR, 'anomaly_model.pkl')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature Engineering for Anomaly Detection
    # Let's use some key ratios and raw values
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome'].fillna(0)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Income_Loan_Ratio'] = df['TotalIncome'] / df['LoanAmount']
    df['Credit_History'] = df['Credit_History'].fillna(1.0)
    
    # Simple encoding for dependents
    df['Dependents_Num'] = df['Dependents'].replace({'3+': 3, '0': 0, '1': 1, '2': 2}).fillna(0).astype(int)
    
    features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Income_Loan_Ratio', 'Credit_History', 'Dependents_Num']
    X = df[features].copy()
    
    # Impute remaining NaNs
    X = X.fillna(X.median())
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Isolation Forest...")
    # contamination = 0.05 implies we expect ~5% of historical data to be anomalous
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X_scaled)
    
    # Calculate baseline scores for portfolio comparison
    scores = iso_forest.score_samples(X_scaled)
    # Convert scores to a 0-100 anomaly scale
    # score_samples returns negative anomaly scores (lower is more anomalous)
    # We want 0-100 where 100 is highly anomalous
    min_score = scores.min()
    max_score = scores.max()
    
    # Store artifacts
    artifacts = {
        "scaler": scaler,
        "iso_forest": iso_forest,
        "features": features,
        "score_range": (min_score, max_score)
    }
    
    joblib.dump(artifacts, model_path)
    print(f"Anomaly model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_anomaly_model()
