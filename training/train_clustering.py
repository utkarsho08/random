import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import TRAIN_DATA_PATH, MODELS_DIR

def train_and_save_clustering_model():
    data_path = TRAIN_DATA_PATH
    model_path = os.path.join(MODELS_DIR, 'clustering_model.pkl')
    
    # Ensure directories exist
    os.makedirs('../models', exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature Engineering
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome'].fillna(0)
    
    # Handle missing values simply for clustering
    df['Credit_History'] = df['Credit_History'].fillna(1.0)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Status_Bool'] = df['Loan_Status'] == 'Y'
    
    features = ['TotalIncome', 'LoanAmount', 'Credit_History']
    X = df[features].copy()
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training KMeans with 5 clusters...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Fit PCA for 2D visualization
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    print("Analyzing centroids to assign segment names...")
    # Analyze clusters
    cluster_stats = df.groupby('Cluster')[features + ['Loan_Status_Bool']].mean()
    
    # Sort by Credit History first, then Income
    cluster_stats = cluster_stats.sort_values(by=['Credit_History', 'TotalIncome'])
    
    # The sorted order will typically be:
    # 0: Bad Credit, Low Income -> High-Risk Borrowers
    # 1: Bad Credit, High Income -> Credit-Constrained Applicants
    # 2: Good Credit, Low Income -> Growth Borrowers
    # 3: Good Credit, Med Income -> Stable Professionals
    # 4: Good Credit, High Income -> Premium Borrowers
    
    sorted_idx = cluster_stats.index.tolist()
    
    cluster_mapping = {
        sorted_idx[0]: "High-Risk Borrowers",
        sorted_idx[1]: "Credit-Constrained Applicants",
        sorted_idx[2]: "Growth Borrowers",
        sorted_idx[3]: "Stable Professionals",
        sorted_idx[4]: "Premium Borrowers"
    }
    
    print("Final Mapping:")
    for k, v in cluster_mapping.items():
        print(f"Cluster {k} -> {v}")
        
    # Calculate composition and approval rates
    counts = df['Cluster'].value_counts().to_dict()
    total = len(df)
    
    segment_profiles = {}
    for c_id, name in cluster_mapping.items():
        stats = cluster_stats.loc[c_id]
        segment_profiles[name] = {
            "avg_income": stats['TotalIncome'],
            "avg_loan": stats['LoanAmount'],
            "avg_credit": stats['Credit_History'],
            "approval_rate": stats['Loan_Status_Bool'] * 100,
            "population_pct": (counts.get(c_id, 0) / total) * 100
        }
        
    # Save artifacts
    artifacts = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "mapping": cluster_mapping,
        "profiles": segment_profiles,
        "features": features
    }
    
    joblib.dump(artifacts, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_clustering_model()
