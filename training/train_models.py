import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TRAIN_DATA_PATH, MODELS_DIR

def build_preprocessor():
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome', 'Income_Loan_Ratio']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    numeric_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def main():
    print("Loading data...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    
    # Feature Engineering
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome'].fillna(0)
    df['LoanAmount_Numeric'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Income_Loan_Ratio'] = df['TotalIncome'] / df['LoanAmount_Numeric']
    df.dropna(subset=['Loan_Status'], inplace=True)
    
    # Target encoding
    y = df['Loan_Status'].map({'Y': 1, 'N': 0}).values
    X = df.drop('Loan_Status', axis=1)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pre/Post SMOTE distribution
    pre_smote_counts = pd.Series(y_train).value_counts().to_dict()
    print(f"Pre-SMOTE Class Distribution: {pre_smote_counts}")
    
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_trans, y_train)
    post_smote_counts = pd.Series(y_res).value_counts().to_dict()
    print(f"Post-SMOTE Class Distribution: {post_smote_counts}")

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    best_score = -1
    best_model_name = ""

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'assets'), exist_ok=True)

    for name, clf in models.items():
        print(f"Training {name}...")
        pipeline = ImbPipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }
        
        # Save ROC curve data for plotting later
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name]['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        # Feature Importance
        model_obj = pipeline.named_steps['classifier']
        try:
            cat_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
            num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome', 'Income_Loan_Ratio']
            feat_names = num_cols + list(cat_cols)
            
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_.tolist()
                results[name]['feature_importances'] = dict(zip(feat_names, importances))
            elif hasattr(model_obj, 'coef_'):
                importances = model_obj.coef_[0].tolist()
                # Use absolute value for importance comparison
                results[name]['feature_importances'] = dict(zip(feat_names, [abs(x) for x in importances]))
        except Exception as e:
            print(f"Warning: Could not extract feature importances for {name}: {e}")

        # Save model
        filename = ""
        if name == 'Logistic Regression': filename = 'lr_model.pkl'
        elif name == 'Random Forest': filename = 'rf_model.pkl'
        elif name == 'XGBoost': filename = 'xgb_model.pkl'
        
        joblib.dump(pipeline, os.path.join(MODELS_DIR, filename))
        
        # Select best model based on F1
        if f1 > best_score:
            best_score = f1
            best_model_name = name

    # Save best model separately
    best_filename = ""
    if best_model_name == 'Logistic Regression': best_filename = 'lr_model.pkl'
    elif best_model_name == 'Random Forest': best_filename = 'rf_model.pkl'
    elif best_model_name == 'XGBoost': best_filename = 'xgb_model.pkl'
    
    import shutil
    shutil.copyfile(os.path.join(MODELS_DIR, best_filename), os.path.join(MODELS_DIR, 'best_model.pkl'))

    # Save metadata
    metadata = {
        'best_model': best_model_name,
        'pre_smote_distribution': pre_smote_counts,
        'post_smote_distribution': post_smote_counts,
        'metrics': results
    }
    
    with open(os.path.join(MODELS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Training complete! Metadata saved.")

if __name__ == "__main__":
    main()
