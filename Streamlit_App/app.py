import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Loan Approval Prediction System", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../Model/loan_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model_pipeline = load_model()

st.title("🏦 Loan Approval Prediction System")

tab1, tab2, tab3 = st.tabs(["Prediction", "Explainability", "Dataset Insights"])

with tab1:
    st.header("Applicant Information")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
    
    with col2:
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0.0, value=0.0, step=100.0)
        loan_amount = st.number_input("Loan Amount (in thousands $)", min_value=1.0, value=150.0, step=10.0)
        loan_amount_term = st.number_input("Loan Amount Term (Days)", min_value=12.0, value=360.0, step=12.0)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Meets Guidelines (1)" if x == 1.0 else "Does Not Meet (0)")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
    if st.button("Predict Loan Status", type="primary"):
        if model_pipeline is None:
            st.error("Model not found. Please ensure the model is trained and saved.")
        else:
            try:
                # Calculate new features
                total_income = applicant_income + coapplicant_income
                income_loan_ratio = total_income / loan_amount if loan_amount > 0 else 0
                
                input_data = pd.DataFrame([{
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": loan_amount_term,
                    "Credit_History": credit_history,
                    "Property_Area": property_area,
                    "TotalIncome": total_income,
                    "Income_Loan_Ratio": income_loan_ratio
                }])
                
                prediction = model_pipeline.predict(input_data)[0]
                probability = model_pipeline.predict_proba(input_data)[0][1]
                
                st.subheader("Prediction Results")
                if prediction == 1:
                    st.success(f"🎉 Loan Approved! (Probability: {probability:.2%})")
                    st.metric(label="Risk Level", value="Low/Moderate Risk")
                else:
                    st.error(f"❌ Loan Rejected. (Approval Probability: {probability:.2%})")
                    st.metric(label="Risk Level", value="High Risk")
                    
                st.info(f"Confidence Score: {max(probability, 1-probability):.2%}")
                
                # Store data in session for SHAP
                st.session_state['input_data'] = input_data
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.header("Model Explainability")
    if 'input_data' in st.session_state and model_pipeline is not None:
        try:
            st.write("Explaining the prediction for the last applicant:")
            input_data = st.session_state['input_data']
            
            # Extract models
            preprocessor = model_pipeline.named_steps['preprocessor']
            classifier = model_pipeline.named_steps['classifier']
            
            X_transformed = preprocessor.transform(input_data)
            
            # Get feature names
            numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome', 'Income_Loan_Ratio']
            categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
            cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            feature_names = list(numeric_features) + list(cat_cols)
            
            # Calculate SHAP values
            import scipy
            if scipy.sparse.issparse(X_transformed):
                X_transformed = X_transformed.toarray()
                
            if hasattr(classifier, 'feature_importances_'):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_transformed)
                
                plt.figure(figsize=(10, 4))
                if isinstance(shap_values, list):
                    shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_transformed[0], feature_names=feature_names, matplotlib=True)
                else:
                    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_transformed[0], feature_names=feature_names))
                st.pyplot(plt.gcf(), clear_figure=True)
                
            else:
                # Use LinearExplainer, requires background dataset
                train_data_path = os.path.join(os.path.dirname(__file__), '../Dataset/train.csv')
                df = pd.read_csv(train_data_path)
                df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
                df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'].replace(0, np.nan))
                df.dropna(subset=['Loan_Status'], inplace=True)
                X_train = df.drop('Loan_Status', axis=1, errors='ignore')
                X_train_transformed = preprocessor.transform(X_train)
                if scipy.sparse.issparse(X_train_transformed):
                    X_train_transformed = X_train_transformed.toarray()
                
                explainer = shap.LinearExplainer(classifier, X_train_transformed)
                shap_values = explainer.shap_values(X_transformed)
                
                plt.figure(figsize=(10, 4))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_transformed[0], feature_names=feature_names))
                st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:
            st.error(f"Could not generate SHAP explanation: {e}")
            st.info("Check if the model uses XGBoost or RandomForest.")
    else:
        st.info("Please run a prediction in the 'Prediction' tab first.")

with tab3:
    st.header("Dataset & Model Insights")
    
    asset_dir = os.path.join(os.path.dirname(__file__), '../assets')
    
    def display_image(filename, caption):
        filepath = os.path.join(asset_dir, filename)
        if os.path.exists(filepath):
            img = Image.open(filepath)
            st.image(img, caption=caption, use_column_width=True)
        else:
            st.warning(f"{filename} not found. Ensure the notebook was run to generate assets.")
            
    colA, colB = st.columns(2)
    with colA:
        display_image('heatmap.png', 'Correlation Heatmap')
        display_image('confusion_matrix.png', 'Confusion Matrix')
    with colB:
        display_image('feature_importance.png', 'Feature Importance')
        display_image('roc_curve.png', 'ROC Curve')
        
    st.subheader("Global Feature Contributions")
    display_image('shap_summary.png', 'SHAP Summary Plot')
