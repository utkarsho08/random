import streamlit as st
import pandas as pd
from config.settings import APP_NAME
from utils.ui import render_header, render_kpi_card, init_session_state
from utils.predictor import ModelPredictor

st.set_page_config(page_title=f"Loan Prediction | {APP_NAME}", layout="wide")
init_session_state()

render_header("Loan Prediction Module", "Input applicant details to receive a machine learning prediction.")

# Form Layout
with st.form("prediction_form"):
    st.subheader("Applicant Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        
    with col2:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        
    with col3:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
    st.subheader("Financial Details")
    col4, col5 = st.columns(2)
    
    with col4:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0, step=500)
        
    with col5:
        loan_amount = st.number_input("Loan Amount ($ in thousands)", min_value=0.0, value=150.0, step=10.0)
        loan_amount_term = st.selectbox("Loan Term (Months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
        
    st.subheader("Credit Profile")
    credit_history = st.radio("Credit History Meets Guidelines?", ["Yes", "No"], index=0)
    
    submitted = st.form_submit_button("Predict Loan Status", type="primary")

if submitted:
    # Prepare input dataframe
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': 1.0 if credit_history == "Yes" else 0.0,
        'Property_Area': property_area
    }
    
    df = pd.DataFrame([input_data])
    
    with st.spinner("Analyzing applicant profile..."):
        predictor = ModelPredictor()
        try:
            result = predictor.predict(df)
            
            st.divider()
            st.subheader("Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if result['approved']:
                    st.success("✅ **Loan Approved**")
                else:
                    st.error("❌ **Loan Rejected**")
                    
                render_kpi_card("Approval Probability", f"{result['probability']*100:.1f}%")
                
            with res_col2:
                st.info(f"The model is {result['confidence']*100:.1f}% confident in this decision.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
