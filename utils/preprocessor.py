import pandas as pd
from utils.logger import logger

def calculate_derived_features(applicant_income, coapplicant_income, loan_amount):
    """
    Calculates derived features for the loan prediction model.
    """
    try:
        total_income = applicant_income + coapplicant_income
        income_loan_ratio = total_income / loan_amount if loan_amount > 0 else 0
        return total_income, income_loan_ratio
    except Exception as e:
        logger.error(f"Error calculating derived features: {e}")
        return applicant_income + coapplicant_income, 0

def prepare_input_data(form_data):
    """
    Takes dictionary of form data and prepares it as a Pandas DataFrame
    suitable for the ML pipeline.
    """
    try:
        # Calculate derived features
        total_income, income_loan_ratio = calculate_derived_features(
            form_data.get("ApplicantIncome", 0),
            form_data.get("CoapplicantIncome", 0),
            form_data.get("LoanAmount", 0)
        )
        
        # Build dataframe
        input_data = pd.DataFrame([{
            "Gender": form_data.get("Gender"),
            "Married": form_data.get("Married"),
            "Dependents": form_data.get("Dependents"),
            "Education": form_data.get("Education"),
            "Self_Employed": form_data.get("Self_Employed"),
            "ApplicantIncome": form_data.get("ApplicantIncome"),
            "CoapplicantIncome": form_data.get("CoapplicantIncome"),
            "LoanAmount": form_data.get("LoanAmount"),
            "Loan_Amount_Term": form_data.get("Loan_Amount_Term"),
            "Credit_History": form_data.get("Credit_History"),
            "Property_Area": form_data.get("Property_Area"),
            "TotalIncome": total_income,
            "Income_Loan_Ratio": income_loan_ratio
        }])
        
        return input_data
    except Exception as e:
        logger.error(f"Error preparing input data: {e}")
        raise
