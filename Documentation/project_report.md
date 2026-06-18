# Project Report: Loan Approval Prediction System

## 1. Introduction
The Loan Approval Prediction System is designed to automate the process of determining loan eligibility in real time based on customer details. By leveraging machine learning, the system can reduce human error, provide consistency, and accelerate the decision-making process for financial institutions.

## 2. Problem Statement
Financial institutions receive numerous loan applications daily. Manually evaluating each application based on various criteria (e.g., income, credit history, marital status) is time-consuming and prone to subjective biases. The objective is to build a predictive model that can classify an applicant's loan eligibility as 'Approved' or 'Rejected' using historical data.

## 3. Objectives
* Automate the loan approval decision process.
* Build a reproducible end-to-end Machine Learning pipeline.
* Evaluate and compare multiple classification models (Logistic Regression, Random Forest, XGBoost).
* Create an interactive user interface for business users to input data and get predictions.
* Provide model explainability using SHAP values to maintain transparency.

## 4. Dataset Description
The dataset contains attributes related to loan applicants.
* **Target Variable**: `Loan_Status` (Y/N)
* **Independent Variables**:
    * `Gender`: Male / Female
    * `Married`: Yes / No
    * `Dependents`: 0, 1, 2, 3+
    * `Education`: Graduate / Not Graduate
    * `Self_Employed`: Yes / No
    * `ApplicantIncome`: Applicant's income
    * `CoapplicantIncome`: Co-applicant's income
    * `LoanAmount`: Loan amount in thousands
    * `Loan_Amount_Term`: Term of loan in days
    * `Credit_History`: Meets guidelines (1.0) / Does not meet (0.0)
    * `Property_Area`: Urban / Semiurban / Rural

## 5. Methodology
The project follows a standard Data Science lifecycle:
1. Data Understanding & Exploratory Data Analysis (EDA)
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Training & Cross-Validation
5. Hyperparameter Tuning
6. Model Evaluation
7. Deployment using Streamlit

## 6. EDA Findings
* **Target Imbalance**: The dataset is slightly imbalanced, with more approved loans than rejected ones.
* **Credit History**: Applicants with a credit history meeting guidelines have a significantly higher chance of loan approval.
* **Income**: Applicant and co-applicant incomes have outliers, which are normalized or combined during feature engineering.

## 7. Feature Engineering
* **TotalIncome**: Created by summing ApplicantIncome and CoapplicantIncome.
* **Income_Loan_Ratio**: Created by dividing TotalIncome by LoanAmount, representing the applicant's ability to repay.

## 8. Model Development
A Scikit-Learn `Pipeline` was utilized to avoid data leakage and streamline inference.
* **Preprocessing**: 
    * `SimpleImputer` for handling missing values (median for numeric, most frequent for categorical).
    * `StandardScaler` for numeric features.
    * `OneHotEncoder` for categorical features.
* **Models Tested**: Logistic Regression, Random Forest, XGBoost.
* **Tuning**: `RandomizedSearchCV` with Stratified 5-Fold CV.

## 9. Evaluation Results
Models were evaluated based on Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
The model with the highest F1 Score and ROC-AUC was automatically selected as the final model. Performance tables, Confusion Matrix, and ROC Curve are saved in the `assets/` directory and viewable in the Streamlit App.

## 10. Conclusion
The pipeline successfully trains a high-performing model for loan prediction. The integration of SHAP values provides necessary explainability, ensuring that stakeholders can understand why an applicant was rejected. The Streamlit app provides an intuitive interface for end-users.

## 11. Future Scope
* Incorporate more advanced techniques for dealing with class imbalance (e.g., SMOTE).
* Retrieve more diverse data to increase robustness.
* Deploy the application to cloud platforms (e.g., AWS, GCP, or Streamlit Cloud).
