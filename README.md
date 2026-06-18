# Loan Approval Prediction System

## Project Overview
The **Loan Approval Prediction System** is a complete end-to-end Machine Learning Capstone Project designed to automate the loan eligibility process based on customer details. It features a fully reproducible data processing and model training pipeline, alongside an interactive Streamlit web application.

## Dataset Description
This project uses the Kaggle Loan Prediction Dataset.
* **Target Variable**: `Loan_Status` (Y/N)
* **Features**: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area.

## Project Structure
* `Dataset/`: Contains the `train.csv` file.
* `Notebook/`: Contains the `loan_prediction.ipynb` notebook which contains the EDA, Feature Engineering, and Model Training steps.
* `Model/`: Contains the `loan_model.pkl` pipeline.
* `Streamlit_App/`: Contains the Streamlit app `app.py`.
* `Documentation/`: Contains the Project Report.
* `assets/`: Contains generated visualizations (Heatmap, ROC Curve, Feature Importance, SHAP Summary, etc.).

## Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
4. Install the requirements: `pip install -r requirements.txt`

## Running Instructions
### To retrain the model and generate the notebook:
Execute the notebook using Jupyter or nbconvert:
```bash
jupyter nbconvert --to notebook --execute Notebook/loan_prediction.ipynb --inplace
```

### To run the Streamlit App:
```bash
streamlit run Streamlit_App/app.py
```

## Model Information
We tested Logistic Regression, Random Forest, and XGBoost. The pipeline handles missing values using SimpleImputer and encodes variables using OneHotEncoder and StandardScaler. Models were evaluated using 5-Fold Stratified K-Fold CV and RandomizedSearchCV.
The best model was chosen automatically based on F1 Score and ROC-AUC.

## Streamlit Usage
1. **Prediction Tab**: Enter applicant details to get a real-time prediction on loan approval, along with probability and confidence score.
2. **Explainability Tab**: View the SHAP (SHapley Additive exPlanations) values to understand the reasoning behind the specific prediction for the last applicant.
3. **Dataset Insights**: View correlation heatmaps, feature importance, and ROC curves generated during the training phase.

## Results Summary
The system provides a robust and fair prediction mechanism, backed by explainable AI (SHAP), allowing stakeholders to understand why a loan was approved or rejected.
# random
