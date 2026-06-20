import streamlit as st
from config.settings import APP_NAME
from utils.ui import render_header, init_session_state

st.set_page_config(page_title=f"About Project | {APP_NAME}", layout="wide")
init_session_state()

render_header("About Project", "AIML Summer Internship Loan Approval Prediction System")

st.markdown("""
### Project Overview
The Loan Approval Prediction System is a machine learning-powered web application designed to automate and augment the decision-making process for personal and business loans. By analyzing historical applicant data, the system predicts the probability of an applicant defaulting or successfully repaying their loan, thereby classifying their application as 'Approved' or 'Rejected'.

### Objectives
- Build a complete, end-to-end machine learning pipeline from raw data to a deployed web interface.
- Address real-world data challenges such as missing values and severe class imbalance using SMOTE.
- Train, evaluate, and compare multiple predictive algorithms (Logistic Regression, Random Forest, XGBoost) to find the optimal solution.
- Provide a seamless, interactive user experience for bank personnel to input data and receive instant ML inference.

### Dataset Summary
The model is trained on a synthetic banking dataset containing key financial and demographic attributes:
- **Financial Features**: Applicant Income, Co-applicant Income, Loan Amount, Loan Term.
- **Credit Profile**: Credit History (binary indicator of past credit health).
- **Demographics**: Gender, Marital Status, Dependents, Education, and Property Area.

### Technology Stack
- **Language**: Python 3.10+
- **Frontend / Framework**: Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Plotly Express, Plotly Graph Objects

### Machine Learning Models
Three diverse algorithms were trained and evaluated to establish a robust baseline:
1. **Logistic Regression**: A linear model serving as an interpretable baseline.
2. **Random Forest**: An ensemble method to capture non-linear relationships and interactions.
3. **XGBoost**: A highly optimized gradient boosting algorithm designed for extreme performance.

### Workflow Diagram
1. **Data Preprocessing**: Handling missing values (median/mode imputation), scaling numerics, and one-hot encoding categoricals.
2. **SMOTE Balancing**: Generating synthetic data points for the minority class (Rejected applications) to prevent model bias.
3. **Model Training & Evaluation**: Training models and evaluating them on Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
4. **Inference**: Loading the best-performing pipeline serialized as `.pkl` to process live user inputs.

### Results Summary
By utilizing SMOTE and comparing algorithms, the pipeline effectively mitigates bias towards the majority class. The best model is serialized and automatically deployed into the Loan Prediction module, demonstrating a complete CI/CD-style machine learning lifecycle.
""")
