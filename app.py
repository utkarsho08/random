import streamlit as st
from config.settings import APP_NAME, PAGE_ICON, LAYOUT
from utils.ui import init_session_state, render_header

st.set_page_config(page_title=APP_NAME, page_icon=PAGE_ICON, layout=LAYOUT)

def main():
    init_session_state()
    render_header("AIML Summer Internship", "Loan Approval Prediction System")
    
    st.markdown("""
    ## Project Overview
    Welcome to the Loan Approval Prediction System! This project is developed as a foundational phase of the AIML Summer Internship. 
    The core objective is to build a machine learning pipeline that can predict whether a loan application will be approved or rejected based on historical applicant data.
    
    ## Internship Objective
    Build a clean, robust, and production-ready Machine Learning application using Python, Scikit-Learn, and Streamlit. This acts as the baseline before integrating more advanced analytical modules.
    
    ## Dataset Description
    The model is trained on a synthetic banking dataset containing key financial and demographic attributes:
    - **Applicant Income**: Primary income of the applicant.
    - **Co-applicant Income**: Secondary income supporting the application.
    - **Loan Amount**: The total principal requested.
    - **Loan Term**: The duration of the loan in months.
    - **Credit History**: A binary indicator of past credit health (1.0 = Good, 0.0 = Poor).
    - **Demographics**: Gender, Marital Status, Dependents, Education, and Property Area.
    
    ## Workflow Summary
    1. **Data Preprocessing**: Handling missing values, scaling numerics, and encoding categoricals.
    2. **Model Training**: Utilizing SMOTE for class balancing and evaluating Logistic Regression, Random Forest, and XGBoost.
    3. **Inference (Loan Prediction)**: A user-friendly interface to test the model with new, unseen applicant data.
    4. **Dataset Insights**: Visualizing the underlying data distributions.
    
    ## Best Model Summary
    """)
    import os
    import json
    from config.settings import MODELS_DIR
    
    metrics_path = os.path.join(MODELS_DIR, 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            best = data.get('best_model', 'N/A')
            metrics = data.get('metrics', {}).get(best, {})
            acc = metrics.get('accuracy', 0) * 100
            f1 = metrics.get('f1_score', 0) * 100
            st.markdown(f"🏆 **Currently Active Model**: `{best}` (Accuracy: **{acc:.2f}%** | F1 Score: **{f1:.2f}%**)")
            st.markdown("The system automatically selected this model during the evaluation phase to provide the highest accuracy and recall for loan predictions.")
    
    st.markdown("""
    ---
    **Please use the sidebar to navigate to the Loan Prediction module to test the model!**
    """)

if __name__ == "__main__":
    main()
