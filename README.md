# Loan Decision Support Platform

This project was developed as a foundational phase of the AIML Summer Internship, focusing on building a robust machine learning pipeline to predict whether a loan application will be approved or rejected based on historical applicant data.

## 🚀 Project Overview
The Loan Decision Support Platform is an interactive web application that provides a complete, end-to-end machine learning workflow. Built using Python, Scikit-Learn, and Streamlit, the platform allows users to explore the underlying dataset, evaluate the performance of various predictive models, and perform real-time loan approval inference through a user-friendly interface.

## 🎯 Project Objectives
- Build a clean and robust Machine Learning application.
- Implement comprehensive data analysis and preprocessing techniques.
- Address dataset class imbalance to ensure fair and accurate predictions.
- Train, evaluate, and compare multiple machine learning algorithms to select the best performing model.
- Deploy the final model through an intuitive web dashboard.

## 🧠 Key Features
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Data Balancing using SMOTE
- Logistic Regression
- Random Forest
- XGBoost
- Model Comparison
- Loan Approval Prediction
- Streamlit Deployment

## 📊 Dataset Description
The model is trained on a banking dataset containing key financial and demographic attributes used to evaluate loan applications:
- **Applicant & Co-applicant Income**: Primary and secondary income sources.
- **Loan Amount & Term**: The principal requested and the duration of the loan.
- **Credit History**: A binary indicator of past credit health (1.0 = Good, 0.0 = Poor).
- **Demographics**: Gender, Marital Status, Dependents, Education, and Property Area.

## 🛠️ Technology Stack
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
- **Data Engineering**: Pandas, NumPy
- **Model Serialization**: Joblib

## 📂 Project Structure
```text
LoanApprovalPrediction/
├── app.py                      # Main Streamlit application entry point
├── config/                     # Configuration and path settings
├── data/                       # Training dataset (train.csv)
├── models/                     # Saved models and evaluation metrics
├── pages/                      # Streamlit dashboard pages (EDA, Prediction, Evaluation)
├── training/
│   └── train_models.py         # Automated pipeline for training models
└── utils/                      # Helper modules (Predictor, UI components)
```

## ⚙️ Data Preprocessing Pipeline
To ensure high model accuracy, the data undergoes strict preprocessing:
1. **Derived Features**: Calculation of `TotalIncome` and `Income_Loan_Ratio`.
2. **Missing Values Handling**: Numeric features are imputed using the median, while categorical features use the most frequent value.
3. **Encoding & Scaling**: Categorical variables are One-Hot Encoded, and numeric features are scaled using `StandardScaler`.

## ⚖️ SMOTE Class Balancing
Historical loan datasets are often heavily skewed towards approvals. To prevent the model from becoming biased against minority classes (rejections), we apply **SMOTE (Synthetic Minority Over-sampling Technique)**. This generates synthetic examples of the minority class, ensuring the models train on a perfectly balanced dataset.

## 🤖 Model Training Pipeline
The training script (`training/train_models.py`) automates the entire learning process:
1. Splits the data into 80% training and 20% testing sets.
2. Applies the preprocessing pipeline and SMOTE.
3. Trains three distinct classifiers:
   - **Logistic Regression** (Baseline)
   - **Random Forest** (Ensemble Bagging)
   - **XGBoost** (Ensemble Boosting)
4. Saves all serialized models to the `models/` directory.
5. Automatically selects the best performing model based on the F1 Score and promotes it to `best_model.pkl` for active use in the web application.

## 📈 Model Evaluation Metrics
Every trained model is evaluated against the unseen 20% test set. The application tracks and displays the following metrics:
- **Accuracy**: The overall percentage of correct predictions.
- **Precision**: The proportion of predicted approvals that were actually correct.
- **Recall**: The ability of the model to find all actual approvals.
- **F1 Score**: The harmonic mean of Precision and Recall, used as the primary metric for model selection.
- **ROC-AUC**: Evaluates the model's ability to distinguish between classes at various threshold settings.
- **Confusion Matrix**: A visual breakdown of True Positives, True Negatives, False Positives, and False Negatives.

## ⚙️ Requirements & Prerequisites
Before starting, ensure you have the following prerequisites:
- **Python Version**: Python 3.10, 3.11, or 3.12 is required.
- **Dataset**: Ensure `data/train.csv` is present in the repository.
- **Model Files**: Inference requires `models/best_model.pkl` and `models/evaluation_metrics.json`. If missing, you must run the training pipeline first.
- **Virtual Environment**: All execution scripts automatically create and utilize a Python virtual environment (`venv`) to prevent global dependency conflicts.

## 📥 Installation & Environment Setup

### Option A: One-Click Startup (Recommended)
We provide automated scripts that handle virtual environment creation, pip upgrades, dependency installation, and application launching in a single command.

**For Windows (CMD/PowerShell):**
```cmd
run_project.bat
```

**For Linux (Ubuntu/Arch):**
```bash
chmod +x run_project.sh
./run_project.sh
```

### Option B: Cross-Platform Python Scripts
Alternatively, you can use the pure Python automated setup utilities:

```bash
# Step 1: Configure the environment and install dependencies
python setup_project.py

# Step 2: Validate the environment and launch Streamlit
python launch.py
```

## 🧠 Training Models
If you need to retrain the models, update features, or if the `models/` directory is empty, run the training pipeline manually. Ensure your virtual environment is activated first:

```bash
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate

# Run the training script
python training/train_models.py
```
This will evaluate Logistic Regression, Random Forest, and XGBoost, saving the best performing model to `models/best_model.pkl`.

## 🩺 Troubleshooting
- **Diagnostic Tool**: If you experience any issues, run `python check_environment.py` to get a detailed PASS/FAIL diagnostic report of your Python version, dataset existence, and dependency health.
- **Missing Module Error (`ModuleNotFoundError`)**: Ensure you are running the application using the automated scripts, or that your virtual environment (`venv`) is activated.
- **Missing Dataset Error**: Verify that `train.csv` exists inside the `data/` directory.
- **Model Loading Error**: If the dashboard fails to load model metrics, ensure `models/best_model.pkl` and `models/evaluation_metrics.json` exist. Run the training script if they are absent.
