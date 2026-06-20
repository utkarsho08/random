import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Application Config
APP_NAME = "Loan Decision Support Platform"
PAGE_ICON = "🏦"
LAYOUT = "wide"

# Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Specific Files
MODEL_PATH = MODELS_DIR / "loan_model.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"

# KPI Constants
KPI_DEFAULTS = {
    "total_applications": 0,
    "approval_rate": 0.0,
    "avg_loan_amount": 0.0,
    "avg_applicant_income": 0.0,
    "avg_credit_history": 0.0
}
