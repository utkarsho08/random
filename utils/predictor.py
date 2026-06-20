import os
import joblib
import pandas as pd
import numpy as np
from config.settings import MODEL_PATH, BEST_MODEL_PATH
from utils.logger import logger

class ModelPredictor:
    """Centralized utility for model loading and prediction."""
    
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelPredictor, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        """Loads the best model pipeline, falling back to loan_model.pkl if missing."""
        if self._model is None:
            try:
                if os.path.exists(BEST_MODEL_PATH):
                    self._model = joblib.load(BEST_MODEL_PATH)
                    logger.info("Best Model loaded successfully.")
                elif os.path.exists(MODEL_PATH):
                    self._model = joblib.load(MODEL_PATH)
                    logger.info("Fallback loan_model.pkl loaded successfully.")
                else:
                    logger.warning(f"No model files found.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return self._model

    def predict(self, input_df):
        """Runs prediction and returns status, probability, and risk level."""
        model = self.load_model()
        if model is None:
            raise ValueError("Model is not loaded or unavailable.")
            
        try:
            # We must apply feature engineering if the model expects it, but our training script already uses the raw columns.
            # However, in train_models.py we engineered TotalIncome, LoanAmount_Numeric, Income_Loan_Ratio.
            # So we must compute them here before prediction.
            
            df = input_df.copy()
            if 'TotalIncome' not in df.columns:
                df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome'].fillna(0)
            if 'LoanAmount_Numeric' not in df.columns:
                df['LoanAmount_Numeric'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
            if 'Income_Loan_Ratio' not in df.columns:
                df['Income_Loan_Ratio'] = df['TotalIncome'] / df['LoanAmount_Numeric']
                
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
            
            result = {
                "approved": bool(prediction == 1),
                "probability": probability,
                "confidence": max(probability, 1 - probability)
            }
            logger.info(f"Prediction successful: {result['approved']}")
            return result
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
