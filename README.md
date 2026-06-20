# Loan Decision Support Platform

An end-to-end, enterprise-grade AI lending platform built to evaluate, optimize, and explain loan applications using advanced Machine Learning, Explainable AI (XAI), and Business Intelligence.

## 🚀 Overview
The Loan Decision Support Platform is not just a binary "Yes/No" prediction model. It is a massive 10-module intelligence suite designed for loan officers and underwriters. It provides deep financial health analytics, active fraud detection, unsupervised customer segmentation, and an autonomous loan restructuring engine that goal-seeks alternative approval paths for rejected applicants.

## 🧠 Key Features
1. **Core Prediction Engine**: High-accuracy Random Forest classifier evaluating core demographics and financials.
2. **Business Intelligence**: A macro-level executive dashboard aggregating thousands of data points into a singular Portfolio Health Index (PHI).
3. **Explainable AI (SHAP)**: Complete transparency. Every decision is mapped to exact positive and negative driving factors, ensuring regulatory compliance.
4. **AI Loan Optimization**: A "What-If" restructuring engine that automatically searches for lower loan amounts or longer tenures to turn a Rejection into an Approval.
5. **Fraud Risk Assessment**: A dual-layer security net combining strict heuristic rules (e.g., Extreme DTI) with an Isolation Forest detecting historical anomalies.
6. **Customer Segmentation**: Unsupervised K-Means clustering that maps applicants to 5 distinct business personas (e.g., Premium Borrowers, Stable Professionals).
7. **Executive PDF Reporting**: One-click generation of professional underwriting reports summarizing all 10 analytics layers via ReportLab.

## 🛠️ Technology Stack
- **Frontend**: Streamlit, Plotly, Altair
- **Machine Learning**: Scikit-Learn, SHAP, K-Means, Isolation Forest
- **Data Engineering**: Pandas, NumPy
- **Exporting**: ReportLab (PDF)

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/LoanApprovalPrediction.git
cd LoanApprovalPrediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the platform
streamlit run app.py
```

## 🌐 Deployment
Please refer to `DEPLOYMENT.md` for comprehensive instructions on deploying to Streamlit Community Cloud, Docker, or local environments.

## 🏗️ Architecture
Please refer to `ARCHITECTURE.md` for a complete Mermaid flowchart detailing how the 10 intelligence modules interact.

## 🔮 Future Scope
- Integration with live Open Banking APIs for real-time transaction history.
- Transitioning the underlying predictive model to a deep learning architecture (e.g., XGBoost or PyTorch).
- Multi-tenant support for different bank branches.
