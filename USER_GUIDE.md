# End-User Guide: Loan Decision Support Platform

Welcome to the AI Lending Platform. This guide is designed for Loan Officers and Underwriters to navigate the 10-module intelligence suite.

## The Workflow

### 1. Macro Analysis (Start of Day)
**Business Intelligence**: Begin your day here. This dashboard provides a macro-view of the entire lending portfolio. You can monitor the overarching Portfolio Health Index (PHI), track the live Approval Rate, and identify demographic bottlenecks using the interactive Plotly funnels.

### 2. Processing a New Applicant
**Loan Prediction**: When a new applicant files a request, enter their specific demographic and financial metrics here. 
- Click **Predict Loan Status**. 
- The system will output an immediate "Approved" or "Rejected" binary decision, alongside a percentage Confidence Score.

### 3. Deep Dive Analytics
Once the baseline prediction is complete, navigate through the analytical modules to understand *why* the decision was made.

- **AI Decision Intelligence**: Provides extreme transparency using SHAP values. It explicitly lists the top 3 positive and negative factors driving the prediction (e.g., "The loan was rejected primarily due to a 35% negative drag from a massive loan request").
- **Financial Health**: Evaluates the applicant's Debt-to-Income (DTI) ratio. 
- **Risk Analytics**: Scores the demographic and credit risk of the applicant on a 0-100 scale.
- **Fraud Risk Assessment**: Automatically flags the application if it triggers heuristic rules (e.g., requesting a $1,000,000 loan with $0 income) or mathematically deviates from historical portfolio norms using Isolation Forest anomaly detection.
- **Customer Segmentation**: Instantly maps the applicant to 1 of 5 behavioral personas (e.g., "Stable Professionals" or "High-Risk Borrowers").

### 4. Rescuing Rejected Applicants
If an applicant is rejected but you wish to retain their business, use the optimization tools:
- **What-If Simulator**: Manually tweak the applicant's Income, Loan Amount, or Tenure. Watch the gauges update in real-time to find the exact breakpoint where a Rejection turns into an Approval.
- **AI Loan Optimization**: Click this to have the AI do the work for you. It will automatically goal-seek and recommend 3 alternative loan structures (e.g., "Reduce Loan by $25,000" or "Extend Tenure by 10 Years") that guarantee an >80% approval probability.

### 5. Finalizing the Underwriting
Once you are satisfied with the evaluation or restructuring, generate the final documentation.
- **PDF Report Generation**: On the Prediction, Intelligence, or Optimization pages, click the `Download Report` button. This will instantly compile a professional ReportLab PDF containing all the analytics, visual scores, and AI recommendations, ready to be attached to the applicant's formal file.
