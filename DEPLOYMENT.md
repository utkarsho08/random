# Deployment Guide

This guide details the three primary methods for deploying the Loan Decision Support Platform into production.

## Option 1: Streamlit Community Cloud (Recommended & Easiest)
Streamlit provides a free, seamless hosting environment directly from your GitHub repository.

1. **Push to GitHub**: Ensure all files, including `requirements.txt` and the `models/` directory, are pushed to your public or private GitHub repository.
2. **Login to Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Deploy App**: Click `New app`.
4. **Configure**:
   - Repository: `your-username/LoanApprovalPrediction`
   - Branch: `main`
   - Main file path: `app.py`
5. **Launch**: Click `Deploy`. Streamlit will automatically install dependencies from `requirements.txt` and host your application securely.

## Option 2: Docker Deployment
For enterprise environments requiring isolation, the application can be containerized.

1. **Create a `Dockerfile`** in the root directory:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build the Image**:
```bash
docker build -t loan-prediction-app .
```

3. **Run the Container**:
```bash
docker run -p 8501:8501 loan-prediction-app
```
4. Access the app at `http://localhost:8501`.

## Option 3: Local/On-Premise Deployment
For internal banking networks.

1. Install Python 3.10+.
2. Clone the repository and navigate to the root directory.
3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
4. Install requirements:
```bash
pip install -r requirements.txt
```
5. Run the application:
```bash
streamlit run app.py
```
*(Optional: Use a process manager like `tmux`, `screen`, or `systemd` to keep the app running continuously on a Linux server).*
