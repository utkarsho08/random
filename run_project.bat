@echo off
echo ============================================================
echo Starting Loan Approval Prediction System (Windows)
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org or the Microsoft Store.
    pause
    exit /b 1
)

REM 1. Create virtual environment if missing
IF NOT EXIST "venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) ELSE (
    echo [INFO] Virtual environment 'venv' already exists.
)

REM 2. Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM 3. Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM 4. Install requirements
IF EXIST "requirements.txt" (
    echo [INFO] Installing requirements...
    pip install -r requirements.txt
) ELSE (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)

REM 5. Launch Streamlit
echo [INFO] Launching application...
streamlit run app.py

pause
