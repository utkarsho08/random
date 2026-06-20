#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================================"
echo "Starting Loan Approval Prediction System (Linux)"
echo "============================================================"

# Determine Python command (Ubuntu uses python3, Arch often maps python directly to Python 3)
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Ubuntu: sudo apt install python3 python3-venv"
    echo "Arch Linux: sudo pacman -S python"
    exit 1
fi

# 1. Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment 'venv'..."
    if ! $PYTHON_CMD -m venv venv; then
        echo "[ERROR] Failed to create virtual environment."
        echo "If you are on Ubuntu/Debian, you may need to install the venv package:"
        echo "sudo apt update && sudo apt install python3-venv"
        exit 1
    fi
else
    echo "[INFO] Virtual environment 'venv' already exists."
fi

# 2. Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip
echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

# 4. Install requirements
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing requirements..."
    pip install -r requirements.txt
else
    echo "[ERROR] requirements.txt not found!"
    exit 1
fi

# 5. Launch Streamlit
echo "[INFO] Launching application..."
streamlit run app.py
