import os
import sys
import platform
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("Launching Loan Approval Prediction System")
    print("=" * 60)
    
    # 1. Detect Operating System
    os_name = platform.system()
    
    # 2. Locate virtual environment Python executable
    venv_dir = Path("venv")
    if os_name == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        
    # 3. Verify environment exists
    if not venv_dir.exists() or not python_exe.exists():
        print("[ERROR] Virtual environment not found or incomplete.")
        print("Please run `python setup_project.py` first to configure the environment.")
        sys.exit(1)
        
    # Display helpful error message if dependencies are missing
    try:
        # Quick check for critical dependencies
        subprocess.run(
            [str(python_exe), "-c", "import streamlit, pandas, xgboost, sklearn"], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print("[ERROR] Required dependencies are missing in the virtual environment.")
        print("Please run `python setup_project.py` to install them.")
        sys.exit(1)
        
    # 4. Launch streamlit run app.py
    app_file = Path("app.py")
    if not app_file.exists():
        print("[ERROR] app.py not found in the current directory.")
        sys.exit(1)
        
    print("[INFO] Virtual environment validated.")
    print("[INFO] Starting Streamlit server...")
    print("[INFO] Press Ctrl+C to stop the server.\n")
    
    try:
        # Use 'python -m streamlit run' to ensure it uses the venv interpreter
        subprocess.run([str(python_exe), "-m", "streamlit", "run", str(app_file)])
    except KeyboardInterrupt:
        print("\n[INFO] Streamlit server stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred while running the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
