import os
import sys
import platform
import subprocess
import venv
from pathlib import Path

def main():
    print("=" * 60)
    print("Environment Setup: Loan Approval Prediction System")
    print("=" * 60)
    
    # 1. Detect Operating System
    os_name = platform.system()
    print(f"[INFO] Detected Operating System: {os_name}")
    
    # 2. Verify Python Version
    version_info = sys.version_info
    if version_info.major != 3 or version_info.minor not in [10, 11, 12]:
        print(f"[ERROR] Unsupported Python version: {version_info.major}.{version_info.minor}")
        print("Please use Python 3.10, 3.11, or 3.12.")
        sys.exit(1)
    else:
        print(f"[INFO] Python Version OK: {version_info.major}.{version_info.minor}.{version_info.micro}")
        
    # 3. Create virtual environment named venv
    venv_dir = Path("venv")
    if not venv_dir.exists():
        print(f"[INFO] Creating virtual environment '{venv_dir}'...")
        try:
            venv.create(venv_dir, with_pip=True)
            print("[SUCCESS] Virtual environment created.")
        except Exception as e:
            print(f"[ERROR] Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print(f"[INFO] Virtual environment '{venv_dir}' already exists.")
        
    # Get executable paths
    if os_name == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
        
    if not python_exe.exists():
        print(f"[ERROR] Virtual environment python executable not found at {python_exe}")
        sys.exit(1)
        
    # 4. Upgrade pip
    print("\n[INFO] Upgrading pip...")
    try:
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True, stdout=subprocess.DEVNULL)
        print("[SUCCESS] pip upgraded.")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Failed to upgrade pip: {e}")
        
    # 5. Install requirements.txt
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("[ERROR] requirements.txt not found in the current directory.")
        sys.exit(1)
        
    print(f"\n[INFO] Installing dependencies from {req_file}...")
    try:
        # We don't hide stdout here so the user can see the progress of the installation
        subprocess.run([str(python_exe), "-m", "pip", "install", "-r", str(req_file)], check=True)
        print("[SUCCESS] Dependencies installed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        sys.exit(1)
        
    # 6. Verify successful installation
    packages_to_verify = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("imbalanced-learn", "imblearn"),
        ("plotly", "plotly"),
        ("joblib", "joblib")
    ]
    
    print("\n[INFO] Verifying installed packages...")
    missing_packages = []
    for pkg_name, import_name in packages_to_verify:
        try:
            subprocess.run(
                [str(python_exe), "-c", f"import {import_name}"], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            print(f"  - {pkg_name}: OK")
        except subprocess.CalledProcessError:
            print(f"  - {pkg_name}: MISSING or ERROR")
            missing_packages.append(pkg_name)
            
    if missing_packages:
        print(f"[ERROR] The following packages failed to verify: {', '.join(missing_packages)}")
    else:
        print("[SUCCESS] All required packages verified successfully.")
        
    # 7. Verify existence of models
    print("\n[INFO] Verifying project assets...")
    models = ["models/best_model.pkl", "models/evaluation_metrics.json"]
    missing_models = []
    for m in models:
        if Path(m).exists():
            print(f"  - {m}: FOUND")
        else:
            print(f"  - {m}: MISSING")
            missing_models.append(m)
            
    if missing_models:
        print(f"\n[WARNING] The following files are missing: {', '.join(missing_models)}")
        print("          You may need to run the training script before launching the app.")
    else:
        print("[SUCCESS] All model assets found.")
        
    # 8. Display clear success/failure summary
    print("\n" + "=" * 60)
    if missing_packages:
        print("[RESULT] SETUP COMPLETED WITH ERRORS.")
        print("Please review the logs above and fix missing dependencies.")
        sys.exit(1)
    elif missing_models:
        print("[RESULT] SETUP SUCCESSFUL (WITH WARNINGS).")
        print("Dependencies are installed, but model files are missing.")
        print("Run `python training/train_models.py` before launching.")
    else:
        print("[RESULT] SETUP SUCCESSFUL.")
        print("The environment is fully configured and ready.")
    print("=" * 60)

if __name__ == "__main__":
    main()
