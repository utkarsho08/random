import sys
from pathlib import Path

def print_result(check_name, status, details=""):
    status_str = "[PASS]" if status else "[FAIL]"
    print(f"{status_str} {check_name:<45} {details}")

def main():
    print("=" * 70)
    print("Environment Diagnostic Utility")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # 1. Python version
    version = sys.version_info
    v_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.major == 3 and version.minor in [10, 11, 12]:
        print_result("Python Version (3.10, 3.11, 3.12)", True, f"Found {v_str}")
    else:
        print_result("Python Version (3.10, 3.11, 3.12)", False, f"Found {v_str}")
        all_passed = False
        
    # 2. Virtual environment status
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print_result("Virtual Environment Active", True, sys.prefix)
    else:
        print_result("Virtual Environment Active", False, "Not running in a venv")
        all_passed = False
        
    # 3. Dataset exists
    dataset_path = Path("data/train.csv")
    if dataset_path.exists():
        print_result("Dataset: data/train.csv", True)
    else:
        print_result("Dataset: data/train.csv", False, "Missing dataset file")
        all_passed = False
        
    # 4. Models exist
    model_path = Path("models/best_model.pkl")
    if model_path.exists():
        print_result("Model: models/best_model.pkl", True)
    else:
        print_result("Model: models/best_model.pkl", False, "Missing model file")
        all_passed = False
        
    metrics_path = Path("models/evaluation_metrics.json")
    if metrics_path.exists():
        print_result("Metrics: models/evaluation_metrics.json", True)
    else:
        print_result("Metrics: models/evaluation_metrics.json", False, "Missing metrics file")
        all_passed = False
        
    # 5. Required libraries import successfully
    packages = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "xgboost": "xgboost",
        "imbalanced-learn": "imblearn",
        "plotly": "plotly",
        "joblib": "joblib"
    }
    
    for pkg_name, import_name in packages.items():
        try:
            __import__(import_name)
            print_result(f"Library: {pkg_name}", True)
        except ImportError as e:
            print_result(f"Library: {pkg_name}", False, f"Import error: {e}")
            all_passed = False
            
    print()
    print("=" * 70)
    if all_passed:
        print("OVERALL STATUS: PASS")
        print("The environment is fully configured and ready to run.")
    else:
        print("OVERALL STATUS: FAIL")
        print("Please resolve the failing checks before launching the app.")
    print("=" * 70)

if __name__ == "__main__":
    main()
