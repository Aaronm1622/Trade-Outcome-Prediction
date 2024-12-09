import os
import sys
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory"""
    # If running from tests directory, go up one level
    current_file = Path(__file__).resolve()
    if current_file.parent.name == 'tests':
        return current_file.parent.parent
    return current_file.parent

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("All required packages successfully imported!")
    except ImportError as e:
        print(f"Error importing packages: {e}")
        return False
    return True

def test_data_access():
    """Test that data files can be accessed"""
    project_root = get_project_root()
    
    data_files = [
        'Data/Congressmen/congress-trading-all.xlsx',
        'Data/archive/spy.csv',
        'Data/archive/Amazon.csv',
        'Data/archive/Apple.csv'
    ]
    
    print(f"\nChecking for data files from project root: {project_root}")
    
    for file in data_files:
        full_path = project_root / file
        if not full_path.exists():
            print(f"Error: Cannot find {file}")
            print(f"Tried to access: {full_path}")
            return False
        else:
            print(f"Found: {file}")
    
    print("\nAll data files successfully located!")
    return True

if __name__ == "__main__":
    print("Testing setup...")
    imports_ok = test_imports()
    data_ok = test_data_access()
    
    if imports_ok and data_ok:
        print("\nSetup complete! You can now run the main analysis.")
    else:
        print("\nSetup incomplete. Please fix the errors above before proceeding.")