import pandas as pd
import os

def inspect_excel_file():
    """Inspect the congress trading excel file"""
    file_path = 'Data/Congressmen/congress-trading-all.xlsx'
    
    print(f"Checking if file exists: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print("\nReading Excel file...")
    df = pd.read_excel(file_path)
    
    print("\nColumns in the file:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData Info:")
    print(df.info())

if __name__ == "__main__":
    inspect_excel_file()