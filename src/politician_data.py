import pandas as pd


file_path = "/Users/aaronmasih/Trade-Outcome-Prediction/Data/Congressmen/congress-trading-all.xlsx" 
df = pd.read_excel(file_path)

# List of politicians to include
politicians = ["Austin Scott", "French Hill", "John Curtis", "Bob Gibbs", "Nancy Pelosi"]

# Filtered data for the specified politicians and the years 2014-2021
filtered_df = df[
    (df['Name'].isin(politicians)) & 
    (df['Traded'].between("2014-01-01", "2021-12-31"))
]


output_file_path = "/Users/aaronmasih/Trade-Outcome-Prediction/Data/Congressmen/filtered_trades_2014_2021.csv"  
filtered_df.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")