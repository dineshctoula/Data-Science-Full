# =========================
# COMMIT 1: DATA LOADING & EXPLORATION
# =========================

import pandas as pd

# Load dataset
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully\n")
        return df
    except Exception as e:
        print("❌ Error:", e)

# Explore dataset
def explore_data(df):
    print("🔹 First 5 rows:\n", df.head(), "\n")
    
    print("🔹 Dataset Info:\n")
    print(df.info(), "\n")
    
    print("🔹 Summary Statistics:\n", df.describe(), "\n")
    
    print("🔹 Missing Values:\n", df.isnull().sum(), "\n")

# Main execution
if __name__ == "__main__":
    df = load_data("housing.csv")
    explore_data(df)