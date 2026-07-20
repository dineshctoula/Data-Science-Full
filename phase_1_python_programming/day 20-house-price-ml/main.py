# =========================
# COMMIT 1: DATA LOADING & EXPLORATION
# =========================

import pandas as pd

# Load dataset
def load_data(path):
    """
    Load dataset from CSV file.
    """
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully.\n")
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        raise


# Explore dataset
def explore_data(df):
    """
    Perform basic exploratory analysis.
    """
    print("🔍 First 5 rows:")
    print(df.head())

    print("\n📊 Dataset Info:")
    print(df.info())

    print("\n📈 Summary Statistics:")
    print(df.describe())

    print("\n❗ Missing Values:")
    print(df.isnull().sum())


# Entry point
if __name__ == "__main__":
    DATA_PATH = "housing.csv"

    df = load_data(DATA_PATH)
    explore_data(df)