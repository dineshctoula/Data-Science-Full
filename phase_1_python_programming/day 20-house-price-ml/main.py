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
    DATA_PATH = "housing.csv"
    df = load_data(DATA_PATH)
    explore_data(df)




# =========================
# COMMIT 2: DATA PREPROCESSING
# =========================

from sklearn.preprocessing import StandardScaler

# Preprocess data
def preprocess_data(df):
    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Convert categorical to numeric
    df = pd.get_dummies(df)

    # Feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    print("✅ Data preprocessing completed\n")
    return scaled_data