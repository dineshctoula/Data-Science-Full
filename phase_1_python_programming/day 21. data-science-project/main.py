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





# =========================
# COMMIT 2: DATA PREPROCESSING
# =========================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # Convert categorical columns
    df = pd.get_dummies(df)

    # Split features and target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("✅ Data preprocessing completed\n")

    return X_train, X_test, y_train, y_test


# Update main
if __name__ == "__main__":
    df = load_data("housing.csv")
    explore_data(df)
    
    X_train, X_test, y_train, y_test = preprocess_data(df)