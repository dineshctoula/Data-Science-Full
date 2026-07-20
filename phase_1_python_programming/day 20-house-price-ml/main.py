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






# =========================
# COMMIT 2: PREPROCESSING
# =========================

import pandas as pd

# Load dataset
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully.\n")
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        raise


# Explore dataset
def explore_data(df):
    print("🔍 First 5 rows:")
    print(df.head())

    print("\n❗ Missing Values:")
    print(df.isnull().sum())


# Clean data
def clean_data(df):
    """
    Handle missing values.
    """
    df = df.copy()

    # Fill numeric missing values with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    print("\n✅ Missing values handled.\n")
    return df


# Prepare features
def prepare_features(df):
    """
    Separate features (X) and target (y).
    """
    X = df[['area', 'bedrooms']]   # input features
    y = df['price']                # target

    print("✅ Features and target prepared.\n")
    return X, y


# Entry point
if __name__ == "__main__":
    DATA_PATH = "housing.csv"

    df = load_data(DATA_PATH)
    explore_data(df)

    df = clean_data(df)
    X, y = prepare_features(df)

    print("📌 Feature Sample:")
    print(X.head())

    print("\n🎯 Target Sample:")
    print(y.head())