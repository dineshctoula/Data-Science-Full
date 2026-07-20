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







# =========================
# COMMIT 3: MODEL TRAINING & EVALUATION
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Load dataset
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully.\n")
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        raise


# Clean data
def clean_data(df):
    df = df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


# Prepare features
def prepare_features(df):
    X = df[['area', 'bedrooms']]
    y = df['price']
    return X, y


# Train model
def train_model(X, y):
    """
    Train Linear Regression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("✅ Model trained successfully.\n")
    return model, X_test, y_test


# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    print("📊 Model Evaluation:")
    print(f"Mean Absolute Error: {mae:.2f}")

    print("\n🔮 Sample Predictions:")
    print(predictions[:5])


# Entry point
if __name__ == "__main__":
    DATA_PATH = "housing.csv"

    # Pipeline
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = prepare_features(df)

    model, X_test, y_test = train_model(X, y)

    evaluate_model(model, X_test, y_test)