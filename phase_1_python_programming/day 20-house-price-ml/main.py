# =========================
# FINAL VERSION (AFTER 3 COMMITS)
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

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = prepare_features(df)

    model, X_test, y_test = train_model(X, y)

    evaluate_model(model, X_test, y_test)