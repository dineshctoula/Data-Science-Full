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





# =========================
# COMMIT 3: MODEL TRAINING
# =========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(data):
    # Assume last column is target
    X = data[:, :-1]
    y = data[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("📊 Model Evaluation:")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    return model