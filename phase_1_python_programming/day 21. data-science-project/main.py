import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# STEP 1: DATA LOADING & EXPLORATION
# =========================

def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully\n")
        return df
    except Exception as e:
        print("❌ Error loading dataset:", e)
        return None

def explore_data(df):
    if df is None:
        return
    print("🔹 First 5 rows:\n", df.head(), "\n")
    print("🔹 Dataset Info:\n")
    df.info()
    print("\n🔹 Summary Statistics:\n", df.describe(), "\n")
    print("🔹 Missing Values:\n", df.isnull().sum(), "\n")

# =========================
# STEP 2: DATA PREPROCESSING
# =========================

def preprocess_data(df, target_col="median_house_value"):
    # 1. Drop rows with missing target values
    df = df.dropna(subset=[target_col]).copy()

    # 2. Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 3. Handle missing values in numerical features
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # 4. Convert categorical variables (returns integers 0/1 instead of booleans)
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Scaling numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("✅ Data preprocessing completed\n")
    return X_train, X_test, y_train, y_test

# =========================
# STEP 3: MODEL TRAINING & EVALUATION
# =========================

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model training completed\n")
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    
    print("📊 Model Evaluation Metrics:")
    print(f" - Mean Squared Error (MSE) : {mse:,.2f}")
    print(f" - Root Mean Squared Error  : ${rmse:,.2f}")
    print(f" - R² Score                : {r2:.4f}\n")

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    filepath = "housing.csv"
    
    df = load_data(filepath)
    if df is not None:
        explore_data(df)
        
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)






# =========================
# COMMIT 4: DATA VISUALIZATION
# =========================

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    print("📊 Generating visualizations...")

    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # Pairplot (for small datasets)
    sns.pairplot(df)
    plt.show()

# Update main
if __name__ == "__main__":
    df = load_data("housing.csv")
    explore_data(df)
    df = preprocess_data(df)
    visualize_data(df)
    train_model(df)






# =========================
# COMMIT 5: ADVANCED MODELS
# =========================

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_advanced_models(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    from sklearn.metrics import mean_squared_error, r2_score

    print("\n🌳 Decision Tree:")
    print("MSE:", mean_squared_error(y_test, dt_pred))
    print("R2:", r2_score(y_test, dt_pred))

    print("\n🌲 Random Forest:")
    print("MSE:", mean_squared_error(y_test, rf_pred))
    print("R2:", r2_score(y_test, rf_pred))