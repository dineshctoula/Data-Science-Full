"""
Data Science Project - End-to-End Machine Learning Pipeline
---------------------------------------------------------
This script performs an end-to-end machine learning pipeline for housing price prediction.
The steps included in this pipeline are:
  1. Data Loading & Exploration: Reads the dataset and outputs summary statistics.
  2. Data Visualization: Generates correlation heatmaps and pairplots to understand features.
  3. Data Preprocessing: Handles missing values, removes outliers (using IQR), and scales features.
  4. Model Evaluation: Trains Random Forest and Gradient Boosting regressors.
  5. Hyperparameter Tuning: Uses GridSearchCV to find the optimal model hyperparameters.
  6. Model Persistence: Saves the best model and preprocessing scaler to disk for inference.
  7. Feature Importance: Visualizes which features are most predictive of housing prices.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# =========================
# STEP 1: DATA LOADING & EXPLORATION
# =========================

def load_data(path):
    """Loads CSV dataset and handles exceptions."""
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully\n")
        return df
    except Exception as e:
        print("❌ Error loading dataset:", e)
        return None

def explore_data(df):
    """Prints overview, structure, summary stats, and missing value counts."""
    if df is None:
        return
    print("🔹 First 5 rows:\n", df.head(), "\n")
    print("🔹 Dataset Info:\n")
    df.info()
    print("\n🔹 Summary Statistics:\n", df.describe(), "\n")
    print("🔹 Missing Values:\n", df.isnull().sum(), "\n")

# =========================
# STEP 2: DATA VISUALIZATION
# =========================

def visualize_data(df):
    """Generates correlation heatmap and pairplot for numerical features."""
    print("📊 Generating visualizations...")
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Pairplot for numerical columns
    key_cols = numeric_df.columns[:5] if len(numeric_df.columns) > 5 else numeric_df.columns
    sns.pairplot(df[key_cols])
    plt.suptitle("Pairplot of Key Numerical Features", y=1.02)
    plt.show()

# =========================
# STEP 3: DATA PREPROCESSING
# =========================

def preprocess_data(df, target_col="median_house_value"):
    """
    Cleans dataset, imputes missing values, encodes categorical features,
    splits data into train/test sets, and standardizes features.
    """
    # Drop rows missing the target column
    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Handle Outliers using IQR for numerical features
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = np.clip(X[col], lower_bound, upper_bound)
    print("✅ Outliers handled using IQR method")

    # Impute missing numerical values using median
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    feature_names = X.columns.tolist()

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling - Initialize the StandardScaler
    scaler = StandardScaler()
    # Fit the scaler on training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test data using the fitted scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference in app.py
    import pickle
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("💾 Scaler saved successfully to scaler.pkl")

    print("✅ Data preprocessing completed\n")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

# =========================
# STEP 4: MODEL EVALUATION
# =========================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Predicts test targets and prints MSE, RMSE, and R2 score metrics."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    
    print(f"📊 {model_name} Evaluation Metrics:")
    print(f" - Mean Squared Error (MSE) : {mse:,.2f}")
    print(f" - Root Mean Squared Error  : ${rmse:,.2f}")
    print(f" - R² Score                : {r2:.4f}\n")
    return {"rmse": rmse, "r2": r2}

def save_predictions(model, X_test, y_test, filename="predictions.csv"):
    """Saves the actual vs predicted values to a CSV file."""
    predictions = model.predict(X_test)
    df_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })
    df_results.to_csv(filename, index=False)
    print(f"💾 Predictions saved successfully to {filename}\n")

# =========================
# STEP 5: HYPERPARAMETER TUNING
# =========================

def tune_random_forest(X_train, y_train):
    """Performs grid search to find optimal hyperparameters for Random Forest."""
    print("🔍 Tuning Random Forest Regressor...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("✅ Best Parameters:", grid.best_params_)
    print(f"🔥 Best Cross-Validation R² Score: {grid.best_score_:.4f}\n")
    return grid.best_estimator_

def tune_gradient_boosting(X_train, y_train):
    """Performs grid search to find optimal hyperparameters for Gradient Boosting."""
    print("🔍 Tuning Gradient Boosting Regressor...")
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    gb = GradientBoostingRegressor(random_state=42)
    grid = GridSearchCV(gb, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("✅ Best Parameters:", grid.best_params_)
    print(f"🔥 Best Cross-Validation R² Score: {grid.best_score_:.4f}\n")
    return grid.best_estimator_

def tune_ridge(X_train, y_train):
    """Performs grid search to find optimal hyperparameters for Ridge Regression."""
    print("🔍 Tuning Ridge Regressor...")
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
    ridge = Ridge(random_state=42)
    grid = GridSearchCV(ridge, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("✅ Best Parameters:", grid.best_params_)
    print(f"🔥 Best Cross-Validation R² Score: {grid.best_score_:.4f}\n")
    return grid.best_estimator_


# =========================
# STEP 6: MODEL PERSISTENCE
# =========================

def save_model(model, filename="best_rf_model.pkl"):
    """Serializes and saves trained model to disk using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved successfully to {filename}\n")

def load_model(filename="best_rf_model.pkl"):
    """Loads and returns serialized model from disk."""
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"📂 Model loaded successfully from {filename}\n")
    return model

# =========================
# STEP 7: FEATURE IMPORTANCE
# =========================

def plot_feature_importance(model, feature_names):
    """Plots horizontal bar chart of relative feature importances."""
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Selected model does not support feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    filepath = "housing.csv"
    
    # 1. Load Data
    raw_df = load_data(filepath)
    
    if raw_df is not None:
        # 2. Explore & Visualize
        explore_data(raw_df)
        visualize_data(raw_df)
        
        # 3. Preprocess & Split Data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(raw_df)
        
        # 4. Tune & Train Models
        best_ridge = tune_ridge(X_train, y_train)
        best_rf = tune_random_forest(X_train, y_train)
        best_gb = tune_gradient_boosting(X_train, y_train)
        
        # 5. Evaluate Performance
        evaluate_model(best_ridge, X_test, y_test, model_name="Tuned Ridge")
        evaluate_model(best_rf, X_test, y_test, model_name="Tuned Random Forest")
        evaluate_model(best_gb, X_test, y_test, model_name="Tuned Gradient Boosting")
        
        # 6. Save Predictions for the best model (using Gradient Boosting as an example)
        save_predictions(best_gb, X_test, y_test, "gb_predictions.csv")
        
        # 7. Save & Load Model
        save_model(best_gb, "best_gb_model.pkl")
        loaded_gb = load_model("best_gb_model.pkl")
        
        # 8. Plot Feature Importances
        plot_feature_importance(loaded_gb, feature_names)