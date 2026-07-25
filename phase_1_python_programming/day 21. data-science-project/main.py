import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

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
# STEP 2: DATA VISUALIZATION
# =========================

def visualize_data(df):
    print("📊 Generating visualizations...")
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    key_cols = numeric_df.columns[:5] if len(numeric_df.columns) > 5 else numeric_df.columns
    sns.pairplot(df[key_cols])
    plt.show()

# =========================
# STEP 3: DATA PREPROCESSING
# =========================

def preprocess_data(df, target_col="median_house_value"):
    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    X = pd.get_dummies(X, drop_first=True, dtype=int)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Data preprocessing completed\n")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

# =========================
# STEP 4: MODEL TRAINING & EVALUATION
# =========================

def evaluate_predictions(y_test, predictions, model_name):
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    
    print(f"📊 {model_name} Evaluation Metrics:")
    print(f" - Mean Squared Error (MSE) : {mse:,.2f}")
    print(f" - Root Mean Squared Error  : ${rmse:,.2f}")
    print(f" - R² Score                : {r2:.4f}\n")

# =========================
# STEP 5: HYPERPARAMETER TUNING
# =========================

def tune_random_forest(X_train, y_train):
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

# =========================
# STEP 6: MODEL PERSISTENCE
# =========================

def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved successfully to {filename}\n")

def load_model(filename="model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"📂 Model loaded successfully from {filename}\n")
    return model

# =========================
# STEP 7: FEATURE IMPORTANCE
# =========================

def plot_feature_importance(model, feature_names):
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
    
    # 1. Load & Explore
    raw_df = load_data(filepath)
    if raw_df is not None:
        explore_data(raw_df)
        visualize_data(raw_df)
        
        # 2. Preprocess & Split
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(raw_df)
        
        # 3. Hyperparameter Tuning
        best_rf = tune_random_forest(X_train, y_train)
        
        # 4. Evaluate Tuned Model
        preds = best_rf.predict(X_test)
        evaluate_predictions(y_test, preds, "Tuned Random Forest")
        
        # 5. Model Persistence
        save_model(best_rf, "best_rf_model.pkl")
        loaded_rf = load_model("best_rf_model.pkl")
        
        # 6. Feature Importance
        plot_feature_importance(loaded_rf, feature_names)