import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    
    # Select numeric columns to avoid string conversion errors
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Pairplot (for subset of key numerical columns to keep render fast)
    key_cols = numeric_df.columns[:5] if len(numeric_df.columns) > 5 else numeric_df.columns
    sns.pairplot(df[key_cols])
    plt.title("Pairplot of Key Features")
    plt.show()

# =========================
# STEP 3: DATA PREPROCESSING
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

    # 4. Convert categorical variables to binary indicators (0/1)
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Scaling numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Data preprocessing completed\n")
    return X_train_scaled, X_test_scaled, y_train, y_test

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

def train_and_evaluate_all_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=100)
    }

    for name, model in models.items():
        print(f"🚀 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        evaluate_predictions(y_test, preds, name)

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    filepath = "housing.csv"
    
    # 1. Load Data
    raw_df = load_data(filepath)
    
    if raw_df is not None:
        # 2. Explore & Visualize Raw Data
        explore_data(raw_df)
        visualize_data(raw_df)
        
        # 3. Preprocess & Split Data
        X_train, X_test, y_train, y_test = preprocess_data(raw_df)
        
        # 4. Train & Compare Models
        train_and_evaluate_all_models(X_train, X_test, y_train, y_test)
















# =========================
# COMMIT 6: HYPERPARAMETER TUNING
# =========================

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }

    rf = RandomForestRegressor()

    grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2')
    grid.fit(X_train, y_train)

    print("✅ Best Parameters:", grid.best_params_)
    print("🔥 Best Score:", grid.best_score_)

    return grid.best_estimator_







# =========================
# COMMIT 7: SAVE & LOAD MODEL
# =========================

import pickle

def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print("💾 Model saved successfully")

def load_model(filename="model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print("📂 Model loaded successfully")
    return model








# =========================
# COMMIT 8: FEATURE IMPORTANCE
# =========================

import matplotlib.pyplot as plt

def feature_importance(model, df):
    importances = model.feature_importances_
    features = df.columns[:-1]

    plt.figure()
    plt.barh(features, importances)
    plt.title("Feature Importance")
    plt.show()