# =========================
# DATA PREPROCESSING
# =========================

def clean_data(df):
    """
    Handle missing values
    """
    print("\n🧹 Cleaning data...")

    # Fill numeric missing values with mean
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Fill categorical missing values with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print("✅ Missing values handled")
    return df