import pandas as pd


def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully\n")
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        raise


def explore_data(df):
    print("🔍 First 5 rows:\n", df.head())

    print("\n📊 Dataset Info:")
    df.info()

    print("\n📈 Statistical Summary:\n", df.describe())

    print("\n❓ Missing Values:\n", df.isnull().sum())