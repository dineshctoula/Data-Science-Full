# =========================
# MAIN ENTRY POINT
# =========================

from src.data_loader import load_data, explore_data
from src.preprocess import clean_data


def main():
    # Load dataset
    df = load_data("data/data.csv")
    
    # Explore dataset
    explore_data(df)

    # Clean dataset
    df = clean_data(df)

    print("\n✅ Data preprocessing completed!")


if __name__ == "__main__":
    main()