# =========================
# MAIN ENTRY POINT
# =========================

from src.data_loader import load_data, explore_data


def main():
    # Load dataset
    df = load_data("data/data.csv")
    
    # Explore dataset
    explore_data(df)


if __name__ == "__main__":
    main()