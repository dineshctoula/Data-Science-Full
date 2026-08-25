"""
===============================================================================
DAY 30: INTERACTIVE DATA VISUALIZATION DASHBOARDS WITH PLOTLY
===============================================================================
Author: Dinesh (100-Day Data Science Challenge)
Phase: Phase 2 - Data Manipulation & Visualization
Topic: Interactive Dashboards, Plotly Express, Subplots, Updatemenus
===============================================================================
"""

import os
import numpy as np
import pandas as pd

def generate_e_commerce_data(rows: int = 1000) -> pd.DataFrame:
    """
    Generates a synthetic e-commerce sales dataset for interactive visualization.
    """
    print(f"[INFO] Generating {rows} rows of synthetic e-commerce data...")
    np.random.seed(42)
    
    # Create hourly data over a period
    dates = pd.date_range(start="2025-01-01", periods=rows, freq="h")
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'Timestamp': dates,
        'Category': np.random.choice(categories, rows),
        'Region': np.random.choice(regions, rows),
        'Revenue': np.random.normal(loc=150, scale=40, size=rows).round(2),
        'Quantity': np.random.randint(1, 10, size=rows),
        'Customer_Satisfaction': np.random.normal(loc=4.2, scale=0.5, size=rows).clip(1, 5).round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce time-of-day trend
    df['Revenue'] = df['Revenue'] + (df['Timestamp'].dt.hour * 5)
    
    print("[SUCCESS] Data generation complete.")
    return df

if __name__ == "__main__":
    df = generate_e_commerce_data(1000)
    print("\n--- Data Sample ---")
    print(df.head())
