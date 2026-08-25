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

def create_basic_visualizations(df: pd.DataFrame) -> None:
    """
    Creates fundamental Plotly Express visualizations.
    """
    print("[INFO] Creating basic Plotly Express visualizations...")
    import plotly.express as px
    
    # Ensure 'Date' is set for grouping
    df['Date'] = df['Timestamp'].dt.date
    
    # 1. Line Chart: Daily Revenue Trend
    daily_revenue = df.groupby('Date')['Revenue'].sum().reset_index()
    fig_line = px.line(daily_revenue, x='Date', y='Revenue', 
                       title="Daily Total Revenue Trend",
                       markers=True, line_shape="spline")
    fig_line.write_html("output/daily_revenue_trend.html")
    
    # 2. Scatter Plot: Revenue vs Customer Satisfaction by Category
    fig_scatter = px.scatter(df, x='Revenue', y='Customer_Satisfaction',
                             color='Category', size='Quantity',
                             hover_data=['Region'],
                             title="Revenue vs Customer Satisfaction",
                             opacity=0.7)
    fig_scatter.write_html("output/revenue_vs_satisfaction.html")
    print("[SUCCESS] Basic visualizations created and saved to 'output/'.")

if __name__ == "__main__":
    df = generate_e_commerce_data(1000)
    print("\n--- Data Sample ---")
    print(df.head())
    
    os.makedirs("output", exist_ok=True)
    create_basic_visualizations(df)
