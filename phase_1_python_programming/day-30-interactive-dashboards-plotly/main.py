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

def create_advanced_visualizations(df: pd.DataFrame) -> None:
    """
    Creates advanced interactive charts using Plotly Express and Graph Objects.
    """
    print("[INFO] Creating advanced Plotly visualizations (Sunburst Chart)...")
    import plotly.express as px
    import plotly.graph_objects as go
    
    # 3. Sunburst Chart: Hierarchical View of Revenue by Region and Category
    # We aggregate revenue for the hierarchy
    hierarchy_df = df.groupby(['Region', 'Category'])['Revenue'].sum().reset_index()
    fig_sunburst = px.sunburst(hierarchy_df, path=['Region', 'Category'], values='Revenue',
                               title="Revenue Distribution by Region and Category",
                               color='Revenue',
                               color_continuous_scale='Blues')
    fig_sunburst.write_html("output/revenue_sunburst.html")
    print("[SUCCESS] Advanced visualizations created and saved to 'output/'.")

def create_interactive_dashboard(df: pd.DataFrame) -> None:
    """
    Creates an interactive Plotly Figure with dropdowns (updatemenus) to toggle data views.
    """
    print("[INFO] Creating interactive dashboard with dropdowns...")
    import plotly.graph_objects as go
    
    # 4. Interactive Dropdown: Revenue by Region
    regions = df['Region'].unique()
    fig_interactive = go.Figure()
    
    for region in regions:
        region_data = df[df['Region'] == region].groupby('Date')['Revenue'].sum().reset_index()
        fig_interactive.add_trace(
            go.Scatter(x=region_data['Date'], y=region_data['Revenue'],
                       mode='lines+markers', name=region, visible=False)
        )
        
    # Make first trace visible
    fig_interactive.data[0].visible = True
    
    # Create dropdown buttons
    buttons = []
    for i, region in enumerate(regions):
        visibility = [False] * len(regions)
        visibility[i] = True
        button = dict(
            label=region,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"Daily Revenue Trend: {region} Region"}]
        )
        buttons.append(button)
        
    # Add 'All Regions' button
    all_visibility = [True] * len(regions)
    buttons.append(dict(label="All Regions", method="update",
                        args=[{"visible": all_visibility},
                              {"title": "Daily Revenue Trend: All Regions"}]))
    
    fig_interactive.update_layout(
        updatemenus=[
            dict(active=0, buttons=buttons, x=1.15, xanchor="right", y=1, yanchor="top")
        ],
        title=f"Daily Revenue Trend: {regions[0]} Region"
    )
    fig_interactive.write_html("output/interactive_dropdown_dashboard.html")
    print("[SUCCESS] Interactive dashboard created and saved to 'output/'.")

if __name__ == "__main__":
    df = generate_e_commerce_data(1000)
    print("\n--- Data Sample ---")
    print(df.head())
    
    os.makedirs("output", exist_ok=True)
    create_basic_visualizations(df)
    create_advanced_visualizations(df)
    create_interactive_dashboard(df)
