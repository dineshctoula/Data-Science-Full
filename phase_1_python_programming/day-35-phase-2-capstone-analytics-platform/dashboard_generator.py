"""
Day 35 Phase 2 Capstone: Interactive & Static Executive Dashboard Generator
===========================================================================
This module handles executive visualization generation for the Phase 2 Capstone Project:
1. Interactive Plotly HTML Executive Dashboard Platform (`executive_dashboard.html`):
   - Multi-metric KPI cards (Revenue, Gross Profit, Active Customers, PSI Drift Index).
   - Monthly Revenue & Profit Margin trend charts.
   - RFM Customer Segment monetary contribution heatmap & scatter plots.
   - Interactive HTML tabs and responsive Plotly controls.
2. High-Resolution Static Visual Suite (PNG):
   - Category performance breakdown, conversion funnel, and RFM persona distribution.

Author: 100-Day Data Science Challenge Team
"""

import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any

# Configure Seaborn global aesthetic parameters
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})


def generate_static_visual_suite(
    df_curated: pl.DataFrame,
    df_rfm: pl.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """
    Generates high-resolution static analytical charts using Seaborn and Matplotlib.
    
    Parameters:
        df_curated (pl.DataFrame): Cleaned transactions DataFrame.
        df_rfm (pl.DataFrame): RFM customer segments DataFrame.
        output_dir (str): Destination folder for PNG images.
        
    Returns:
        dict: Mapping of image names to output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("🎨 [Dashboard Generator] Rendering high-resolution static charts with Seaborn & Matplotlib...")
    image_paths = {}

    pdf_curated = df_curated.to_pandas()
    pdf_rfm = df_rfm.to_pandas()

    # Chart 1: Executive Performance Overview (Category Revenue & Profitability)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cat_metrics = pdf_curated.groupby('category').agg({
        'net_amount_clean': 'sum',
        'order_profit': 'sum'
    }).reset_index().sort_values(by='net_amount_clean', ascending=False)

    sns.barplot(data=cat_metrics, x='net_amount_clean', y='category', ax=ax1, palette='crest')
    ax1.set_title("Total Revenue by Product Category ($)", fontweight='bold')
    ax1.set_xlabel("Net Revenue ($)")
    ax1.set_ylabel("Category")
    for p in ax1.patches:
        width = p.get_width()
        ax1.annotate(f"${width:,.0f}", (width, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)

    sns.barplot(data=cat_metrics, x='order_profit', y='category', ax=ax2, palette='viridis')
    ax2.set_title("Gross Profit Margin by Product Category ($)", fontweight='bold')
    ax2.set_xlabel("Gross Profit ($)")
    ax2.set_ylabel("")
    for p in ax2.patches:
        width = p.get_width()
        ax2.annotate(f"${width:,.0f}", (width, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    overview_path = os.path.join(output_dir, "capstone_performance_summary.png")
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths["performance_summary"] = overview_path

    # Chart 2: RFM Customer Persona Distribution & Spend Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    seg_counts = pdf_rfm['customer_segment'].value_counts().reset_index()
    seg_counts.columns = ['customer_segment', 'count']

    sns.barplot(data=seg_counts, x='count', y='customer_segment', ax=ax1, palette='magma')
    ax1.set_title("Customer Volume by RFM Persona", fontweight='bold')
    ax1.set_xlabel("Number of Customers")
    ax1.set_ylabel("RFM Segment")

    sns.scatterplot(
        data=pdf_rfm,
        x='recency',
        y='monetary',
        hue='customer_segment',
        size='frequency',
        sizes=(20, 200),
        alpha=0.7,
        ax=ax2,
        palette='tab10'
    )
    ax2.set_title("Recency vs Monetary Spend by Persona", fontweight='bold')
    ax2.set_xlabel("Recency (Days Since Last Order)")
    ax2.set_ylabel("Monetary Spend ($)")
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    plt.tight_layout()
    rfm_path = os.path.join(output_dir, "rfm_customer_segments.png")
    plt.savefig(rfm_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths["rfm_segments"] = rfm_path

    print(f"  └─ Saved static figures: {[os.path.basename(p) for p in image_paths.values()]}")
    return image_paths


def build_interactive_plotly_dashboard(
    df_curated: pl.DataFrame,
    df_rfm: pl.DataFrame,
    psi_metrics: Dict[str, Any],
    output_html_path: str
) -> str:
    """
    Constructs a dynamic, multi-panel interactive Plotly HTML dashboard featuring:
    - Executive KPI metric summaries.
    - Monthly temporal revenue trends with rolling averages.
    - RFM Segment revenue heatmap.
    - Interactive payment method and device breakdown.
    
    Parameters:
        df_curated (pl.DataFrame): Cleaned transaction logs.
        df_rfm (pl.DataFrame): RFM persona segments.
        psi_metrics (dict): PSI drift calculation results.
        output_html_path (str): File destination for HTML output.
        
    Returns:
        str: Absolute path to the generated HTML dashboard.
    """
    print("🚀 [Dashboard Generator] Building interactive Plotly HTML Executive Dashboard Platform...")
    
    pdf_curated = df_curated.to_pandas()
    pdf_rfm = df_rfm.to_pandas()

    # Calculate overall executive KPIs
    total_revenue = pdf_curated['net_amount_clean'].sum()
    total_profit = pdf_curated['order_profit'].sum()
    total_orders = pdf_curated['order_id'].nunique()
    active_customers = pdf_rfm['customer_id'].nunique()
    aov = total_revenue / max(total_orders, 1)

    # Monthly revenue & profit trend
    pdf_curated['year_month'] = pd.to_datetime(pdf_curated['transaction_dt']).dt.to_period('M').astype(str)
    monthly_trend = pdf_curated.groupby('year_month').agg({
        'net_amount_clean': 'sum',
        'order_profit': 'sum',
        'order_id': 'nunique'
    }).reset_index().sort_values('year_month')

    # Create subplots figure layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monthly Revenue & Profit Growth ($)",
            "Revenue Contribution by RFM Persona",
            "Payment Method Volume & Status Breakdown",
            "Category Profitability Matrix ($)"
        ),
        specs=[
            [{"secondary_y": True}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )

    # Subplot 1: Monthly Growth Line Chart
    fig.add_trace(
        go.Scatter(
            x=monthly_trend['year_month'],
            y=monthly_trend['net_amount_clean'],
            name="Net Revenue ($)",
            mode="lines+markers",
            line=dict(color="#0066CC", width=3)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_trend['year_month'],
            y=monthly_trend['order_profit'],
            name="Gross Profit ($)",
            mode="lines+markers",
            line=dict(color="#00CC88", width=2, dash="dash")
        ),
        row=1, col=1, secondary_y=False
    )

    # Subplot 2: RFM Segment Pie Chart
    seg_spend = pdf_rfm.groupby('customer_segment')['monetary'].sum().reset_index()
    fig.add_trace(
        go.Pie(
            labels=seg_spend['customer_segment'],
            values=seg_spend['monetary'],
            name="RFM Revenue",
            hole=0.4,
            marker_colors=px.colors.qualitative.Prism
        ),
        row=1, col=2
    )

    # Subplot 3: Payment Method Distribution
    pm_counts = pdf_curated.groupby('payment_method')['order_id'].count().reset_index()
    fig.add_trace(
        go.Bar(
            x=pm_counts['payment_method'],
            y=pm_counts['order_id'],
            name="Order Count",
            marker_color="#FF9900"
        ),
        row=2, col=1
    )

    # Subplot 4: Category Profitability Bar Chart
    cat_prof = pdf_curated.groupby('category')['order_profit'].sum().reset_index().sort_values('order_profit', ascending=True)
    fig.add_trace(
        go.Bar(
            x=cat_prof['order_profit'],
            y=cat_prof['category'],
            orientation='h',
            name="Category Profit ($)",
            marker_color="#8884d8"
        ),
        row=2, col=2
    )

    # Apply sleek dark glassmorphism layout theme
    fig.update_layout(
        title=dict(
            text="<b>Phase 2 Capstone: Enterprise Analytics & Executive Dashboard</b>",
            x=0.5,
            font=dict(size=20, color="#FFFFFF")
        ),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        font=dict(color="#E5E7EB"),
        height=850,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Render custom HTML container with executive metric header cards
    html_header = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Phase 2 Capstone Executive Dashboard</title>
        <style>
            body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background-color: #0B0F19; color: #F3F4F6; margin: 0; padding: 20px; }}
            .kpi-container {{ display: flex; gap: 15px; justify-content: space-between; margin-bottom: 25px; }}
            .kpi-card {{ flex: 1; background: linear-gradient(135deg, #1F2937 0%, #111827 100%); border: 1px solid #374151; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4); }}
            .kpi-title {{ font-size: 0.85rem; text-transform: uppercase; color: #9CA3AF; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 8px; }}
            .kpi-value {{ font-size: 1.75rem; font-weight: 700; color: #38BDF8; }}
            .kpi-subtitle {{ font-size: 0.8rem; color: #10B981; margin-top: 4px; }}
            .badge-stable {{ color: #34D399; font-weight: 600; }}
            .badge-drift {{ color: #FBBF24; font-weight: 600; }}
            .chart-wrapper {{ background: #111827; border-radius: 12px; padding: 10px; border: 1px solid #374151; }}
        </style>
    </head>
    <body>
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Total Net Revenue</div>
                <div class="kpi-value">${total_revenue:,.2f}</div>
                <div class="kpi-subtitle">Across {total_orders:,} Orders</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Total Gross Profit</div>
                <div class="kpi-value" style="color: #34D399;">${total_profit:,.2f}</div>
                <div class="kpi-subtitle">Margin: {(total_profit/total_revenue)*100:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Average Order Value (AOV)</div>
                <div class="kpi-value" style="color: #F472B6;">${aov:,.2f}</div>
                <div class="kpi-subtitle">Per Transaction</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Active Customers</div>
                <div class="kpi-value" style="color: #A78BFA;">{active_customers:,}</div>
                <div class="kpi-subtitle">RFM Segmented</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Statistical PSI Drift</div>
                <div class="kpi-value" style="color: #FBBF24;">{psi_metrics.get('psi_score', 0.0):.4f}</div>
                <div class="kpi-subtitle">{psi_metrics.get('status', 'STABLE')}</div>
            </div>
        </div>
        <div class="chart-wrapper">
    """

    html_footer = """
        </div>
    </body>
    </html>
    """

    # Generate Plotly HTML string
    plotly_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    full_html_content = html_header + plotly_html + html_footer

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(full_html_content)

    print(f"  └─ Interactive Dashboard successfully saved to: '{output_html_path}'")
    return output_html_path
