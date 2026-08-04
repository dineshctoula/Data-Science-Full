"""
Day 22 – Advanced Data Visualization with Matplotlib & Seaborn
===============================================================
100-Day Data Science Challenge | Phase 2: Data Manipulation & Visualization

Topics Covered:
  1. Matplotlib Fundamentals   – Line charts, markers, labels, legends
  2. Bar & Horizontal Charts   – Comparing categorical data
  3. Histograms & KDE          – Exploring distributions
  4. Scatter Plots             – Spotting relationships between variables
  5. Seaborn Statistical Plots – Box plots, violin plots, heat maps
  6. Subplots & Figure Layout  – Multi-panel dashboards

Learning Goals:
  - Understand the Figure / Axes object model in Matplotlib
  - Choose the right chart type for each data question
  - Customise aesthetics: colors, fonts, grid, spines
  - Combine Matplotlib and Seaborn in the same workflow
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────
# Global style – applied to every chart in this file
# ─────────────────────────────────────────────
# 'seaborn-v0_8-darkgrid' gives a clean grid background.
# Fallback to 'ggplot' on older Matplotlib versions.
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")

# A consistent colour palette used across all charts
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ===========================================================
# SECTION 1 – LINE CHARTS
# ===========================================================

def plot_line_chart():
    """
    Demonstrates a multi-line chart using Matplotlib's object-oriented API.

    Key concepts:
      - plt.subplots()  → creates a Figure + Axes pair (preferred over pyplot)
      - ax.plot()       → draws a line series
      - ax.set_*()      → titles, axis labels
      - ax.legend()     → automatic legend from the 'label' keyword
    """
    print("\n📈 Plotting Line Chart …")

    # ── Sample data: monthly revenue for three products ──────────────────
    months = np.arange(1, 13)          # January to December (1–12)

    # Simulate revenue with a base trend + random noise for realism
    rng = np.random.default_rng(seed=42)   # fixed seed → reproducible results
    revenue_a = 5_000 + np.cumsum(rng.integers(200, 600, size=12))
    revenue_b = 4_500 + np.cumsum(rng.integers(100, 500, size=12))
    revenue_c = 3_000 + np.cumsum(rng.integers(300, 700, size=12))

    # Month abbreviations for the x-axis tick labels
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── Build the chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each product as a separate line; 'o' adds circle markers
    ax.plot(months, revenue_a, marker="o", color=PALETTE[0], label="Product A", linewidth=2)
    ax.plot(months, revenue_b, marker="s", color=PALETTE[1], label="Product B", linewidth=2)
    ax.plot(months, revenue_c, marker="^", color=PALETTE[2], label="Product C", linewidth=2)

    # ── Cosmetic tweaks ──────────────────────────────────────────────────
    ax.set_title("Monthly Revenue by Product (2024)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Revenue ($)", fontsize=12)

    # Replace numeric ticks with month abbreviations
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)

    # Format the y-axis values as currency (e.g. $12,500)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Add a horizontal dashed line at the overall mean revenue of Product A
    ax.axhline(revenue_a.mean(), color=PALETTE[0], linestyle="--",
               linewidth=1, alpha=0.6, label="Avg Product A")

    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("line_chart.png", dpi=150)   # save to disk for inspection
    plt.show()
    print("   ✅ line_chart.png saved")


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Day 22 – Advanced Data Visualization")
    print("=" * 55)

    # Section 1
    plot_line_chart()

    print("\n✅ All charts rendered successfully!")
