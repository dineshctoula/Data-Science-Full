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
# SECTION 2 – BAR CHARTS (Categorical Comparison)
# ===========================================================

def plot_bar_chart():
    """
    Demonstrates vertical and horizontal bar charts for comparing
    categorical data (e.g., sales by region or department scores).

    Key concepts:
      - ax.bar()     → vertical bars; pass x positions + heights
      - ax.barh()    → horizontal bars; useful when category labels are long
      - ax.bar_label() → annotate bars with their exact values
      - fig.suptitle() → shared title across multiple subplots
    """
    print("\n📊 Plotting Bar Charts …")

    # ── Sample data: quarterly sales by region ──────────────────────────
    regions = ["North", "South", "East", "West", "Central"]
    q1_sales = [12_500, 9_800, 14_200, 11_000, 8_600]   # Q1 figures
    q2_sales = [13_100, 10_500, 13_800, 12_400, 9_200]  # Q2 figures

    # x positions for the bars; we'll offset Q1 & Q2 side by side
    x = np.arange(len(regions))   # [0, 1, 2, 3, 4]
    bar_width = 0.35               # width of each bar

    # ── Create a 1×2 grid: vertical bar (left) | horizontal bar (right) ─
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Quarterly Sales by Region", fontsize=15, fontweight="bold")

    # ── LEFT: Grouped vertical bar chart ─────────────────────────────────
    bars1 = ax1.bar(x - bar_width / 2, q1_sales, bar_width,
                    color=PALETTE[0], label="Q1")
    bars2 = ax1.bar(x + bar_width / 2, q2_sales, bar_width,
                    color=PALETTE[1], label="Q2")

    # Annotate each bar with its numeric value above the bar
    ax1.bar_label(bars1, fmt="$%,.0f", fontsize=8, padding=3)
    ax1.bar_label(bars2, fmt="$%,.0f", fontsize=8, padding=3)

    ax1.set_title("Grouped Bar Chart (Vertical)")
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Sales ($)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.legend()

    # ── RIGHT: Horizontal bar chart (good for long category names) ────────
    # Compute the percentage growth between Q1 and Q2 for each region
    growth_pct = [(q2 - q1) / q1 * 100 for q1, q2 in zip(q1_sales, q2_sales)]

    # Colour bars green if growth > 0, red if negative (for visual clarity)
    colours = [PALETTE[2] if g > 0 else PALETTE[3] for g in growth_pct]
    bars3 = ax2.barh(regions, growth_pct, color=colours)

    # Annotate each bar with the percentage value
    ax2.bar_label(bars3, fmt="%.1f%%", padding=4, fontsize=9)

    ax2.set_title("Q1→Q2 Growth (Horizontal Bar)")
    ax2.set_xlabel("Growth (%)")
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")  # zero reference line

    plt.tight_layout()
    plt.savefig("bar_chart.png", dpi=150)
    plt.show()
    print("   ✅ bar_chart.png saved")


# ===========================================================
# SECTION 3 – HISTOGRAMS & KDE (Distribution Analysis)
# ===========================================================

def plot_histogram():
    """
    Shows how to explore the distribution of a continuous variable
    using histograms and Kernel Density Estimates (KDE).

    Key concepts:
      - ax.hist()          → frequency histogram; 'bins' controls bucket width
      - density=True       → normalise so the y-axis shows probability density
      - sns.kdeplot()      → smooth KDE curve drawn on the same Axes
      - ax.axvline()       → vertical line to mark mean / median
      - Overlaying KDE on hist lets us see both exact bins AND smooth shape
    """
    print("\n📉 Plotting Histograms & KDE …")

    # ── Simulate exam scores for three student cohorts ──────────────────
    rng = np.random.default_rng(seed=7)

    # Group A: well-prepared students (higher mean, smaller spread)
    scores_a = rng.normal(loc=72, scale=8, size=300)

    # Group B: average preparation (medium mean, wider spread)
    scores_b = rng.normal(loc=65, scale=12, size=300)

    # Group C: mix of high and low scorers (bimodal distribution)
    scores_c = np.concatenate([
        rng.normal(loc=55, scale=6, size=150),   # lower mode
        rng.normal(loc=85, scale=5, size=150),   # upper mode
    ])

    # Clip scores to the valid range [0, 100]
    scores_a = np.clip(scores_a, 0, 100)
    scores_b = np.clip(scores_b, 0, 100)
    scores_c = np.clip(scores_c, 0, 100)

    # ── Build a 1×3 subplot grid, one panel per cohort ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Exam Score Distributions by Cohort", fontsize=14, fontweight="bold")

    datasets = [
        (scores_a, "Group A (High Prep)",    PALETTE[0]),
        (scores_b, "Group B (Average Prep)", PALETTE[1]),
        (scores_c, "Group C (Bimodal)",      PALETTE[2]),
    ]

    for ax, (scores, label, colour) in zip(axes, datasets):
        # ── Histogram: density=True normalises area under bars to 1 ──────
        ax.hist(scores, bins=20, density=True, alpha=0.55,
                color=colour, edgecolor="white", label="Histogram")

        # ── KDE overlay from Seaborn (smooth probability density curve) ──
        # Using the same Axes object so both plots share the coordinate space
        sns.kdeplot(scores, ax=ax, color=colour, linewidth=2.5, label="KDE")

        # ── Reference lines for mean and median ──────────────────────────
        ax.axvline(scores.mean(),   color="red",    linestyle="--",
                   linewidth=1.5, label=f"Mean  {scores.mean():.1f}")
        ax.axvline(np.median(scores), color="black", linestyle=":",
                   linewidth=1.5, label=f"Median {np.median(scores):.1f}")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("histogram_kde.png", dpi=150)
    plt.show()
    print("   ✅ histogram_kde.png saved")


# ===========================================================
# SECTION 4 – SCATTER PLOTS (Relationship Analysis)
# ===========================================================

def plot_scatter():
    """
    Uses scatter plots to visualise relationships between two continuous
    variables and adds a regression trend line for quick interpretation.

    Key concepts:
      - ax.scatter()    → plot (x, y) pairs as individual points
      - 'c' parameter   → colour each point by a third variable (heat map style)
      - 'alpha'         → transparency for dense data points (overplotting)
      - np.polyfit()    → fit a degree-1 polynomial (linear regression line)
      - np.poly1d()     → convert polynomial coefficients to a callable function
    """
    print("\n🔵 Plotting Scatter Plots …")

    # ── Simulate house data: area (sqft) vs price ($) ───────────────────
    rng = np.random.default_rng(seed=21)
    n = 200

    area = rng.uniform(500, 3_500, size=n)        # house area in sqft
    rooms = rng.integers(1, 7, size=n)            # number of rooms (1–6)

    # Price increases with area, with some added noise to simulate real data
    # Larger houses have higher variance in price (heteroscedasticity)
    noise = rng.normal(0, 20_000 + area * 8, size=n)
    price = 80_000 + area * 120 + rooms * 15_000 + noise
    price = np.clip(price, 50_000, None)          # no negative prices

    # ── Build the figure ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("House Area vs Price", fontsize=14, fontweight="bold")

    # ── LEFT: Basic scatter coloured by number of rooms ──────────────────
    sc = ax1.scatter(area, price, c=rooms, cmap="viridis",
                     alpha=0.65, edgecolors="none", s=40)

    # Add a colour bar so we know what each colour represents
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("# Rooms", fontsize=9)

    # Fit and plot a linear trend line using NumPy polynomial fitting
    # np.polyfit returns [slope, intercept] for degree=1
    coeffs = np.polyfit(area, price, deg=1)
    trend_fn = np.poly1d(coeffs)             # callable trend function
    x_line = np.linspace(area.min(), area.max(), 300)
    ax1.plot(x_line, trend_fn(x_line), color="red", linewidth=2,
             linestyle="--", label=f"Trend (slope={coeffs[0]:,.0f}$/sqft)")

    ax1.set_title("Scatter + Linear Trend")
    ax1.set_xlabel("Area (sqft)")
    ax1.set_ylabel("Price ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
    ax1.legend(fontsize=9)

    # ── RIGHT: Annotated scatter highlighting outliers ────────────────────
    # Compute z-scores to detect price outliers (|z| > 2 is unusual)
    z_scores = (price - price.mean()) / price.std()
    is_outlier = np.abs(z_scores) > 2

    ax2.scatter(area[~is_outlier], price[~is_outlier],
                color=PALETTE[0], alpha=0.6, s=35, label="Normal")
    ax2.scatter(area[is_outlier], price[is_outlier],
                color=PALETTE[3], alpha=0.9, s=70, marker="*", label="Outlier")

    ax2.set_title("Scatter with Outlier Highlighting")
    ax2.set_xlabel("Area (sqft)")
    ax2.set_ylabel("Price ($)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("scatter_plot.png", dpi=150)
    plt.show()
    print("   ✅ scatter_plot.png saved")


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Day 22 – Advanced Data Visualization")
    print("=" * 55)

    # Section 1 – Line chart
    plot_line_chart()

    # Section 2 – Bar charts (categorical comparison)
    plot_bar_chart()

    # Section 3 – Histograms & KDE (distribution analysis)
    plot_histogram()

    # Section 4 – Scatter plots (relationship + outliers)
    plot_scatter()

    print("\n✅ All charts rendered successfully!")
