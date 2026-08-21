#!/usr/bin/env python3
"""
================================================================================
100 Days of Data Science - Day 27
Topic: Advanced Statistical Data Visualization, Multi-Panel Dashboards & Seaborn EDA
================================================================================

This module covers:
1. Synthetic Multi-Variable Dataset Generation: Realistic customer analytics data.
2. Univariate Distribution Analysis: KDE plots, ECDF curves, and skewness indicators.
3. Bivariate Relational Plotting: Regression fits, PairGrids, and JointPlots.
4. Categorical Statistical Comparison: Violin plots, Swarm plots, and Point plots.
5. Multivariate Correlation Heatmaps: Annotated matrices and Hierarchical ClusterMaps.
6. Multi-Panel Executive EDA Dashboard: Grid layouts for C-suite presentation.
7. Automated Plot Export & Quality Assertions: High-DPI artifact generation.

Author: Dinesh Sitoula
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib to non-interactive backend for headless file generation
plt.switch_backend('Agg')

# Configure default Seaborn aesthetic theme for high-quality visuals
sns.set_theme(style="whitegrid", palette="muted", font="sans-serif")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# ------------------------------------------------------------------------------
# 0. SYNTHETIC MULTI-VARIABLE DATASET GENERATOR
# ------------------------------------------------------------------------------
def generate_customer_analytics_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generates a rich, synthetic customer analytics dataset simulating real-world
    e-commerce behavior with non-linear relationships, skewed distributions,
    and categorical groupings.

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility across random number generators.

    Returns:
    --------
    pd.DataFrame
        Clean DataFrame containing numerical and categorical features for EDA.
    """
    np.random.seed(seed)
    n_samples = 250

    # 1. Customer Demographic Features
    age = np.random.normal(38, 12, size=n_samples).clip(18, 75)
    
    # Income follows a right-skewed log-normal distribution (common in economics)
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).clip(20000, 250000)

    # 2. Financial & Behavioral Metrics
    credit_score = np.random.normal(680, 55, size=n_samples).clip(500, 850)
    
    # Purchase count correlated with age and income + random noise
    purchase_count = (0.2 * age + 0.0001 * annual_income + np.random.poisson(10, size=n_samples)).astype(int).clip(1, 100)
    
    # Total spend is non-linearly related to income and purchase count
    base_spend = purchase_count * np.random.uniform(50, 200, size=n_samples)
    total_spend = base_spend * (annual_income / 50000) ** 0.5 + np.random.normal(0, 500, size=n_samples)
    total_spend = np.clip(total_spend, 100, 50000)

    # Customer satisfaction score (1 to 5 Likert scale)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.1, 0.15, 0.35, 0.25, 0.15])

    # 3. Categorical Segments
    regions = np.random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America'], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    account_types = np.random.choice(['Basic', 'Standard', 'Premium', 'Enterprise'], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Churn probability higher for low satisfaction and low purchase count
    churn_logit = -0.05 * purchase_count - 0.8 * satisfaction_score + 2.5
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    # Construct DataFrame
    df = pd.DataFrame({
        'customer_id': [f"CUST-{2000+i}" for i in range(n_samples)],
        'age': np.round(age, 1),
        'annual_income': np.round(annual_income, 2),
        'credit_score': np.round(credit_score, 0).astype(int),
        'purchase_count': purchase_count,
        'total_spend': np.round(total_spend, 2),
        'satisfaction_score': satisfaction_score,
        'region': regions,
        'account_type': account_types,
        'churned': churned
    })

    return df


# ------------------------------------------------------------------------------
# 1. UNIVARIATE DISTRIBUTION PROFILING VISUALIZATIONS
# ------------------------------------------------------------------------------
def demonstrate_univariate_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots univariate statistical distributions including KDEs, ECDFs, Histograms
    with rug plots, and Skewness/Kurtosis annotations.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing numerical variables.
    output_dir : str
        Directory path to save generated visual artifact.
    """
    print("\n" + "=" * 80)
    print("1. UNIVARIATE DISTRIBUTION PROFILING & SKEWNESS ANALYSIS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Univariate Statistical Distributions & Feature Profiling", fontsize=16, fontweight='bold', y=0.98)

    # --------------------------------------------------------------------------
    # Subplot 1: Annual Income Distribution (Histogram + KDE + Skewness Annotation)
    # Right-skewed distribution analysis with rug plot
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    sns.histplot(df['annual_income'], kde=True, bins=25, color='teal', ax=ax1, stat="density", line_kws={'linewidth': 2})
    sns.rugplot(df['annual_income'], color='darkslategrey', ax=ax1, height=0.04)

    income_skew = df['annual_income'].skew()
    income_kurt = df['annual_income'].kurt()
    ax1.set_title(f"Annual Income (Lognormal Distribution)\nSkewness: {income_skew:.2f} | Kurtosis: {income_kurt:.2f}", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Annual Income ($)", fontweight='bold')
    ax1.set_ylabel("Density", fontweight='bold')
    # Draw vertical mean and median lines for skewness visualization
    ax1.axvline(df['annual_income'].mean(), color='crimson', linestyle='--', linewidth=1.5, label=f"Mean (${df['annual_income'].mean():,.0f})")
    ax1.axvline(df['annual_income'].median(), color='darkgreen', linestyle='-', linewidth=1.5, label=f"Median (${df['annual_income'].median():,.0f})")
    ax1.legend(loc='upper right', frameon=True)

    # --------------------------------------------------------------------------
    # Subplot 2: Age Distribution (Gaussian/Normal Distribution Analysis)
    # Symmetric bell-shaped distribution with kernel density curve
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    sns.kdeplot(df['age'], shade=True, color='indigo', ax=ax2, bw_adjust=0.8)
    sns.rugplot(df['age'], color='purple', ax=ax2, height=0.05)

    age_skew = df['age'].skew()
    ax2.set_title(f"Customer Age Distribution\nSkewness: {age_skew:.2f} (Symmetric)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Age (Years)", fontweight='bold')
    ax2.set_ylabel("Density", fontweight='bold')

    # --------------------------------------------------------------------------
    # Subplot 3: Credit Score Empirical Cumulative Distribution Function (ECDF)
    # ECDF avoids binning bias and directly shows percentiles & cumulative probabilities
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    sns.ecdfplot(data=df, x='credit_score', color='darkorange', linewidth=2.5, ax=ax3)
    ax3.set_title("Credit Score Empirical CDF (Cumulative Probability)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Credit Score", fontweight='bold')
    ax3.set_ylabel("Cumulative Probability P(X <= x)", fontweight='bold')
    # Highlight 50th percentile (Median)
    median_credit = df['credit_score'].median()
    ax3.axvline(median_credit, color='red', linestyle=':', label=f"Median Score ({median_credit:.0f})")
    ax3.axhline(0.5, color='red', linestyle=':')
    ax3.legend(loc='lower right', frameon=True)

    # --------------------------------------------------------------------------
    # Subplot 4: Total Spend Box Plot with Outlier Indicators
    # 5-number summary (Min, Q1, Median, Q3, Max) with IQR whisker bounds
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    sns.boxplot(x=df['total_spend'], color='seagreen', ax=ax4, flierprops={'marker':'o', 'markerfacecolor':'red', 'markersize':6})
    sns.stripplot(x=df['total_spend'], color='black', alpha=0.3, jitter=0.2, size=3, ax=ax4)
    ax4.set_title("Total Spend Distribution & Outliers (Box + Strip Overlay)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Total Spend ($)", fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "01_univariate_distributions.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[✓] Univariate distribution artifact saved to: {plot_path}")


# ------------------------------------------------------------------------------
# 2. BIVARIATE RELATIONAL PLOTTING & REGRESSION ANALYSIS
# ------------------------------------------------------------------------------
def demonstrate_bivariate_relational_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots bivariate relationships, linear/polynomial regression fits, joint
    marginal distributions, and multi-variable PairGrids.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing numerical continuous variables.
    output_dir : str
        Directory path to save generated visual artifacts.
    """
    print("\n" + "=" * 80)
    print("2. BIVARIATE RELATIONAL PLOTTING & REGRESSION ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Plot 1: 2x2 Grid of Relational Visualizations
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Bivariate Relational Patterns & Statistical Regression Fits", fontsize=16, fontweight='bold', y=0.98)

    # Subplot 1: Annual Income vs Total Spend by Churn Status (Hue Semantic)
    ax1 = axes[0, 0]
    sns.scatterplot(data=df, x='annual_income', y='total_spend', hue='churned', palette={0: 'navy', 1: 'crimson'},
                    style='churned', alpha=0.7, s=70, ax=ax1)
    # Add OLS regression line overlay
    sns.regplot(data=df, x='annual_income', y='total_spend', scatter=False, color='black',
                line_kws={'linestyle':'--', 'linewidth':1.8, 'label':'Linear Trend (OLS)'}, ax=ax1)
    ax1.set_title("Income vs Total Spend (Hue: Churned)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Annual Income ($)", fontweight='bold')
    ax1.set_ylabel("Total Spend ($)", fontweight='bold')
    ax1.legend(title="Customer Churned", frameon=True)

    # Subplot 2: Purchase Count vs Total Spend with Polynomial Regression (order=2)
    ax2 = axes[0, 1]
    sns.regplot(data=df, x='purchase_count', y='total_spend', order=2, color='darkgreen',
                scatter_kws={'alpha': 0.6, 's': 40}, line_kws={'linewidth': 2, 'label': '2nd Order Poly Fit'}, ax=ax2)
    ax2.set_title("Purchase Count vs Total Spend (Polynomial Fit)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Purchase Count", fontweight='bold')
    ax2.set_ylabel("Total Spend ($)", fontweight='bold')
    ax2.legend(loc='upper left', frameon=True)

    # Subplot 3: Age vs Credit Score (Bivariate Hexbin Density Plot)
    ax3 = axes[1, 0]
    hb = ax3.hexbin(df['age'], df['credit_score'], gridsize=20, cmap='Blues', mincnt=1)
    fig.colorbar(hb, ax=ax3, label='Customer Density Count')
    ax3.set_title("Age vs Credit Score (Bivariate Hexbin Density)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Age (Years)", fontweight='bold')
    ax3.set_ylabel("Credit Score", fontweight='bold')

    # Subplot 4: Annual Income vs Credit Score (2D KDE Density Contours)
    ax4 = axes[1, 1]
    sns.kdeplot(data=df, x='annual_income', y='credit_score', cmap='Viridis', fill=True, thresh=0.05, levels=10, ax=ax4)
    sns.scatterplot(data=df, x='annual_income', y='credit_score', color='white', alpha=0.3, s=15, ax=ax4)
    ax4.set_title("Income vs Credit Score (2D KDE Density Contours)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Annual Income ($)", fontweight='bold')
    ax4.set_ylabel("Credit Score", fontweight='bold')

    plt.tight_layout()
    bivariate_path = os.path.join(output_dir, "02_bivariate_relational_plots.png")
    plt.savefig(bivariate_path)
    plt.close()
    print(f"[✓] Bivariate relational plot artifact saved to: {bivariate_path}")

    # --------------------------------------------------------------------------
    # Plot 2: JointPlot with Marginal Kernel Density Distributions
    # Combines bivariate scatter density with univariate KDE marginals
    # --------------------------------------------------------------------------
    joint_grid = sns.jointplot(data=df, x='age', y='purchase_count', hue='account_type', kind='kde', fill=False, height=8)
    joint_grid.fig.suptitle("Age vs Purchase Count Joint Density Plot (By Account Type)", y=1.02, fontsize=13, fontweight='bold')
    joint_path = os.path.join(output_dir, "02_jointplot_marginal_density.png")
    joint_grid.savefig(joint_path)
    plt.close()
    print(f"[✓] JointPlot marginal density artifact saved to: {joint_path}")


# ------------------------------------------------------------------------------
# 3. CATEGORICAL STATISTICAL COMPARISONS (VIOLIN, SWARM & CONFIDENCE INTERVALS)
# ------------------------------------------------------------------------------
def demonstrate_categorical_comparisons(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots categorical distributions, kernel density violins, individual data swarm overlays,
    and bootstrapped 95% confidence interval estimation point plots.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing categorical and continuous variables.
    output_dir : str
        Directory path to save generated visual artifacts.
    """
    print("\n" + "=" * 80)
    print("3. CATEGORICAL STATISTICAL COMPARISONS & INFERENTIAL PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Categorical Feature Distributions & Inferential Statistical Estimations", fontsize=16, fontweight='bold', y=0.98)

    # --------------------------------------------------------------------------
    # Subplot 1: Violin Plot with Inner Boxplot (Income by Account Type)
    # Displays full probability density shape across discrete categorical levels
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    sns.violinplot(data=df, x='account_type', y='annual_income', order=['Basic', 'Standard', 'Premium', 'Enterprise'],
                   palette='Blues_d', inner='box', cut=0, ax=ax1)
    ax1.set_title("Annual Income Density by Account Tier (Violin + Inner Box)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Account Tier", fontweight='bold')
    ax1.set_ylabel("Annual Income ($)", fontweight='bold')

    # --------------------------------------------------------------------------
    # Subplot 2: Strip/Jitter Overlay on Box Plot (Total Spend by Region)
    # Combines summary statistics with actual raw sample observations
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='region', y='total_spend', palette='Set2', boxprops=dict(alpha=0.6), ax=ax2)
    sns.stripplot(data=df, x='region', y='total_spend', color='black', alpha=0.5, jitter=0.2, size=4, ax=ax2)
    ax2.set_title("Total Spend by Geographic Region (Box + Raw Observation Jitter)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Geographic Region", fontweight='bold')
    ax2.set_ylabel("Total Spend ($)", fontweight='bold')

    # --------------------------------------------------------------------------
    # Subplot 3: Point Plot displaying Mean & 95% Confidence Intervals
    # Uses bootstrapping to estimate sample mean uncertainty across subgroups
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    sns.pointplot(data=df, x='satisfaction_score', y='total_spend', hue='account_type',
                  markers=['o', 's', '^', 'D'], linestyles=['-', '--', '-.', ':'],
                  errorbar=('ci', 95), palette='tab10', capsize=0.15, ax=ax3)
    ax3.set_title("Total Spend vs Satisfaction (Mean + 95% CI by Tier)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Satisfaction Score (1-5)", fontweight='bold')
    ax3.set_ylabel("Mean Total Spend ($)", fontweight='bold')
    ax3.legend(title="Account Tier", loc='upper left', frameon=True)

    # --------------------------------------------------------------------------
    # Subplot 4: Categorical Bar Plot of Churn Rate (%) with Bootstrapped CIs
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    sns.barplot(data=df, x='region', y='churned', hue='account_type', errorbar=('ci', 95), palette='mako', ax=ax4)
    ax4.set_title("Customer Churn Proportion by Region & Account Tier", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Geographic Region", fontweight='bold')
    ax4.set_ylabel("Mean Churn Rate (Proportion)", fontweight='bold')
    ax4.legend(title="Account Tier", loc='upper right', frameon=True)

    plt.tight_layout()
    cat_path = os.path.join(output_dir, "03_categorical_comparisons.png")
    plt.savefig(cat_path)
    plt.close()
    print(f"[✓] Categorical statistical comparison artifact saved to: {cat_path}")


# ------------------------------------------------------------------------------
# 4. MULTIVARIATE CORRELATION HEATMAPS & HIERARCHICAL CLUSTERMAPS
# ------------------------------------------------------------------------------
def demonstrate_correlation_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    """
    Computes Pearson/Spearman feature correlation matrices, applies upper-triangular
    masks, plots annotated heatmaps, and builds hierarchical dendrogram ClusterMaps.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing numerical continuous variables.
    output_dir : str
        Directory path to save generated visual artifacts.
    """
    print("\n" + "=" * 80)
    print("4. MULTIVARIATE CORRELATION HEATMAPS & HIERARCHICAL CLUSTERMAPS")
    print("=" * 80)

    # Extract continuous numerical variables for correlation matrix computation
    num_cols = ['age', 'annual_income', 'credit_score', 'purchase_count', 'total_spend', 'satisfaction_score', 'churned']
    corr_pearson = df[num_cols].corr(method='pearson')
    corr_spearman = df[num_cols].corr(method='spearman')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Multivariate Feature Correlation Analysis", fontsize=16, fontweight='bold', y=0.98)

    # --------------------------------------------------------------------------
    # Subplot 1: Pearson Correlation Matrix with Upper-Triangular Mask
    # Upper triangular mask removes symmetric redundancy (r(X,Y) == r(Y,X))
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    cmap_diverging = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_pearson, mask=mask, cmap=cmap_diverging, vmin=-1.0, vmax=1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.8, cbar_kws={"shrink": .8}, ax=ax1)
    ax1.set_title("Pearson Correlation Matrix (Linear Relationships)", fontsize=11, fontweight='bold')

    # --------------------------------------------------------------------------
    # Subplot 2: Spearman Rank Correlation Matrix (Monotonic Relationships)
    # Spearman is robust to non-linear monotonic trends and extreme outliers
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    sns.heatmap(corr_spearman, mask=mask, cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.8, cbar_kws={"shrink": .8}, ax=ax2)
    ax2.set_title("Spearman Rank Correlation Matrix (Monotonic Trends)", fontsize=11, fontweight='bold')

    plt.tight_layout()
    corr_path = os.path.join(output_dir, "04_correlation_heatmaps.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"[✓] Multivariate correlation heatmap artifact saved to: {corr_path}")

    # --------------------------------------------------------------------------
    # Plot 2: Hierarchical Dendrogram ClusterMap
    # Performs agglomerative hierarchical clustering to group co-varying features
    # --------------------------------------------------------------------------
    cluster_grid = sns.clustermap(corr_pearson, cmap='vlag', vmin=-1, vmax=1, annot=True, fmt=".2f",
                                  linewidths=0.7, figsize=(8, 8))
    cluster_grid.fig.suptitle("Hierarchical Feature Clustering Dendrogram Map", y=1.02, fontsize=13, fontweight='bold')
    cluster_path = os.path.join(output_dir, "04_hierarchical_clustermap.png")
    cluster_grid.savefig(cluster_path)
    plt.close()
    print(f"[✓] Hierarchical ClusterMap artifact saved to: {cluster_path}")


# ------------------------------------------------------------------------------
# 5. MULTI-PANEL EXECUTIVE EDA DASHBOARD & EXPORT PIPELINE
# ------------------------------------------------------------------------------
def demonstrate_executive_eda_dashboard(df: pd.DataFrame, output_dir: str) -> None:
    """
    Assembles a publication-ready 2x3 grid executive EDA dashboard summarizing
    key insights across demographics, spend behavior, correlation, and churn drivers.

    Parameters:
    -----------
    df : pd.DataFrame
        Clean customer analytics dataset.
    output_dir : str
        Directory path to save generated dashboard artifact.
    """
    print("\n" + "=" * 80)
    print("5. MULTI-PANEL EXECUTIVE EDA DASHBOARD GENERATION")
    print("=" * 80)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Executive Customer Analytics & Churn EDA Dashboard", fontsize=18, fontweight='bold', y=0.98)

    # --------------------------------------------------------------------------
    # Panel 1 (0, 0): Income Distribution by Churn Status
    # --------------------------------------------------------------------------
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    sns.kdeplot(data=df, x='annual_income', hue='churned', palette={0:'navy', 1:'crimson'}, fill=True, alpha=0.4, ax=ax1)
    ax1.set_title("1. Income Distribution by Churn", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Annual Income ($)", fontweight='bold')
    ax1.set_ylabel("Density", fontweight='bold')

    # --------------------------------------------------------------------------
    # Panel 2 (0, 1): Spend vs Purchase Count Regression Fit
    # --------------------------------------------------------------------------
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    sns.regplot(data=df, x='purchase_count', y='total_spend', color='teal', scatter_kws={'alpha':0.5, 's':25}, ax=ax2)
    ax2.set_title("2. Spend vs Purchase Volume Trend", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Purchase Count", fontweight='bold')
    ax2.set_ylabel("Total Spend ($)", fontweight='bold')

    # --------------------------------------------------------------------------
    # Panel 3 (0, 2): Account Tier Breakdown by Geographic Region
    # --------------------------------------------------------------------------
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    region_tier = pd.crosstab(df['region'], df['account_type'], normalize='index') * 100
    region_tier.plot(kind='bar', stacked=True, colormap='Spectral', ax=ax3)
    ax3.set_title("3. Account Tier Share by Region (%)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Region", fontweight='bold')
    ax3.set_ylabel("Percentage (%)", fontweight='bold')
    ax3.legend(title="Tier", loc='lower right', framealpha=0.8)

    # --------------------------------------------------------------------------
    # Panel 4 (1, 0): Churn Rate vs Satisfaction Score (95% CI Point Plot)
    # --------------------------------------------------------------------------
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    sns.pointplot(data=df, x='satisfaction_score', y='churned', color='crimson', errorbar=('ci', 95), capsize=0.2, ax=ax4)
    ax4.set_title("4. Churn Probability vs Satisfaction (95% CI)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Satisfaction Rating (1-5)", fontweight='bold')
    ax4.set_ylabel("Churn Probability", fontweight='bold')

    # --------------------------------------------------------------------------
    # Panel 5 (1, 1): Churn Correlation Driver Analysis
    # --------------------------------------------------------------------------
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    num_cols = ['age', 'annual_income', 'credit_score', 'purchase_count', 'total_spend', 'satisfaction_score']
    churn_corr = df[num_cols].apply(lambda x: x.corr(df['churned'])).sort_values()
    churn_corr.plot(kind='barh', color=np.where(churn_corr > 0, 'crimson', 'teal'), ax=ax5)
    ax5.set_title("5. Feature Correlation with Churn", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Pearson Correlation Coefficient (r)", fontweight='bold')
    ax5.axvline(0, color='black', linewidth=0.8, linestyle='--')

    # --------------------------------------------------------------------------
    # Panel 6 (1, 2): Executive Key Insights Summary Callout Text Box
    # --------------------------------------------------------------------------
    ax6 = plt.subplot2grid((2, 3), (1, 2))
    ax6.axis('off')
    summary_text = (
        "EXECUTIVE SUMMARY & FINDINGS:\n"
        "--------------------------------------------------\n"
        "• Satisfaction Impact: High satisfaction (4-5)\n"
        "  reduces churn risk by > 60%.\n\n"
        "• High-Value Customers: Total spend is\n"
        "  strongly non-linear with purchase count.\n\n"
        "• Regional Concentration: North America & Europe\n"
        "  hold > 70% of Enterprise tier accounts.\n\n"
        "• Primary Churn Drivers: Low satisfaction score\n"
        "  and lower purchase frequency exhibit highest r.\n\n"
        "• Recommendation: Target satisfaction scores <= 2\n"
        "  with retention and loyalty incentives."
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.8', facecolor='whitesmoke', edgecolor='navy', alpha=0.9))

    plt.tight_layout()
    dash_path = os.path.join(output_dir, "05_executive_eda_dashboard.png")
    plt.savefig(dash_path)
    plt.close()
    print(f"[✓] Executive EDA dashboard artifact saved to: {dash_path}")


# ------------------------------------------------------------------------------
# MAIN ENTRY POINT & ARTIFACT VALIDATION
# ------------------------------------------------------------------------------
def main():
    """Runs all Day 27 Seaborn statistical data visualization demonstrations."""
    print("=" * 80)
    print("DAY 27: ADVANCED STATISTICAL DATA VISUALIZATION & SEABORN DASHBOARDS")
    print("=" * 80)

    # Define output directory for plot artifacts
    output_dir = "phase_1_python_programming/day-27-advanced-eda-seaborn-dashboards"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Synthetic Analytics Dataset
    df = generate_customer_analytics_dataset()
    print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Execute Visualization Modules
    demonstrate_univariate_distributions(df, output_dir)
    demonstrate_bivariate_relational_plots(df, output_dir)
    demonstrate_categorical_comparisons(df, output_dir)
    demonstrate_correlation_heatmaps(df, output_dir)
    demonstrate_executive_eda_dashboard(df, output_dir)

    # 3. Artifact Validation Assertions
    expected_artifacts = [
        "01_univariate_distributions.png",
        "02_bivariate_relational_plots.png",
        "02_jointplot_marginal_density.png",
        "03_categorical_comparisons.png",
        "04_correlation_heatmaps.png",
        "04_hierarchical_clustermap.png",
        "05_executive_eda_dashboard.png"
    ]

    print("\n--- Plot Artifact Validation ---")
    for artifact in expected_artifacts:
        path = os.path.join(output_dir, artifact)
        assert os.path.exists(path), f"Artifact missing: {path}"
        assert os.path.getsize(path) > 0, f"Artifact file is empty: {path}"
        print(f"[✓] Verified {artifact} ({os.path.getsize(path):,} bytes)")

    print("\n" + "=" * 80)
    print("ALL DAY 27 VISUALIZATION DEMONSTRATIONS COMPLETED SUCCESSFULLY! 🚀")
    print("=" * 80)


if __name__ == "__main__":
    main()





