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


if __name__ == "__main__":
    df = generate_customer_analytics_dataset()
    print("Dataset generated successfully. Shape:", df.shape)
    print(df.head())



