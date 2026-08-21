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


if __name__ == "__main__":
    df = generate_customer_analytics_dataset()
    print("Dataset generated successfully. Shape:", df.shape)
    print(df.head())
