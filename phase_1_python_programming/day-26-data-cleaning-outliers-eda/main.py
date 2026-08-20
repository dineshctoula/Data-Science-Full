#!/usr/bin/env python3
"""
================================================================================
100 Days of Data Science - Day 26
Topic: Advanced Data Cleaning, Outlier Detection, Imputation & EDA Profiling
================================================================================

This module covers:
1. Synthetic Messy Data Generation: Creating real-world data with missingness, noise, & outliers.
2. Advanced Missing Data Handling: MCAR vs MAR mechanisms, conditional median/mode imputation.
3. Outlier Detection & Treatment: IQR bounds filtering, Z-score thresholding, and Winsorization capping.
4. Feature Transformation & Scaling: Skewness reduction (log1p), Min-Max scaling, and Z-score standardization.
5. Categorical Encoding & Text Normalization: Regex string cleaning, Ordinal, One-Hot, and Frequency encoding.
6. Automated Data Quality Audit: Skewness, missingness ratio, cardinality metrics, and reporting.
7. End-to-End Cleaning Pipeline: Real-world customer churn dataset processing case study.

Author: Dinesh Sitoula
================================================================================
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# 0. MESSY DATASET GENERATOR
# ------------------------------------------------------------------------------
def generate_messy_customer_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, dirty dataset simulating e-commerce customer behavior.
    Includes:
    - Missing values (MCAR and MAR mechanisms)
    - Extreme numerical outliers in income and purchase values
    - Inconsistent string casing and whitespace anomalies in location/category
    - Duplicate customer IDs and records
    - Inconsistent date strings
    """
    np.random.seed(seed)
    n_samples = 100

    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    # Intentionally inject 5 duplicate customer IDs
    customer_ids[10:15] = customer_ids[5:10]

    ages = np.random.randint(18, 70, size=n_samples).astype(float)
    # Inject missing age values
    ages[np.random.choice(n_samples, 8, replace=False)] = np.nan

    incomes = np.random.normal(55000, 15000, size=n_samples)
    # Inject negative values and extreme high-income outliers
    incomes[12] = -12000.0
    incomes[45] = 450000.0
    incomes[88] = 620000.0
    # Inject missing incomes
    incomes[np.random.choice(n_samples, 10, replace=False)] = np.nan

    spending_score = np.random.uniform(1, 100, size=n_samples)
    spending_score[np.random.choice(n_samples, 5, replace=False)] = np.nan

    join_dates = [
        f"2025/0{np.random.randint(1, 9)}/{np.random.randint(10, 28)}" if i % 2 == 0
        else f"0{np.random.randint(1, 9)}-{np.random.randint(10, 28)}-2025"
        for i in range(n_samples)
    ]

    tier_choices = ['Standard', 'standard ', 'PREMIUM', 'Premium', 'VIP', ' vip']
    tiers = np.random.choice(tier_choices, size=n_samples)

    locations = np.random.choice(['New York', 'NEW YORK', ' Chicago', 'San Francisco', 'san francisco'], size=n_samples)

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'annual_income': incomes,
        'spending_score': spending_score,
        'membership_tier': tiers,
        'location': locations,
        'join_date': join_dates
    })

    return df


if __name__ == "__main__":
    df_raw = generate_messy_customer_dataset()
    print("--- Initial Messy Dataset Head ---")
    print(df_raw.head(10))
    print(f"Shape of messy dataset: {df_raw.shape}")
