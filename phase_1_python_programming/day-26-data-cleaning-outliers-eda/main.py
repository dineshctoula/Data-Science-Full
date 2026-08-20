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


# ------------------------------------------------------------------------------
# 1. MISSING DATA ANALYSIS AND IMPUTATION STRATEGIES
# ------------------------------------------------------------------------------
def demonstrate_missing_data_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates missing value detection, missingness mechanisms (MCAR/MAR),
    and advanced conditional/grouped imputation techniques.
    """
    print("=" * 80)
    print("1. MISSING DATA ANALYSIS & CONDITIONAL IMPUTATION STRATEGIES")
    print("=" * 80)

    data = df.copy()

    # --------------------------------------------------------------------------
    # 1. Missing Data Audit: Count & Percentage of NaNs per column
    # --------------------------------------------------------------------------
    missing_count = data.isna().sum()
    missing_pct = (data.isna().mean() * 100).round(2)
    audit_df = pd.DataFrame({'missing_count': missing_count, 'missing_pct': missing_pct})

    print("--- Initial Missing Value Audit ---")
    print(audit_df[audit_df['missing_count'] > 0])

    # --------------------------------------------------------------------------
    # 2. Simple Global Imputation (Median for Age)
    # Median is preferred over Mean when distributions may be skewed.
    # --------------------------------------------------------------------------
    overall_median_age = data['age'].median()
    data['age_imputed'] = data['age'].fillna(overall_median_age)

    # --------------------------------------------------------------------------
    # 3. Conditional / Grouped Imputation (Income by Location / Tier)
    # Imputing income based on median within each location/tier group.
    # --------------------------------------------------------------------------
    # Clean location strings temporarily for accurate grouping
    temp_location = data['location'].str.strip().str.title()
    group_medians = data.groupby(temp_location)['annual_income'].transform('median')

    # Fill NaN with group median, fallback to global median if group median is NaN
    global_median_income = data['annual_income'].median()
    data['annual_income_imputed'] = data['annual_income'].fillna(group_medians).fillna(global_median_income)

    # --------------------------------------------------------------------------
    # 4. Forward/Backward Fill Interpolation for Spending Score
    # Useful for sequential or time-series structured data records
    # --------------------------------------------------------------------------
    data['spending_score_imputed'] = data['spending_score'].ffill().bfill()

    print("\n--- Summary Post-Imputation ---")
    print(f"Age NaN Count (Before -> After): {df['age'].isna().sum()} -> {data['age_imputed'].isna().sum()}")
    print(f"Income NaN Count (Before -> After): {df['annual_income'].isna().sum()} -> {data['annual_income_imputed'].isna().sum()}")
    print(f"Spending Score NaN Count (Before -> After): {df['spending_score'].isna().sum()} -> {data['spending_score_imputed'].isna().sum()}")

    # Assertions to ensure missing values are completely handled
    assert data['age_imputed'].isna().sum() == 0, "Age should have 0 missing values post-imputation!"
    assert data['annual_income_imputed'].isna().sum() == 0, "Income should have 0 missing values post-imputation!"
    assert data['spending_score_imputed'].isna().sum() == 0, "Spending score should have 0 missing values!"

    print("\n[✓] Missing data analysis and imputation completed successfully.")
    return data


# ------------------------------------------------------------------------------
# 2. OUTLIER DETECTION & TREATMENT TECHNIQUES (IQR & Z-SCORE)
# ------------------------------------------------------------------------------
def demonstrate_outlier_detection_and_capping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates statistical outlier detection using IQR and Z-score methods,
    and applies Winsorization (capping/flooring) to treat extreme values.
    """
    print("\n" + "=" * 80)
    print("2. OUTLIER DETECTION & TREATMENT (IQR, Z-SCORE & WINSORIZATION)")
    print("=" * 80)

    data = df.copy()
    income_col = 'annual_income_imputed'

    # First, handle impossible negative income values (domain rule validation)
    negative_count = (data[income_col] < 0).sum()
    data[income_col] = data[income_col].clip(lower=0)
    print(f"Domain Validation: Floor-adjusted {negative_count} negative income values to 0.")

    # --------------------------------------------------------------------------
    # 1. Interquartile Range (IQR) Method
    # Q1 = 25th percentile, Q3 = 75th percentile
    # IQR = Q3 - Q1
    # Lower Bound = Q1 - 1.5 * IQR, Upper Bound = Q3 + 1.5 * IQR
    # --------------------------------------------------------------------------
    q1 = data[income_col].quantile(0.25)
    q3 = data[income_col].quantile(0.75)
    iqr = q3 - q1

    lower_iqr_bound = q1 - 1.5 * iqr
    upper_iqr_bound = q3 + 1.5 * iqr

    iqr_outliers = data[(data[income_col] < lower_iqr_bound) | (data[income_col] > upper_iqr_bound)]
    print(f"\n--- IQR Outlier Detection ---")
    print(f"Q1 (25%): ${q1:,.2f} | Q3 (75%): ${q3:,.2f} | IQR: ${iqr:,.2f}")
    print(f"IQR Lower Bound: ${lower_iqr_bound:,.2f} | Upper Bound: ${upper_iqr_bound:,.2f}")
    print(f"Number of IQR Outliers Detected: {len(iqr_outliers)}")

    # --------------------------------------------------------------------------
    # 2. Z-Score Method (|Z| > 3.0)
    # Z = (X - Mean) / Std
    # --------------------------------------------------------------------------
    mean_inc = data[income_col].mean()
    std_inc = data[income_col].std()
    data['income_zscore'] = (data[income_col] - mean_inc) / std_inc

    z_outliers = data[data['income_zscore'].abs() > 3.0]
    print(f"\n--- Z-Score Outlier Detection (|Z| > 3.0) ---")
    print(f"Mean Income: ${mean_inc:,.2f} | Std Dev: ${std_inc:,.2f}")
    print(f"Number of Z-Score Outliers (|Z| > 3.0): {len(z_outliers)}")

    # --------------------------------------------------------------------------
    # 3. Outlier Treatment: Winsorization (Capping & Flooring)
    # --------------------------------------------------------------------------
    data['income_capped'] = data[income_col].clip(lower=max(0, lower_iqr_bound), upper=upper_iqr_bound)

    print("\n--- Post-Winsorization Comparison ---")
    print(f"Max Income Before Capping: ${data[income_col].max():,.2f}")
    print(f"Max Income After Capping:  ${data['income_capped'].max():,.2f}")

    # Assertions to verify outlier capping logic
    assert data['income_capped'].max() <= upper_iqr_bound + 1e-5, "Income post-capping must not exceed IQR upper bound!"
    assert data['income_capped'].min() >= 0, "Income post-capping must be non-negative!"

    print("\n[✓] Outlier detection and Winsorization capping completed successfully.")
    return data


# ------------------------------------------------------------------------------
# 3. FEATURE TRANSFORMATION & NORMALIZATION (SKEWNESS REDUCTION & SCALING)
# ------------------------------------------------------------------------------
def demonstrate_transformation_and_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates distribution skewness analysis, log transformation (`np.log1p`),
    Min-Max Scaling [0, 1], and Standard Z-score Normalization (mean=0, std=1).
    """
    print("\n" + "=" * 80)
    print("3. FEATURE TRANSFORMATION & NORMALIZATION (LOG TRANSFORMS & SCALING)")
    print("=" * 80)

    data = df.copy()
    income_raw = data['annual_income_imputed']

    # --------------------------------------------------------------------------
    # 1. Skewness and Kurtosis Analysis
    # Skewness > 1 indicate heavy right-skewed distribution.
    # --------------------------------------------------------------------------
    skew_raw = income_raw.skew()
    kurt_raw = income_raw.kurt()
    print(f"Original Income Skewness: {skew_raw:.4f} | Kurtosis: {kurt_raw:.4f}")

    # --------------------------------------------------------------------------
    # 2. Log Transformation: np.log1p(x) = log(1 + x)
    # Reduces variance and stabilizes right-skewed feature distributions
    # --------------------------------------------------------------------------
    data['income_log1p'] = np.log1p(data['annual_income_imputed'].clip(lower=0))
    skew_log = data['income_log1p'].skew()
    kurt_log = data['income_log1p'].kurt()

    print(f"Log-Transformed Income Skewness: {skew_log:.4f} | Kurtosis: {kurt_log:.4f}")
    print(f"Skewness Reduction: {abs(skew_raw) - abs(skew_log):.4f} reduction achieved.")

    # --------------------------------------------------------------------------
    # 3. Feature Scaling: Min-Max Scaling [0, 1]
    # Formula: (X - X_min) / (X_max - X_min)
    # --------------------------------------------------------------------------
    min_val = data['income_capped'].min()
    max_val = data['income_capped'].max()
    data['income_minmax'] = (data['income_capped'] - min_val) / (max_val - min_val)

    # --------------------------------------------------------------------------
    # 4. Feature Scaling: Standardization / Z-score Normalization (Mean=0, Std=1)
    # Formula: (X - Mean) / Std
    # --------------------------------------------------------------------------
    mean_capped = data['income_capped'].mean()
    std_capped = data['income_capped'].std()
    data['income_standardized'] = (data['income_capped'] - mean_capped) / std_capped

    print("\n--- Scaling Verification Statistics ---")
    print(f"Min-Max Range: [{data['income_minmax'].min():.4f}, {data['income_minmax'].max():.4f}]")
    print(f"Standardized Mean: {data['income_standardized'].mean():.6f} (Expected ≈ 0)")
    print(f"Standardized Std:  {data['income_standardized'].std():.6f} (Expected ≈ 1)")

    # Mathematical assertions
    assert data['income_minmax'].min() == 0.0 and data['income_minmax'].max() == 1.0, "Min-Max scaling must map to [0, 1]!"
    assert np.isclose(data['income_standardized'].mean(), 0.0, atol=1e-5), "Standardized mean must be zero!"
    assert np.isclose(data['income_standardized'].std(), 1.0, atol=1e-5), "Standardized std must be one!"

    print("\n[✓] Feature transformation and scaling completed successfully.")
    return data


# ------------------------------------------------------------------------------
# 4. CATEGORICAL NORMALIZATION & FEATURE ENCODING STRATEGIES
# ------------------------------------------------------------------------------
def demonstrate_categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates text normalization, regex cleaning, Ordinal Encoding,
    One-Hot Encoding (`pd.get_dummies`), and Frequency Encoding.
    """
    print("\n" + "=" * 80)
    print("4. CATEGORICAL NORMALIZATION & FEATURE ENCODING STRATEGIES")
    print("=" * 80)

    data = df.copy()

    # --------------------------------------------------------------------------
    # 1. Text Normalization: Whitespace Stripping & Casing Harmonization
    # --------------------------------------------------------------------------
    print("--- Categorical Values Before Cleaning ---")
    print("Raw Tier Levels:", data['membership_tier'].unique())
    print("Raw Locations:  ", data['location'].unique())

    # Standardize membership tier string format
    data['membership_tier_clean'] = data['membership_tier'].str.strip().str.title()
    # Standardize location string format
    data['location_clean'] = data['location'].str.strip().str.title()

    print("\n--- Categorical Values After Cleaning ---")
    print("Clean Tier Levels:", data['membership_tier_clean'].unique())
    print("Clean Locations:  ", data['location_clean'].unique())

    # --------------------------------------------------------------------------
    # 2. Ordinal Encoding (Explicit Mapping for Ordered Categories)
    # Standard = 1, Premium = 2, VIP = 3
    # --------------------------------------------------------------------------
    tier_mapping = {'Standard': 1, 'Premium': 2, 'Vip': 3}
    data['membership_tier_encoded'] = data['membership_tier_clean'].map(tier_mapping)

    # --------------------------------------------------------------------------
    # 3. Frequency Encoding (High Cardinality Alternative)
    # Maps categories to their relative proportions in the dataset
    # --------------------------------------------------------------------------
    location_freq = data['location_clean'].value_counts(normalize=True)
    data['location_freq_encoded'] = data['location_clean'].map(location_freq)

    # --------------------------------------------------------------------------
    # 4. One-Hot Encoding (pd.get_dummies) for Nominal Features
    # --------------------------------------------------------------------------
    location_dummies = pd.get_dummies(data['location_clean'], prefix='loc', dtype=int)
    data = pd.concat([data, location_dummies], axis=1)

    print("\n--- Encoding Outputs Sample ---")
    print(data[['customer_id', 'membership_tier_clean', 'membership_tier_encoded', 'location_clean', 'location_freq_encoded']].head(6))

    # Assertions for encoding correctness
    assert set(data['membership_tier_clean'].unique()) == {'Standard', 'Premium', 'Vip'}, "Should have exactly 3 clean tier levels!"
    assert data['membership_tier_encoded'].isna().sum() == 0, "Ordinal encoding should have 0 unmapped NaNs!"
    assert 'loc_New York' in data.columns, "One-Hot dummy columns must be present!"

    print("\n[✓] Categorical normalization and feature encoding completed successfully.")
    return data


# ------------------------------------------------------------------------------
# 5. AUTOMATED EDA & DATA QUALITY AUDIT REPORT GENERATION
# ------------------------------------------------------------------------------
def demonstrate_automated_eda_profiling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an automated, structured Data Quality Audit & Profiling Report
    computing completeness, cardinality, skewness, and outlier flags per column.
    """
    print("\n" + "=" * 80)
    print("5. AUTOMATED EDA & DATA QUALITY AUDIT REPORT GENERATION")
    print("=" * 80)

    report_list = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_cnt = df[col].nunique()
        missing_cnt = df[col].isna().sum()
        missing_pct = (missing_cnt / len(df)) * 100

        skewness = np.nan
        outlier_cnt = 0

        if pd.api.types.is_numeric_dtype(df[col]):
            skewness = round(df[col].skew(), 3)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_cnt = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()

        report_list.append({
            'Column': col,
            'DataType': dtype,
            'UniqueCount': unique_cnt,
            'MissingCount': missing_cnt,
            'MissingPct': f"{missing_pct:.1f}%",
            'Skewness': skewness if not np.isnan(skewness) else 'N/A',
            'IQROutliers': outlier_cnt
        })

    eda_report_df = pd.DataFrame(report_list)

    print("--- Automated Data Health Audit Summary ---")
    print(eda_report_df.to_string(index=False))

    # Verification assertions
    assert len(eda_report_df) == len(df.columns), "EDA audit report must contain metrics for all columns!"
    assert (eda_report_df['MissingCount'] >= 0).all(), "Missing count must be non-negative!"

    print("\n[✓] Automated EDA profiling report generated successfully.")
    return eda_report_df


if __name__ == "__main__":
    df_raw = generate_messy_customer_dataset()
    print("--- Initial Messy Dataset Head ---")
    print(df_raw.head(10))
    print(f"Shape of messy dataset: {df_raw.shape}")

    df_imputed = demonstrate_missing_data_imputation(df_raw)
    df_capped = demonstrate_outlier_detection_and_capping(df_imputed)
    df_scaled = demonstrate_transformation_and_scaling(df_capped)
    df_encoded = demonstrate_categorical_encoding(df_scaled)
    df_report = demonstrate_automated_eda_profiling(df_encoded)





