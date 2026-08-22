#!/usr/bin/env python3
"""
================================================================================
100 Days of Data Science - Day 28
Topic: Advanced DateTime Handling, Timezone Analytics & Vectorized Text Regex Pipelines
================================================================================

This module covers:
1. Synthetic Data Generation: Creating complex multi-format timestamp and messy text logs.
2. Advanced DateTime Analytics: Parsing heterogeneous formats, timezone conversion,
   business calendars (BDay, MonthEnd), and elapsed delta calculations.
3. Vectorized Text & Regex Processing: String cleanup, email domain extraction,
   regex capture groups, phone standardization, and token splitting.
4. Text Feature Engineering: Custom Term Frequency-Inverse Document Frequency (TF-IDF)
   vectorizer implemented from scratch using Pandas & NumPy.
5. End-to-End Case Study: Customer support log processing pipeline integrating time-series
   response metrics with urgent keyword regex detection.

Author: Dinesh Sitoula
================================================================================
"""

import math
import re
from datetime import datetime
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ------------------------------------------------------------------------------
def generate_synthetic_ecom_logs(n_samples: int = 100) -> pd.DataFrame:
    """
    Generates a realistic, messy synthetic dataset of customer interaction logs.

    Parameters:
        n_samples (int): Number of log records to create.

    Returns:
        pd.DataFrame: DataFrame containing raw timestamps, emails, addresses,
                      product text, search queries, and response notes.
    """
    np.random.seed(42)

    # Heterogeneous timestamp formats (ISO, US standard, timestamp with timezone)
    raw_timestamps = [
        "2026-08-01 08:30:15",
        "08/02/2026 14:22:10",
        "2026-08-03T18:45:00+02:00",
        "2026-08-04 23:10:05",
        "08/05/2026 09:15:30",
        "2026-08-06T11:00:00-05:00",
        "2026-08-07 16:50:20",
        "invalid_timestamp_log",  # Intentional dirty data for error handling
        "2026-08-08 12:00:00",
        "2026-08-09 20:30:45",
    ] * (n_samples // 10 + 1)

    # Messy customer emails with whitespace, uppercase, and invalid formats
    raw_emails = [
        "  JOHN.DOE@Gmail.com  ",
        "alice_smith99@yahoo.co.uk",
        "BOB.MARLEY@company.ORG",
        "charlie.brown@CORP.NET",
        "invalid_email_at_domain",
        "  eva.green@GMAIL.COM",
        "frank.wright@tech-hub.io",
        "grace.hopper@mit.edu",
        "  HELEN.KELLER@FOUNDATION.ORG  ",
        "ian.fleming@MI6.GOV.UK",
    ] * (n_samples // 10 + 1)

    # Phone numbers with inconsistent formatting
    raw_phones = [
        "+1 (555) 234-5678",
        "555-987-6543",
        "555.345.6789",
        "(555) 876 5432",
        "15554567890",
        "555-111-2222 x104",
        "+1-555-333-4444",
        "5556667777",
        "N/A",
        "+1 (555) 999-0000",
    ] * (n_samples // 10 + 1)

    # Raw user search queries with special symbols and varying cases
    raw_queries = [
        "best Wireless Noise-Canceling Headphones under $200!!",
        "BUY Ergonomic Mechanical Keyboard (RGB) - SALE",
        "ultra-fast USB-C fast charging cable 10ft",
        "portable 4k OLED monitor for gaming & work",
        "smartwatch battery life waterproof IP68",
        "URGENT: laptop screen flickering issue replacement needed",
        "Bluetooth speaker bass boost outdoor heavy duty",
        "wireless mouse ergonomic silent click black",
        "HIGH PRIORITY: refund missing item package tracking #TRK-9872",
        "gaming headset microphone 7.1 surround sound",
    ] * (n_samples // 10 + 1)

    # Slice to exact n_samples
    df = pd.DataFrame({
        "ticket_id": [f"TCK-{1000 + i}" for i in range(n_samples)],
        "raw_timestamp": raw_timestamps[:n_samples],
        "customer_email": raw_emails[:n_samples],
        "phone_number": raw_phones[:n_samples],
        "user_query": raw_queries[:n_samples],
        "priority": np.random.choice(["Low", "Medium", "High", "Critical"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        "customer_rating": np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.1, 0.15, 0.4, 0.3]),
    })

    return df


# ------------------------------------------------------------------------------
# 2. ADVANCED DATETIME HANDLING, TIMEZONES & BUSINESS CALENDARS
# ------------------------------------------------------------------------------
def demonstrate_datetime_timezone_and_calendars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates heterogeneous timestamp parsing, UTC timezone localization,
    multiregional timezone conversion, custom business calendars, and timedelta calculations.

    Parameters:
        df (pd.DataFrame): Raw synthetic log DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame with parsed datetime features.
    """
    print("\n" + "=" * 80)
    print("2. ADVANCED DATETIME ANALYTICS, TIMEZONES & BUSINESS CALENDARS")
    print("=" * 80)

    transformed_df = df.copy()

    # --------------------------------------------------------------------------
    # 2.1 Robust Timestamp Parsing with `errors='coerce'` & `utc=True`
    # --------------------------------------------------------------------------
    # `errors='coerce'` converts invalid string logs into NaT (Not a Time)
    # `utc=True` standardizes timezone-naive and timezone-aware inputs into UTC
    transformed_df["parsed_utc_dt"] = pd.to_datetime(
        transformed_df["raw_timestamp"], format="ISO8601", errors="coerce", utc=True
    )

    print("--- 2.1 Raw Timestamp Parsing Output (First 6 Records) ---")
    print(transformed_df[["ticket_id", "raw_timestamp", "parsed_utc_dt"]].head(6))

    # Assert that invalid timestamp log was converted to NaT
    nat_count = transformed_df["parsed_utc_dt"].isna().sum()
    assert nat_count > 0, "Invalid timestamp strings should be coerced to NaT!"
    print(f"[✓] Successfully handled {nat_count} invalid/corrupted timestamp records (converted to NaT).")

    # --------------------------------------------------------------------------
    # 2.2 Timezone Conversions (US/Eastern & Asia/Kathmandu)
    # --------------------------------------------------------------------------
    # Convert UTC timestamps to local operating timezones
    transformed_df["dt_us_eastern"] = transformed_df["parsed_utc_dt"].dt.tz_convert("US/Eastern")
    transformed_df["dt_asia_kathmandu"] = transformed_df["parsed_utc_dt"].dt.tz_convert("Asia/Kathmandu")

    print("\n--- 2.2 Multi-Region Timezone Conversions ---")
    print(transformed_df[["parsed_utc_dt", "dt_us_eastern", "dt_asia_kathmandu"]].dropna().head(4))

    # --------------------------------------------------------------------------
    # 2.3 Temporal Feature Extraction (.dt accessor)
    # --------------------------------------------------------------------------
    transformed_df["year"] = transformed_df["parsed_utc_dt"].dt.year
    transformed_df["month_name"] = transformed_df["parsed_utc_dt"].dt.month_name()
    transformed_df["day_name"] = transformed_df["parsed_utc_dt"].dt.day_name()
    transformed_df["hour_utc"] = transformed_df["parsed_utc_dt"].dt.hour
    transformed_df["is_weekend"] = transformed_df["parsed_utc_dt"].dt.dayofweek >= 5

    # --------------------------------------------------------------------------
    # 2.4 Custom Business Calendar Offsets (3 Business Days SLA Target)
    # --------------------------------------------------------------------------
    # Apply 3 Business Days offset (`pd.offsets.BDay(3)`) to calculate SLA deadline
    transformed_df["sla_deadline_utc"] = transformed_df["parsed_utc_dt"] + pd.offsets.BDay(3)

    print("\n--- 2.4 SLA Deadline Calculation (+3 Business Days) ---")
    valid_sla = transformed_df[["parsed_utc_dt", "day_name", "sla_deadline_utc"]].dropna()
    print(valid_sla.head(4))

    # --------------------------------------------------------------------------
    # 2.5 Elapsed Delta Calculation (Timedelta from reference audit date)
    # --------------------------------------------------------------------------
    audit_reference_time = pd.to_datetime("2026-08-10 00:00:00", utc=True)
    transformed_df["elapsed_hours_to_audit"] = (
        (audit_reference_time - transformed_df["parsed_utc_dt"]).dt.total_seconds() / 3600.0
    )

    print("\n--- 2.5 Elapsed Time Delta to System Audit Date ---")
    print(transformed_df[["parsed_utc_dt", "elapsed_hours_to_audit"]].dropna().head(4))

    # Assert SLA calculation logic (ensure SLA deadline is strictly after ticket creation timestamp)
    valid_rows = transformed_df.dropna(subset=["parsed_utc_dt", "sla_deadline_utc"])
    assert (valid_rows["sla_deadline_utc"] > valid_rows["parsed_utc_dt"]).all(), "SLA deadline must be in the future!"

    print("\n[✓] Datetime parsing, timezone localization, and business calendar calculations completed.")
    return transformed_df


def main():
    """Runs Day 28 module demonstrations."""
    print("=" * 80)
    print("DAY 28: ADVANCED DATETIME HANDLING & VECTORIZED TEXT REGEX PIPELINES")
    print("=" * 80)

    # 1. Generate synthetic logs
    logs_df = generate_synthetic_ecom_logs(n_samples=50)
    print(f"\n[✓] Synthetic dataset generated with {len(logs_df)} records.")

    # 2. Advanced DateTime Analytics
    parsed_logs_df = demonstrate_datetime_timezone_and_calendars(logs_df)


if __name__ == "__main__":
    main()

