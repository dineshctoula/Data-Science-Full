# 🛡️ Day 34 Automated Data Quality Audit & Statistical Drift Report

## 📌 Executive Summary
This report details the execution of an **Automated Data Quality & Statistical Drift Engine** across an incoming multi-source transactional stream (**100,000 raw records**). 
The multi-stage ETL pipeline identified schema irregularities, missing values, range constraint violations, and distribution drift, automatically remediating anomalies before exporting a validated dataset to the **Parquet Data Mart** and **DuckDB Analytical Staging Database**.

---

## ⏱️ Pipeline Execution Performance
- **Pipeline Processing Time**: `1.4669 seconds`
- **Baseline Dataset Volume**: `100,000 rows`
- **Incoming Raw Volume**: `100,000 rows`
- **Curated Data Mart Volume**: `100,000 rows`

---

## 🚨 Audit Issues & Anomaly Detection Summary

| Severity | Check Type | Target Column | Findings & Description | Affected Records |
| :--- | :--- | :--- | :--- | :---: |
| 🟡 `WARNING` | `NULL_THRESHOLD` | `user_id` | Null ratio (3.00%) exceeded maximum threshold (1.00%) | 3,000 |
| 🟡 `WARNING` | `NULL_THRESHOLD` | `amount` | Null ratio (1.99%) exceeded maximum threshold (0.50%) | 1,995 |
| 🔴 `CRITICAL` | `RANGE_BOUND` | `amount` | Found 150 invalid negative transaction amounts | 150 |
| 🟡 `WARNING` | `RANGE_BOUND` | `amount` | Found 50 transaction amounts exceeding extreme boundary ($20,000.00) | 50 |
| 🟡 `WARNING` | `DOMAIN_CATEGORY` | `category` | Found 18543 non-standard category values (e.g. ['groceries ', '  Electronics ', '  Home']) | 18,543 |
| 🔴 `CRITICAL` | `DRIFT` | `amount` | Significant data drift detected (PSI = 2.4283 >= 0.25, Mean Shift = +2.68σ) | 0 |
| 🔴 `CRITICAL` | `DRIFT` | `risk_score` | Significant data drift detected (PSI = 1.1332 >= 0.25, Mean Shift = +1.30σ) | 0 |


---

## 📈 Statistical Data Drift Analysis (Population Stability Index - PSI)

Statistical drift was evaluated between historical baseline distributions and current incoming raw batches:

| Feature Name | PSI Value | Status | Interpretation | Baseline Mean | Incoming Mean |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **`amount`** | `2.4283` | 🔴 `CRITICAL DRIFT` | Significant upward distribution shift | `$90.27` | `$242.63` |
| **`risk_score`** | `1.1332` | 🔴 `CRITICAL DRIFT` | Significant increase in transaction risk | `0.1666` | `0.3002` |

> [!IMPORTANT]
> **PSI Interpretation Standard:**
> - **PSI < 0.10**: Stable distribution (no action required).
> - **0.10 ≤ PSI < 0.25**: Moderate drift (monitor metric closely).
> - **PSI ≥ 0.25**: Significant drift (retrain downstream models & trigger data team alerts).

---

## 🛠️ Automated Anomaly Remediation & Pipeline Transformation
1. **String Normalization**: Trimmed whitespace and applied title-case formatting to `category`. Unrecognized values remapped to `"Other"`.
2. **Invalid Amount Handling**: Negative amounts converted to `null` and imputed using category-level medians.
3. **Missing Key Recovery**: Imputed null `user_id` values with fallback identifier (`-1`) and matched against customer profiles.
4. **Relational Enrichment**: Joined customer profiles (`email`, `membership_tier`) via left join.
5. **Feature Engineering**: Engineered `processing_fee` (`amount * 0.025`), mapped `risk_tier` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and computed relative spending ratio (`amount_to_cat_avg_ratio`).

---

## 📊 Curated Data Mart Analytics (DuckDB Queries)

### 1. Revenue & Fee Performance by Membership Tier
| membership_tier | total_transactions | avg_transaction_val | total_revenue | total_fees | avg_risk_score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard | 48603 | 239.31 | 11631207.68 | 290779.57 | 0.2995 |
| Silver | 29778 | 245.95 | 7323814.0 | 183094.44 | 0.3007 |
| Gold | 14092 | 245.39 | 3458094.69 | 86451.8 | 0.3011 |
| Platinum | 4527 | 232.5 | 1052516.88 | 26312.57 | 0.3025 |
| Guest | 3000 | 264.52 | 793564.13 | 19838.85 | 0.2985 |

### 2. Transaction Risk Tier Distribution
| risk_tier | transaction_count | pct_of_total |
| :--- | :---: | :---: |
| MEDIUM | 64952 | 64.95 |
| LOW | 26052 | 26.05 |
| HIGH | 8876 | 8.88 |
| CRITICAL | 120 | 0.12 |

---

## 💡 Key Architectural Takeaways
1. **Automated Auditing**: Proactive detection of schema shifts and distribution drift prevents corrupted data from polluting production data warehouses.
2. **Statistical Stability Index**: PSI provides mathematically rigorous drift tracking independently of sample sizes.
3. **Hybrid Engine Synergy**: Leveraging **Polars** for eager/lazy vector transformations and **DuckDB** for analytical SQL staging ensures sub-second throughput on enterprise datasets.
