# Project 1 — Telco customer churn (Day 81: kickoff)

Phase 5 starts with a real dataset. Today sets up the question, the data, the split, the metric, and the bar to beat. There is no model yet.

## The question

A retention team can call **one customer in five** each month. If they call the riskiest fifth, what share of the customers who are about to leave do they reach?

- **Headline metric:** recall in the top 20% of the ranked call list. When scores tie at the cut-off, the remaining calls are shared across the tied group. A constant score therefore gets exactly 20%, and row order can't move the number.
- **Kept alongside:** AUC (ranking quality across every cut-off) and log loss (are the probabilities honest?).
- **Rule:** choices are made on validation. Test is opened once, at the end of the project.

## The data

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (Apache-2.0). `churn_data.fetch()` downloads it to `data/` (git-ignored) and refuses any file whose sha256 isn't `16320c9c…5e91`.

7043 customers, 21 columns: an ID, 16 categorical, 3 numeric, and `Churn` (left within the last month). The churn rate is **26.5%** (1869 customers).

### What cleaning changed

| change | why |
|---|---|
| `TotalCharges` blank → 0 for 11 rows | All 11 have tenure 0. They haven't been billed yet, so 0 is the real value, not a gap to fill. A blank on any other tenure raises an error. |
| `SeniorCitizen` 0/1 → No/Yes | It's a yes/no flag like `Partner` and `Dependents`. |
| `Churn` No/Yes → 0/1 | Target. |

After cleaning there are no missing values.

### What the audit found

- **Repeated facts.** "No internet service" in six add-on columns marks the same 1526 customers as `InternetService = No`, and "No phone service" mirrors `PhoneService = No`. One-hot encoding as-is would state each fact several times over.
- **`TotalCharges` ≈ tenure × `MonthlyCharges`.** Correlation 0.9996, median gap 2%. It adds almost nothing that those two columns don't already say.
- **Lopsided flags.** `PhoneService` is 90% Yes and `SeniorCitizen` is 84% No.

![churn profile](output/churn_profile.png)

| column | highest churn | lowest churn |
|---|---|---|
| Contract | month-to-month 42.7% | two year 2.8% |
| InternetService | fiber optic 41.9% | none 7.4% |
| PaymentMethod | electronic check 45.3% | credit card (automatic) 15.2% |
| tenure | 0–6 months 52.9% | 49–72 months 9.5% |

## The split (frozen)

Stratified on churn with seed 81, then written to **`split_ids.tsv`** (customerID → slice), which is committed. Later days read this file instead of re-splitting.

| slice | rows | churn |
|---|---:|---:|
| train | 4930 | 0.265 |
| validation | 1056 | 0.265 |
| test | 1057 | 0.266 |

## Baselines (fit on train, scored on validation)

Each baseline is a lookup table: the training churn rate of the customer's cell. The rate is shrunk toward the overall rate as if 20 average customers sat in every cell, so a tiny cell can't claim 0% or 100%.

On validation, 211 calls (20% of 1056) can reach at most 211 of the 280 leavers, so a perfect list scores **75.4%**.

| baseline | cells | top-20% recall | AUC | log loss |
|---|---:|---:|---:|---:|
| everyone at the train rate | 1 | 0.200 | 0.500 | 0.5784 |
| rate by contract | 3 | 0.324 | 0.743 | 0.4764 |
| rate by contract × tenure band | 15 | 0.446 | 0.794 | 0.4642 |
| rate by contract × tenure band × internet | 43 | **0.503** | **0.858** | **0.4094** |

![gains](output/gains.png)

Three columns and some counting already reach half the leavers with a fifth of the calls. **That's the bar:** the first real model has to beat 0.503 top-20% recall on validation, or it isn't earning its complexity.

## Files

| file | what it does |
|---|---|
| `churn_data.py` | fetch with hash check, load as text, clean |
| `audit_engine.py` | column audit, churn by level and tenure band, mirrored levels, charges check |
| `split_engine.py` | stratified split, write and load `split_ids.tsv` |
| `baseline_engine.py` | top-20% recall, AUC, log loss, shrunk lookup tables |
| `visualizer.py` | churn profile and gains plots |
| `main.py` | the walkthrough (`python3 main.py`) |
| `test_kickoff.py` | 15 tests; the two real-data tests skip until the CSV is cached |
