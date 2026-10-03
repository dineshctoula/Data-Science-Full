# Day 62 — Feature scaling

Age and income both drive the label, but income is in the tens of thousands.
kNN uses raw Euclidean distance, so without scaling it mostly sees income.

Scalers are fit on the training rows only. The peek in section 2 of `main.py`
fits on everything just to print means — the accuracy comparison does not.

## Ranges (seed=62, n=240)

| | min | max | std |
|---|---:|---:|---:|
| age | 17 | 60 | 7.5 |
| income | 21,779 | 84,590 | 11,885 |

After standard scaling both columns have mean 0 and std 1.

## kNN validation accuracy (k=5)

| scaling | val acc |
|---|---:|
| raw | 0.583 |
| standard | 0.639 |
| minmax | 0.639 |

Not a huge jump on this seed, but raw is the worst. Standard and minmax tied.

Distance between a 28-year-old earning 40k and a 45-year-old earning 70k:
raw **30,000** (the income gap), scaled **3.40**.

## Takeaways

- Scale when the model uses distances or gradients.
- Fit the scaler on train. Transform validation with those same numbers.
- A constant column is left alone instead of dividing by zero.
