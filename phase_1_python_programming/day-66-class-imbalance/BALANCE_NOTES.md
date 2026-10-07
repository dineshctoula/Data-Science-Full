# Day 66 — Class imbalance

800 rows, class 1 shows up **5.9%** of the time. Column 0 is only a little shifted for that class. Column 1 is noise. Same 30% holdout for every model (20 positives in the test slice, 27 in train).

| | accuracy | precision | recall | f1 | caught | false alarms |
|---|---:|---:|---:|---:|---:|---:|
| always class 0 | 0.917 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| plain logistic | 0.917 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| class weight 12 | 0.804 | 0.186 | 0.400 | 0.254 | 8 | 35 |
| oversample train | 0.654 | 0.129 | 0.550 | 0.210 | 11 | 74 |

The plain fit matches the majority guess. It never calls class 1, so the accuracy number is the base rate.

Weighting the rare class (gradient multiplied by 12) catches 8 of 20 and raises 35 false alarms. Copying the minority rows until the training counts match (1066 rows) catches 11 of 20 and raises 74 false alarms. Both lose to "always class 0" on accuracy. Oversampling is not the winner here; it just hunts harder.

## Takeaway

If missing a rare row is the actual cost, accuracy is the wrong headline. Recall and the false-alarm count say what the model did. Balance the training rows only. The test slice should keep the real mix.
