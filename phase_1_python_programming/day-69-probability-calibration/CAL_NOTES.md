# Day 69 — Probability calibration

600 rows, class 1 rate 0.460. Logistic fit on 300 training rows. Temperature 0.5 divides the logit, which pushes probabilities toward 0 and 1. A second logistic (Platt) is fit on the 150 validation rows only, then applied to the 150 test rows.

| | accuracy | Brier | ECE |
|---|---:|---:|---:|
| raw | 0.780 | 0.160 | 0.045 |
| sharpened | 0.780 | 0.174 | 0.117 |
| Platt on the sharpened scores | 0.767 | 0.161 | 0.059 |

Accuracy did not move when the probabilities were sharpened. The 0.5 calls are the same. Brier went from **0.160** to **0.174** and the calibration error from **0.045** to **0.117**.

The validation fit came back with intercept **-0.071** and slope **0.503**, close to the temperature that caused the damage. On the test slice Brier is back to **0.161** and ECE to **0.059**. Accuracy dipped to **0.767** because a few probabilities crossed 0.5. Calibration is not the accuracy number.

Sharpened bins on the test slice, the two ends:

- mean probability 0.037, but the event happened 0.105 of the time (57 rows)
- mean probability 0.968, but it happened 0.848 of the time (33 rows)

The middle bins are small (one of them has 4 rows), so those frequencies jump around.

## Takeaway

A probability can be useful as a ranking and still be a bad frequency. Check a reliability bin, not just accuracy. Fit the repair on the validation rows.
