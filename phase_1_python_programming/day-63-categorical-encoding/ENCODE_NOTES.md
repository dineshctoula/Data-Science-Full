# Day 63 — Categorical encoding

Plan is ordered (basic < plus < pro). Channel is not. Both were fit on the
training rows only.

## Mean scores (seed=63)

| plan | mean | channel | mean |
|---|---:|---|---:|
| basic | 55.6 | email | 56.0 |
| plus | 64.5 | ads | 61.6 |
| pro | 74.1 | referral | 67.7 |

`enterprise` was not in training, so the ordinal encoder returns **-1**.
`sms` was not a channel, so its one-hot row is **[0, 0, 0]**.

## Same holdout

| encoding | MSE | R² |
|---|---:|---:|
| ordinal plan (basic=0 … pro=2) | 36.55 | 0.488 |
| flipped plan (pro=0 … basic=2) | 36.55 | 0.488 |
| one-hot plan + channel | 15.21 | 0.787 |

The flipped codes match the original fit. They are just `2 - code`, and least
squares flips the slope to compensate. One-hot wins because the channel is in
the model, and channel has no order to encode as 0, 1, 2.

## Takeaways

- Ordinal codes when the order is real and the steps are roughly even.
- One-hot when the labels are just names.
- Don't add a column for a category you only saw in validation.
