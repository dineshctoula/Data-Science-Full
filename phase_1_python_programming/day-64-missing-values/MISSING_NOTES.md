# Day 64 — Missing values

Hours was left blank more often for lower scores. Sleep is complete.
Fill values are computed on the training rows only.

## Blanks (seed=64, n=240)

| column | missing | rate |
|---|---:|---:|
| hours | 124 | 0.517 |
| sleep | 0 | 0 |

Mean score when hours is blank: **65.3**. When it's filled: **73.1**.
Train mean used to fill hours: **5.54**. The holdout's own mean was 5.71 and was not used.

## Holdout

| treatment | rows kept for training | MSE |
|---|---:|---:|
| drop incomplete rows | 84 | 15.52 |
| mean fill | 180 | 45.49 |
| median fill | 180 | 45.51 |
| mean fill + missing indicator | 180 | 33.58 |

Drop looks best and isn't a fair win. Its MSE is only on the validation rows that still have hours, which are the higher scores and the easier ones. Mean and median are almost the same. Adding a 0/1 "was blank" column cuts the error from 45.5 to 33.6, because the blank itself is a clue.

## Takeaways

- Don't fill with zero unless zero is a real answer.
- Fit the fill on train.
- If blanks track the outcome, keep an indicator. A mean hides that.
- Compare drop-rows on the same rows as the other methods before declaring it the winner. This script doesn't.
