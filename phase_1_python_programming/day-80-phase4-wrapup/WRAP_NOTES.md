# Day 80 — Phase 4 wrap-up

There's no new topic today. One churn table goes through the Phase 4 habits in order. There are 900 customers with tenure, monthly spend, support calls, plan, and region. Churn was generated to rise with calls and fall with tenure. Plan matters and region does not. Spend is blank for 10.1% of rows: 21.3% of leavers but only 5.6% of stayers.

## The order things happen in

1. **Split first** into 540 train, 180 validation, and 180 test rows (Days 61, 65, 68).
2. **Fit the preprocessing on the 540 training rows only.** Blank spend gets the training median (56.05) plus a 0/1 missing flag (Day 64). Plan and region are one-hot encoded (Day 63), and every column is standardized with training means and stds (Day 62).
3. **Train a logistic model** with mini-batch Adam (batch 32, lr 0.005), weight decay on the slopes but not the intercept, and early stopping with patience 25 (Days 72–78).
4. **Pick the decay and the threshold on validation** (Days 66, 68).
5. **Score the test slice once.**
6. **Permutation importance** on the validation rows (Day 70).

## Decay search (validation log loss at each model's kept epoch)

| decay | kept epoch | train | validation |
|---:|---:|---:|---:|
| 0 | 21 | 0.4501 | 0.5039 |
| 0.003 | 21 | 0.4505 | 0.5033 |
| 0.01 | 29 | 0.4475 | 0.5017 |
| **0.03** | 38 | 0.4505 | **0.4977** |
| 0.1 | 35 | 0.4684 | 0.4982 |
| 0.3 | 35 | 0.5053 | 0.5158 |

Validation picks **0.03**, but the gaps are tiny. On test, decay 0.03 scored 0.4426 and decay 0 scored 0.4416. With 540 rows and 11 columns there isn't much room to overfit, so decay has little to fix. Day 78 was different: it had 42 columns on 36 rows.

![decay grid](output/decay_grid.png)

## Test slice, opened once

Log loss **0.4426**, against 0.5915 for always predicting the training churn rate. AUC **0.835**.

| rule | accuracy | precision | recall | F1 |
|---|---:|---:|---:|---:|
| always "stays" | 0.722 | 0 | 0 | 0 |
| model, threshold 0.5 | **0.794** | **0.760** | 0.380 | 0.507 |
| model, threshold 0.25 (picked on validation) | 0.739 | 0.519 | **0.800** | **0.630** |

At 0.5 the model catches only 38% of leavers. Lowering the cut-off to 0.25 catches 80% of them, at the cost of more false alarms. Accuracy drops from 0.794 to 0.739, so accuracy alone would have picked the worse rule for a churn team (Day 66).

## What the model leaned on

| column | rise in validation log loss when shuffled |
|---|---:|
| tenure | +0.0704 |
| calls | +0.0669 |
| spend | +0.0519 |
| plan | +0.0224 |
| region | −0.0008 |

Region does nothing, which matches how the data was built. Spend ranks higher than its small true slope would suggest. Shuffling the raw column moves the blank flag along with it, and blanks are almost four times more common among leavers.

![importance](output/importance.png)

## Early stopping on the picked model

![loss curves](output/loss_curves.png)

Both curves flatten by epoch 20. Validation bottoms out at epoch 38, and training stops 25 epochs later. The kept weights come from epoch 38, not from the last epoch.

## What this phase was

Days 61–79 were the pieces: compare against a baseline, scale, encode, fill, avoid leakage, handle imbalance, read learning curves, tune on validation, calibrate, measure importance, go multiclass, and then the optimizer days (early stopping, learning rate, mini-batches, momentum, Adam, schedules, weight decay, dropout). Today ran most of those pieces in order on one table. The one rule that held the rest together: every number the model learns from comes from the training rows, every choice comes from validation, and test is opened once.
