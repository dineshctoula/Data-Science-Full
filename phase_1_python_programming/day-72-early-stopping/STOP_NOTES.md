# Day 72 — Early stopping

220 rows, 22 columns. Columns 0 and 1 make the label. The other 20 are noise. Logistic regression, full batch, learning rate 0.25, 200 epochs. 70 rows train, 60 pick the epoch, 90 are the test slice. Column scales come from the training rows.

| epoch | train log loss | validation log loss |
|---:|---:|---:|
| 1 | 0.651 | 0.670 |
| 5 | 0.541 | 0.613 |
| 10 | 0.467 | 0.584 |
| 20 | 0.394 | 0.572 |
| 50 | 0.311 | 0.603 |
| 100 | 0.260 | 0.677 |
| 200 | 0.215 | 0.824 |

Validation bottoms at epoch **20** (0.572). Training loss is still falling at epoch 200 (0.215) while validation has climbed to **0.824**.

| | test log loss | test accuracy |
|---|---:|---:|
| epoch 20 | 0.562 | 0.678 |
| epoch 200 | 0.828 | 0.667 |

Accuracy barely moves, 0.678 to 0.667. The log loss does not: 0.562 versus 0.828. The late model is more sure and more wrong.

## Takeaway

Keep the weights from the best validation epoch. The run here goes all the way to 200 so the curve is visible. The saved model is epoch 20. Accuracy can look fine either way. Watch the loss.
