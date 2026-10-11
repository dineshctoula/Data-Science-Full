# Day 71 — Multiclass

270 rows, three classes, 90 each. The class centers sit on a triangle and the clouds overlap (separation 1.4, noise 1.1). One logistic per class, fit on the training rows only. A test row goes to the class whose model gives the highest probability.

| | accuracy |
|---|---:|
| majority class | 0.321 |
| one-vs-rest | 0.753 |

Confusion on the test slice. Rows are the actual class, columns are the guess.

| actual | predicted 0 | predicted 1 | predicted 2 |
|---|---:|---:|---:|
| 0 | 20 | 2 | 4 |
| 1 | 2 | 22 | 3 |
| 2 | 6 | 3 | 19 |

Recall: class 0 **0.769**, class 1 **0.815**, class 2 **0.679**.

Most of the swaps are between class 0 and class 2 (4 one way, 6 the other). Class 1 is the cleanest of the three. Accuracy hides that.

## Takeaway

With more than two labels, fit one yes/no model per class and keep the highest score. Read the confusion matrix. The class you miss is not always the one the accuracy number is talking about.
