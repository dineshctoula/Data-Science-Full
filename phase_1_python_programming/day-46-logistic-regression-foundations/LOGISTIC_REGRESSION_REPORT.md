# Day 46 — Logistic Regression Foundations

## Deliverables

- `logistic_engine.py`: sigmoid/logit, binary cross-entropy, batch-gradient logistic fits, accuracy/precision/recall, confusion matrices, and exam-pass datasets.
- `visualizer.py`: sigmoid curve, one-feature probability fit, BCE loss history, and annotated confusion matrix.
- `main.py`: a runnable lesson that fits models, scores new students, and writes the charts.
- Two test modules covering identities, training progress, metrics, and plot output.

## Key results

| Check | Result |
| --- | ---: |
| σ(0) | 0.5 |
| One-feature hours model | accuracy 0.810, loss 0.693 → 0.318 |
| Two-feature hours + GPA | accuracy 0.825, precision 0.867, recall 0.914 |
| Slopes | study_hours ≈ +0.56, gpa ≈ +0.76 |
| Confusion matrix | [[14, 13], [8, 85]] |

## Reading the formulas

The comments in `logistic_engine.py` mark the ideas to look for:

1. The sigmoid squashes any real score into a probability between 0 and 1.
2. Binary cross-entropy punishes confident wrong probabilities more than mild mistakes.
3. The logistic gradient is `Xᵀ (p − y) / n` — the same design matrix idea as linear regression, with probabilities instead of fitted values.
4. A threshold of 0.5 turns probabilities into hard pass/fail labels.
5. Precision asks “of predicted passes, how many were real?”; recall asks “of real passes, how many did we catch?”

## Practical checks

- Labels must be binary 0/1 and include both classes.
- Logit rejects probabilities of exactly 0 or 1.
- Learning rate and iterations must be positive.
- Confusion-matrix plots expect a 2×2 array in `[[TN, FP], [FN, TP]]` order.

## Run

```bash
cd phase_1_python_programming/day-46-logistic-regression-foundations
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/sigmoid_curve.png`, `output/probability_curve.png`, `output/loss_history.png`, and `output/confusion_matrix.png`.
