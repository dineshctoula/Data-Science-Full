# Day 47 — ROC Curves & Classification Thresholds

## Deliverables

- `roc_engine.py`: per-threshold TPR/FPR/precision/recall/F1, ROC/PR sweeps, trapezoidal AUC, Youden/F1 threshold selection, and scored-label datasets.
- `visualizer.py`: ROC curve, precision–recall curve, metric-vs-threshold overlay, and AUC comparison bars.
- `main.py`: a runnable lesson that compares rankings, chooses thresholds, and writes the charts.
- Two test modules covering AUC extremes, metric identities, selectors, and plot output.

## Key results

| Check | Result |
| --- | ---: |
| Useful ranking AUC | 0.993 |
| Random scores AUC | 0.499 |
| Threshold 0.5 | TPR 0.960, FPR 0.050, F1 0.955 |
| Youden best | t ≈ 0.524, J = 0.920 |
| F1 best | t ≈ 0.504, F1 = 0.960 |
| Perfect tiny ranking | AUC = 1.000 |

## Reading the formulas

The comments in `roc_engine.py` mark the ideas to look for:

1. Raising the threshold usually lowers both TPR and FPR; the ROC curve traces that path.
2. AUC summarizes ranking quality without committing to one cutoff.
3. Youden's J = TPR − FPR picks a balanced operating point on the ROC curve.
4. F1 balances precision and recall when class costs are similar for false positives and false negatives.
5. A random scorer sits near the diagonal with AUC ≈ 0.5.

## Practical checks

- Labels must be binary and include both classes.
- Extreme thresholds that classify everything as 0 or 1 give the ROC endpoints.
- AUC integrates TPR against FPR after sorting by FPR.
- Charts require a `CurveResult` so plot helpers stay aligned with the engine.

## Run

```bash
cd phase_1_python_programming/day-47-roc-curves-thresholds
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/roc_curve.png`, `output/precision_recall_curve.png`, `output/threshold_tradeoff.png`, and `output/auc_comparison.png`.
