# Day 48 — Naive Bayes Classification

## Deliverables

- `naive_bayes_engine.py`: Laplace-smoothed priors, per-feature Gaussian likelihoods, log-joint posteriors, MAP prediction, confusion matrices, and flower/exam datasets.
- `visualizer.py`: class-prior bars, class-conditional density overlays, 2-D decision regions, and confusion-matrix charts.
- `main.py`: a runnable lesson that fits models, scores new students, and writes the charts.
- Two test modules covering density identities, accuracy, posterior normalization, and plot output.

## Key results

| Check | Result |
| --- | ---: |
| Gaussian density peaks at μ | N(1\|1,1) > N(0\|1,1) |
| Three-class flower model | accuracy = 1.000, priors ≈ 1/3 |
| Flower confusion | diagonal [[40,0,0],[0,40,0],[0,0,40]] |
| Exam pass/fail model | accuracy = 0.992 |
| Pass vs fail mean study hours | 8.41 vs 3.24 |

## Reading the formulas

The comments in `naive_bayes_engine.py` mark the ideas to look for:

1. Bayes' rule: posterior ∝ prior × likelihood.
2. The naive assumption factors the likelihood across features.
3. Log-space products become sums, which avoids underflow.
4. MAP prediction chooses the class with the largest posterior.
5. Laplace smoothing keeps priors away from exact zero.

## Practical checks

- Training labels must include at least two classes.
- Feature variances are floored with `var_smoothing` so Normal densities stay defined.
- Posterior rows always sum to 1 after log-sum-exp normalization.
- Decision-region plots require exactly two features.

## Run

```bash
cd phase_1_python_programming/day-48-naive-bayes-classification
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/class_priors.png`, `output/feature_likelihoods.png`, `output/decision_regions.png`, and `output/confusion_matrix.png`.
