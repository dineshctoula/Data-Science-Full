"""Run the Day 60 phase-3 wrap-up."""

import numpy as np

from wrap_engine import (
    bootstrap_slope,
    column_summary,
    fit_ols,
    make_exam_scores,
    mse,
    r2_score,
    train_test_split,
)
from visualizer import WrapPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 60: PHASE 3 WRAP-UP")
    print("=" * 72)

    X, y, names = make_exam_scores(n=180, seed=60)
    print("\n1) what the columns look like")
    for row in column_summary(X, y, names):
        print(
            f"   {row['name']:<6} mean={row['mean']:6.2f}  "
            f"std={row['std']:5.2f}  corr={row['corr_with_y']:.3f}"
        )
    print(f"   score  mean={y.mean():6.2f}  std={y.std(ddof=1):5.2f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.25, seed=60)
    fit = fit_ols(Xtr, ytr, names)
    pred = fit.predict(Xte)
    print("\n2) ols on a 25% holdout")
    print("  ", fit.summary())
    print("   test MSE:", round(mse(yte, pred), 2))
    print("   test R²: ", round(r2_score(yte, pred), 3))
    # predicting the training mean is the boring baseline
    baseline = mse(yte, np.full_like(yte, ytr.mean()))
    print("   mean-only MSE:", round(baseline, 2))

    print("\n3) bootstrap the hours slope (full sample)")
    boot = bootstrap_slope(X, y, feature=0, n_boot=300, seed=60)
    print(f"   mean={boot['mean']:.2f}   95% interval [{boot['lo']:.2f}, {boot['hi']:.2f}]")
    print("   (the data was built with a true hours slope of 4.2)")

    plots = WrapPlots()
    p1 = plots.hours_vs_score(X, y, names)
    p2 = plots.residuals(yte, pred, title="holdout residuals")
    p3 = plots.slope_hist(boot["samples"], boot["lo"], boot["hi"])

    print("\n4) plots saved")
    for p in (p1, p2, p3):
        print("  ", p)
    print("\nDay 60 done. Phase 3 closes here.")


if __name__ == "__main__":
    run_pipeline()
