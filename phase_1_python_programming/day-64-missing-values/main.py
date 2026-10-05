"""Run the Day 64 missing-values walkthrough."""

import numpy as np

from missing_engine import (
    SimpleImputer,
    compare_impute,
    make_study_table,
    missing_report,
    train_val_split,
)
from visualizer import MissingPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 64: MISSING VALUES")
    print("=" * 72)

    X, y, names = make_study_table(n=240, seed=64)
    print("\n1) where the blanks are")
    for row in missing_report(X, names):
        print(f"   {row['name']:<8} missing={row['n_missing']:3d}  rate={row['rate']:.3f}")

    # do low scores go blank more often? quick check, not a formal test
    hours_missing = np.isnan(X[:, 0])
    print(f"\n   mean score when hours is blank:  {y[hours_missing].mean():.1f}")
    print(f"   mean score when hours is filled: {y[~hours_missing].mean():.1f}")

    Xtr, Xva, ytr, yva = train_val_split(X, y, seed=64)
    imp = SimpleImputer(strategy="mean").fit(Xtr)
    print("\n2) train-only fill values")
    for name, value in zip(names, imp.fill_):
        print(f"   {name:<8} {value:.2f}")
    # the holdout mean of observed hours, just to show we didn't use it
    observed_val = Xva[:, 0][~np.isnan(Xva[:, 0])]
    print(f"   (holdout hours mean, not used: {observed_val.mean():.2f})")

    print("\n3) same model, different treatment of blanks")
    rows = compare_impute(X, y, seed=64)
    for row in rows:
        print("  ", row.summary())

    plots = MissingPlots()
    p1 = plots.missing_bars(missing_report(X, names))
    p2 = plots.mse_bars(rows)

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 64 done.")


if __name__ == "__main__":
    run_pipeline()
