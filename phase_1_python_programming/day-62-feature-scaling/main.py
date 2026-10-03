"""Run the Day 62 feature-scaling walkthrough."""

import numpy as np

from scale_engine import (
    StandardScaler,
    compare_scalers,
    make_age_income,
)
from visualizer import ScalePlots


def run_pipeline():
    print("=" * 72)
    print("DAY 62: FEATURE SCALING")
    print("=" * 72)

    X, y, names = make_age_income(n=240, seed=62)
    print(f"\n1) raw ranges  (positive rate {y.mean():.3f})")
    for j, name in enumerate(names):
        col = X[:, j]
        print(f"   {name:<8} min={col.min():10.1f}  max={col.max():10.1f}  std={col.std():10.1f}")

    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    print("\n2) after standard scaling (fit on all rows just to peek)")
    for j, name in enumerate(names):
        print(f"   {name:<8} mean={Z[:, j].mean():7.3f}  std={Z[:, j].std():6.3f}")

    print("\n3) kNN on a holdout — scaler fit on train only")
    rows = compare_scalers(X, y, k=5, val_frac=0.3, seed=62)
    for row in rows:
        print("  ", row.summary())

    # distance check: one young low-income vs one old high-income, raw vs scaled
    a = np.array([[28.0, 40000.0]])
    b = np.array([[45.0, 70000.0]])
    raw_d = float(np.linalg.norm(a - b))
    za = scaler.transform(a)
    zb = scaler.transform(b)
    scaled_d = float(np.linalg.norm(za - zb))
    print(f"\n4) distance between two people")
    print(f"   raw units:    {raw_d:,.1f}   (basically just the income gap)")
    print(f"   scaled units: {scaled_d:.3f}")

    plots = ScalePlots()
    p1 = plots.before_after(X, Z, names)
    p2 = plots.accuracy_bars(rows, title="kNN validation accuracy by scaling")

    print("\n5) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 62 done.")


if __name__ == "__main__":
    run_pipeline()
