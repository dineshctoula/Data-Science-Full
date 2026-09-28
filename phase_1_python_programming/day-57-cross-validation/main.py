"""Run the Day 57 cross-validation walkthrough."""

import numpy as np

from cv_engine import cross_val_accuracy, make_two_class_cloud, repeated_holdout
from visualizer import CVPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 57: CROSS-VALIDATION")
    print("=" * 72)

    X, y = make_two_class_cloud(n_per=50, sep=1.5, seed=57)

    print("\n1) one 5-fold stratified run")
    cv = cross_val_accuracy(X, y, k=5, stratified=True, seed=57)
    print("  ", cv.summary())
    for i, s in enumerate(cv.scores, start=1):
        print(f"   fold {i}: {s:.3f}")

    print("\n2) plain k-fold (no class balancing) on the same data")
    plain = cross_val_accuracy(X, y, k=5, stratified=False, seed=57)
    print("  ", plain.summary())

    print("\n3) eight random 30% holdouts — watch them jump around")
    holdouts = repeated_holdout(X, y, n_repeats=8, test_frac=0.3, seed=57)
    print("   holdouts:", np.round(holdouts, 3))
    print(f"   holdout mean={holdouts.mean():.3f}  std={holdouts.std(ddof=1):.3f}")
    print(f"   cv      mean={cv.mean:.3f}  std={cv.std:.3f}")

    print("\n4) does k change the story much?")
    ks, means, stds = [], [], []
    for k in (2, 3, 5, 8, 10):
        r = cross_val_accuracy(X, y, k=k, stratified=True, seed=57)
        ks.append(k)
        means.append(r.mean)
        stds.append(r.std)
        print(f"   k={k:2d}: {r.summary()}")

    plots = CVPlots()
    p1 = plots.fold_bars(cv.scores, title="stratified 5-fold")
    p2 = plots.holdout_vs_cv(holdouts, cv.scores, title="holdout repeats vs 5-fold band")
    p3 = plots.k_sweep(ks, means, stds, title="stratified CV vs k")

    print("\n5) plots saved")
    for p in (p1, p2, p3):
        print("  ", p)
    print("\nDay 57 done.")


if __name__ == "__main__":
    run_pipeline()
