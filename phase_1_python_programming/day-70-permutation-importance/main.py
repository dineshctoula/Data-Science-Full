"""Run the Day 70 permutation-importance walkthrough."""

from perm_engine import make_table, permutation_rises, split_indices
from visualizer import PermPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 70: PERMUTATION IMPORTANCE")
    print("=" * 72)

    X, y = make_table(n=240, seed=70)
    train, val = split_indices(len(X), test_frac=0.3, seed=70)
    base, coef, rises = permutation_rises(X[train], y[train], X[val], y[val], repeats=20, seed=70)

    print(f"\n1) table {X.shape}. columns 0 and 1 drive y, 2 and 3 are noise")
    print(f"   train {len(train)}, validation {len(val)}")
    print(f"   validation MSE before any shuffle: {base:.3f}")

    print("\n2) fitted slopes (intercept first)")
    print("  ", " ".join(f"{value:.3f}" for value in coef))

    print("\n3) mean rise in validation MSE over 20 shuffles")
    for item in rises:
        print(f"   x{item.index}  rise={item.mean_rise:.3f}  std={item.std_rise:.3f}")

    plots = PermPlots()
    p1 = plots.rise_bars(rises, title="seed 70 — shuffle damage on the validation rows")
    p2 = plots.coef_bars(coef, title="seed 70 — OLS slopes")

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 70 done.")


if __name__ == "__main__":
    run_pipeline()
