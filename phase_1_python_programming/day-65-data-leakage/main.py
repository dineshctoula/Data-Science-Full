"""Run the Day 65 leakage walkthrough."""

import numpy as np

from leak_engine import compare_selection, make_wide_table, repeat_gap
from visualizer import LeakPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 65: PREP-STEP LEAKAGE")
    print("=" * 72)

    X, y = make_wide_table(n=80, n_noise=40, seed=65)
    print(f"\n1) table shape {X.shape} — columns 0 and 1 are real, the rest are noise")

    one = compare_selection(X, y, k=3, seed=65)
    print("\n2) one split, pick the 3 columns most correlated with y")
    print("  ", one.summary())
    print("   honest columns:", one.honest_cols)
    print("   leaky columns: ", one.leaky_cols)

    print("\n3) same trick over 24 seeds")
    gaps = repeat_gap(
        lambda s: make_wide_table(n=80, n_noise=40, seed=s),
        n_seeds=24,
        k=3,
        base_seed=65,
    )
    print(f"   mean gap (honest − leaky): {gaps.mean():.3f}")
    print(f"   seeds where the leak looked better: {int(np.sum(gaps > 0))}/{len(gaps)}")
    print(f"   median gap: {np.median(gaps):.3f}")

    plots = LeakPlots()
    p1 = plots.one_seed_bars(one.honest_mse, one.leaky_mse, title="seed 65")
    p2 = plots.gap_hist(gaps, title="how much the leaked test error flatters you")

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 65 done.")


if __name__ == "__main__":
    run_pipeline()
