"""Run the Day 66 class-imbalance walkthrough."""

from balance_engine import holdout_compare, make_rare_events
from visualizer import BalancePlots


def run_pipeline():
    print("=" * 72)
    print("DAY 66: CLASS IMBALANCE")
    print("=" * 72)

    X, y = make_rare_events(n=800, rate=0.06, seed=66)
    print(f"\n1) table {X.shape}, class-1 rate {y.mean():.3f}")
    print("   column 0 is a weak signal, column 1 is junk")

    result = holdout_compare(X, y, seed=66, pos_weight=12.0)
    print(f"\n2) holdout positives: train {result['train_pos']}, test {result['test_pos']}")
    print(f"   oversampled training rows: {result['balanced_rows']}")

    order = ["majority", "plain", "weighted", "oversample"]
    print("\n3) same test rows, four ways of calling class 1")
    for name in order:
        print(f"   {name:11} {result[name].summary()}")

    plots = BalancePlots()
    p1 = plots.metric_bars(
        order,
        [result[name].accuracy for name in order],
        [result[name].recall for name in order],
        title="seed 66 — accuracy vs recall",
    )
    p2 = plots.catch_bars(
        order,
        [result[name].tp for name in order],
        [result[name].fp for name in order],
        title="seed 66 — what you catch, and what you flag by mistake",
    )

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 66 done.")


if __name__ == "__main__":
    run_pipeline()
