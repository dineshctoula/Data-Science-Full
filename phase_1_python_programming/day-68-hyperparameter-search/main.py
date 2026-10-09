"""Run the Day 68 hyperparameter search."""

from search_engine import choose_by_val, make_curve, run_search
from visualizer import SearchPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 68: HYPERPARAMETER SEARCH")
    print("=" * 72)

    x, y = make_curve(n=280, noise=0.35, seed=68)
    ks = [1, 3, 5, 9, 15, 25, 41]
    trials = run_search(x, y, ks=ks, seed=68)
    by_train = min(trials, key=lambda trial: (trial.train_mse, trial.k))
    by_val = choose_by_val(trials)
    by_test = min(trials, key=lambda trial: (trial.test_mse, trial.k))

    print(f"\n1) {len(x)} rows, half train, quarter val, quarter test")
    print("   k grid:", ks)

    print(f"\n2) {'k':>4}  {'train':>8}  {'val':>8}  {'test':>8}")
    for trial in trials:
        print(f"   {trial.k:4d}  {trial.train_mse:8.3f}  {trial.val_mse:8.3f}  {trial.test_mse:8.3f}")

    print("\n3) who each column would hire, and that k's test error")
    print(f"   train column wants k={by_train.k}, test MSE={by_train.test_mse:.3f}")
    print(f"   val column wants   k={by_val.k}, test MSE={by_val.test_mse:.3f}")
    print(f"   peeking at test    k={by_test.k}, test MSE={by_test.test_mse:.3f}")

    plots = SearchPlots()
    p1 = plots.mse_lines(trials, title="seed 68 — train error loves k = 1")
    p2 = plots.pick_bars(
        [f"train k={by_train.k}", f"val k={by_val.k}", f"test k={by_test.k}"],
        [by_train.test_mse, by_val.test_mse, by_test.test_mse],
        title="test error after the pick",
    )

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 68 done.")


if __name__ == "__main__":
    run_pipeline()
